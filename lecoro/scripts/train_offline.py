import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat

import hydra
import torch
from deepdiff import DeepDiff
from hydra_zen import instantiate
from lerobot.scripts.train import log_train_info, log_eval_info, format_big_number, eval_policy
from omegaconf import OmegaConf
from termcolor import colored

from coro.common.algo.algo import Algo
from coro.common.config_gen import Config, register_configs
from coro.common.dataset.lecoro_dataset import LeCoroDataset
from coro.common.logger import Logger
from coro.common.utils.dataset_utils import MultiLeRobotDataset, cycle
from coro.common.utils.hydra_utils import init_hydra_config
from coro.common.utils.log_utils import init_logging, log_output_dir
from coro.common.utils.obs_utils import initialize_obs_utils_with_config
from coro.common.utils.script_utils import set_global_seed
from coro.common.utils.tensor_utils import to_device
from coro.common.utils.torch_utils import get_safe_torch_device

register_configs()


def train(cfg: Config, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    if cfg.resume:
        if not Logger.get_last_checkpoint_dir(out_dir).exists():
            raise RuntimeError(
                "You have set resume=True, but there is no model checkpoint in "
                f"{Logger.get_last_checkpoint_dir(out_dir)}"
            )
        checkpoint_cfg_path = str(Logger.get_last_pretrained_model_dir(out_dir) / "config.yaml")
        logging.info(
            colored(
                "You have set resume=True, indicating that you wish to resume a run",
                color="yellow",
                attrs=["bold"],
            )
        )
        # Get the configuration file from the last checkpoint.
        checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)

        diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))
        # Ignore the `resume` and parameters.
        if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
            del diff["values_changed"]["root['resume']"]
        # Log a warning about differences between the checkpoint configuration and the provided
        # configuration.
        if len(diff) > 0:
            logging.warning(
                "At least one difference was detected between the checkpoint configuration and "
                f"the provided configuration: \n{pformat(diff)}\nNote that the checkpoint configuration "
                "takes precedence.",
            )
        # Use the checkpoint config instead of the provided config (but keep `resume` parameter).
        cfg = checkpoint_cfg
        cfg.resume = True
    elif Logger.get_last_checkpoint_dir(out_dir).exists():
        raise RuntimeError(
            f"The configured output directory {Logger.get_last_checkpoint_dir(out_dir)} already exists. If "
            "you meant to resume training, please use `resume=true` in your command or yaml configuration."
        )

    if cfg.training.eval.batch_size > cfg.training.eval.n_episodes:
        raise ValueError(
            "The eval batch size is greater than the number of eval episodes "
            f"({cfg.training.eval.batch_size} > {cfg.eval.n_episodes}). As a result, {cfg.training.eval.batch_size} "
            f"eval environments will be instantiated, but only {cfg.training.eval.n_episodes} will be used. "
            "This might significantly slow down evaluation. To fix this, you should update your command "
            f"to increase the number of episodes to match the batch size (e.g. `training.eval.n_episodes={cfg.training.eval.batch_size}`), "
            f"or lower the batch size (e.g. `training.eval.batch_size={cfg.training.eval.n_episodes}`)."
        )

    # log metrics to terminal and wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)
    obs_config = instantiate(cfg.observation)
    initialize_obs_utils_with_config(obs_config)

    # Check device is available
    device = get_safe_torch_device(cfg.training.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_dataset")
    offline_dataset: LeCoroDataset = instantiate(cfg.training.dataset, obs_config=obs_config)
    if isinstance(offline_dataset.dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.dataset.repo_id_to_index , indent=2)}"
        )

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.training.eval.enable:
        logging.info("make_env")
        eval_env = instantiate(cfg.env)

    # create algo that knows a lot about the observations its receiving
    algo: Algo = instantiate(cfg.algo)()  # todo: look at why this gets instantiated as partial
    algo.make(
       obs_config=obs_config,
       shape_meta=offline_dataset.shape_meta,
       dataset_stats=offline_dataset.meta.stats
    )
    assert isinstance(algo, Algo)

    # create dataloader for offline training
    offline_dataset.set_delta_timestamps(algo.get_delta_timestamps(offline_dataset.fps))
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=offline_dataset.shuffle,
        sampler=offline_dataset.get_sampler(),
        pin_memory=device.type != "cpu",
        drop_last=False
    )
    dl_iter = cycle(dataloader)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step = logger.load_last_training_state(algo)

    # log algo + dataset info
    num_learnable_params = sum(p.numel() for p in algo.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in algo.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.training.offline_steps=} ({format_big_number(cfg.training.offline_steps)})")
    logging.info(f"{offline_dataset.dataset.num_frames=} ({format_big_number(offline_dataset.dataset.num_frames)})")
    logging.info(f"{offline_dataset.dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops.
    def evaluate_and_checkpoint_if_needed(step, is_online):
        _num_digits = max(6, len(str(cfg.training.offline_steps + cfg.training.online_steps)))
        step_identifier = f"{step:0{_num_digits}d}"

        if cfg.training.eval_freq > 0 and step % cfg.training.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.algo.get('use_amp', False) else nullcontext():
                assert eval_env is not None
                eval_info = eval_policy(
                    eval_env,
                    algo,
                    cfg.eval_n_episodes,
                    videos_dir=Path(out_dir) / "eval" / f"videos_step_{step_identifier}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_online=is_online)
            if cfg.wandb.enable:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.training.save_checkpoint and (
            step % cfg.training.save_freq == 0
            or step == cfg.training.offline_steps + cfg.training.online_steps
        ):
            logging.info(f"Checkpoint policy after step {step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            logger.save_checkpoint(
                step,
                algo,
                identifier=step_identifier,
            )
            logging.info("Resume training")

    algo.train()
    offline_step = 0
    for _ in range(step, cfg.training.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")

        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        train_info = algo(to_device(batch, device, non_blocking=True))

        train_info["dataloading_s"] = dataloading_s

        if step % cfg.training.logging.frequency == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset.dataset, is_online=False)

        # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
        # so we pass in step + 1.
        evaluate_and_checkpoint_if_needed(step + 1, is_online=False)

        step += 1
        offline_step += 1  # noqa: SIM113

    if eval_env:
        eval_env.close()
    logging.info("End of training")


@hydra.main(version_base="1.2", config_name="train_offline", config_path="../config")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train_notebook(out_dir=None, job_name=None, config_name="train_offline", config_path="../config"):
    from hydra import compose, initialize

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    train(cfg, out_dir=out_dir, job_name=job_name)


if __name__ == "__main__":
    train_cli()
