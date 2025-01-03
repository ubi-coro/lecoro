from dataclasses import replace
from typing import List, Dict
import tqdm
import time

from coro.common.env.base import ManipulatorEnv
import coro.common.utils.control_utils as CtrlUtils
import coro.common.utils.config_utils as ConfUtils
import coro.common.utils.dataset_utils as DataUtils
import coro.common.utils.obs_utils as ObsUtils
from coro.common.config_gen.recorder import BaseRecorderConfig
from coro.common.config_gen.experiment import ExperimentConfig


class BaseRecorder:
    """
    Base class for recording robot environment data.

    Attributes:
        env (ManipulatorEnv): The environment being recorded.
        obs_keys (List[str]): List of observation keys with modality prefixes (e.g., 'rgb.cam_low')
        config (BaseRecorderConfig): Recorder configuration object.
    """

    def __init__(self,
                 env: ManipulatorEnv,
                 obs_keys: List[str],
                 rec_config: BaseRecorderConfig | None = None,
                 exp_config: ExperimentConfig | None = None,
                 **kwargs):
        if rec_config is None:
            rec_config = BaseRecorderConfig()
        if exp_config is None:
            exp_config = ExperimentConfig()

        # Overwrite config arguments using kwargs
        self.config = ConfUtils.replace(rec_config, **kwargs)

        super().__init__(env, obs_keys)
        self.env = env
        self.obs_keys = obs_keys
        self.root = exp_config.root
        self.name = exp_config.name
        self.task = exp_config.name if self.config.task is None else self.config.task
        self.dataset: DataUtils.LeRobotDataset | None = None

        self.img_keys = [
            k for k in self.obs_keys if
            ObsUtils.is_image_modality(ObsUtils.OBS_KEYS_TO_MODALITIES[DataUtils.remove_modality_prefix(k)])
        ]
        self.non_img_keys = [
            k for k in self.obs_keys if
            not ObsUtils.is_image_modality(ObsUtils.OBS_KEYS_TO_MODALITIES[DataUtils.remove_modality_prefix(k)])
        ]

    def run(self):
        """
        Main recording loop for capturing episodes.

        Returns:
        """
        self.init_dataset()

        listener, events = CtrlUtils.init_keyboard_listener(self.config.use_foot_switch, self.config.play_sounds)

        while True:
            if self.dataset is not None and self.dataset.num_episodes >= self.config.num_episodes:
                break

            self.env.reset()

            # close gripper to start
            CtrlUtils.log_say("Close both gripper to start!", self.config.play_sounds)
            while not all([pos < 5.0 for pos in self.env.gripper_pos]):
                time.sleep(0.1)
            self.env.toggle_torque(leader=False)

            # recording loop
            episode_index = self.dataset.num_episodes
            CtrlUtils.log_say(f"Recording episode {episode_index}", self.config.play_sounds)
            self.record_episode(events)

            if events["rerecord_episode"]:
                CtrlUtils.log_say("Re-record episode", self.config.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                self.dataset.clear_episode_buffer()  # to-do: fix this
                continue

            # Increment by one dataset["current_episode_index"]
            self.dataset.save_episode(self.task)

            if events["stop_recording"]:
                break

        self.dataset.consolidate(self.config.run_compute_stats)

        if self.config.push_to_hub:
            self.dataset.push_to_hub(tags=None)

        CtrlUtils.log_say("Exiting", self.config.play_sounds)
        return self.dataset

    def record_episode(self, events: Dict) -> None:
        """
        Records a single episode of environment interaction.

        Args:
            events (Dict): Event handlers for controlling recording flow.
        """
        timestamp = 0
        start_episode_t = time.perf_counter()
        with tqdm.tqdm(total=self.config.episode_time_s, color='green') as pbar:
            while timestamp < self.config.episode_time_s:

                frame = self.env.teleop_step(record_data=True)

                self.dataset.add_frame(frame)

                timestamp = time.perf_counter() - start_episode_t
                pbar.update(min([timestamp - pbar.n, self.config.episode_time_s - pbar.n]))
                if events["exit_early"]:
                    events["exit_early"] = False
                    break

    def init_dataset(self):
        """
        Initializes the episode directory structure and session dictionary
        """
        # Create empty dataset or load existing saved episodes
        if self.config.resume:
            dataset = DataUtils.LeRobotDataset(
                repo_id=self.name,
                root=self.root,
                local_files_only=True
            )
            CtrlUtils.sanity_check_dataset_compatibility(
                dataset,
                features=self.get_features(),
                robot_type=self.env.config.robot_type,
                fps=self.env.fps)

            dataset.start_image_writer(
                num_processes=self.config.num_image_writer_processes,
                num_threads=self.config.num_image_writer_threads_per_camera * len(self.img_keys)
            )

        else:
            dataset = DataUtils.LeRobotDataset.create(
                repo_id=self.name,
                fps=self.env.fps,
                root=self.root,
                use_videos=self.config.video,
                features=self.get_features(),
                robot_type=self.env.config.robot_type,
                image_writer_processes=self.config.num_image_writer_processes,
                image_writer_threads=self.config.num_image_writer_threads_per_camera * len(self.img_keys)
            )
        self.dataset = dataset

    def get_features(self) -> Dict:
        """
        Filter environment keys so we collect only the keys we want.

        Returns:
            dataset (Dataset): A Hugging Face Dataset object.
        """
        available_features = self.env.features

        if 'action' not in available_features:
            raise ValueError("Environments must offer an 'action' feature")
        features = {'action': self.env.features['action']}

        for key in self.img_keys:
            if self.config.video:
                features[key] = available_features[key] | {"dtype": "video"}
            else:
                features[key] = available_features[key] | {"dtype": "video"}

        for key in self.non_img_keys:
            features[key] = available_features[key]
        return features


class HiLRecorder:
    def __init__(self,
                 env: ManipulatorEnv,
                 config: BaseRecorderConfig | None = None,
                 **kwargs):
        if config is None:
            config = BaseRecorderConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.env = env

    def run(self):
        pass

    def init_dataset(self, obs_config):
        pass

    def add_frame(self):
        pass

    def to_hf_dataset(self):
        pass

    def from_dataset_to_lerobot_dataset(self):
        pass


class SubtaskRecorder:
    def __init__(self,
                 env: ManipulatorEnv,
                 config: BaseRecorderConfig | None = None,
                 **kwargs):
        if config is None:
            config = BaseRecorderConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.env = env

    def run(self):
        pass

    def init_dataset(self, obs_config):
        pass

    def add_frame(self):
        pass

    def to_hf_dataset(self):
        pass

    def from_dataset_to_lerobot_dataset(self):
        pass
