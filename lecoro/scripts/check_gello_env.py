import hydra
from hydra_zen import instantiate

from coro.common.env.aloha import AlohaManipulatorEnv
from coro.common.config_gen import register_configs, Config
import coro.common.utils.obs_utils as ObsUtils

register_configs()


@hydra.main(version_base=None, config_path='../config', config_name="config")
def main(cfg: Config):
    ObsUtils.initialize_obs_utils_with_config(cfg.observation)

    env: AlohaManipulatorEnv = instantiate(cfg.env)

    env.connect()
    env.toggle_torque(leader=False, follower=False)
    while True:
        obs = env.capture_observation()
        print(obs)
