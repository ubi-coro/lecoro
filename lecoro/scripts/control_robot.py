import hydra
from hydra_zen import instantiate

from lecoro.common.config_gen import Config, register_configs
from lecoro.common.logger import Logger
from lecoro.common.utils.obs_utils import initialize_obs_utils_with_config, merge_obs_keys
from lecoro.common.utils.utils import set_global_seed

# for some reason, hydra does not see sub-modules unless I specifically import them before
# might be due to missing __init__ files?
import lecoro.common.envs.aloha.environment_aloha
import lecoro.common.workspace.demonstration
import lecoro.common.robot_devices.robots.dynamixel

register_configs()

@hydra.main(version_base="1.2", config_name="default", config_path="../config")
def control_robot(cfg: Config):

    set_global_seed(cfg.seed)
    obs_config = instantiate(cfg.observation)
    initialize_obs_utils_with_config(obs_config)

    # build and connect to environment
    env = instantiate(cfg.env)
    env.connect()

    # compare requested obs_keys in the config file to the available ones in the environment
    obs_keys = merge_obs_keys(obs_config, env)

    # build and run workspace
    instantiate(cfg.workspace, env=env, obs_keys=obs_keys).run()


if __name__ == "__main__":
    control_robot()
