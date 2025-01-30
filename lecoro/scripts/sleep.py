import time

import hydra
from hydra_zen import instantiate

import lecoro.common.utils.obs_utils as ObsUtils
from lecoro.common.config_gen import register_configs, Config

register_configs()


@hydra.main(version_base=None, config_path='../config', config_name="default")
def main(cfg: Config):
    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(cfg.observation)

    # build and connect to environment
    env = instantiate(cfg.env)
    env.connect()

    env.close()
    time.sleep(3.0)

if __name__ == "__main__":
    main()

