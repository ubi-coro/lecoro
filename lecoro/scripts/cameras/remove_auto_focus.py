import cv2
import hydra
from hydra_zen import instantiate

from lecoro.common.config_gen import Config, register_configs
from lecoro.common.utils.obs_utils import initialize_obs_utils_with_config, merge_obs_keys
from lecoro.common.utils.utils import set_global_seed

register_configs()

@hydra.main(version_base="1.2", config_name="default", config_path="../../config")
def control_robot(cfg: Config):

    set_global_seed(cfg.seed)
    obs_config = instantiate(cfg.observation)
    initialize_obs_utils_with_config(obs_config)

    for key, camera in cfg.env.cameras.items():
        camera = instantiate(camera)
        camera.connect()
        camera.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)


if __name__ == "__main__":
    control_robot()
