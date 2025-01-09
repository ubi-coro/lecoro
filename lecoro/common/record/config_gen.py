from dataclasses import dataclass

from hydra_zen import MISSING

@dataclass
class BaseRecorderConfig:
    root: str = MISSING
    dataset_repo_id: str = MISSING
    name: str = MISSING

    episode_time_s: float = 10.0
    num_episodes: int = 50
    task: str | None = None

    resume: bool = False
    play_sounds: bool = True
    video: bool = True
    use_foot_switch: bool = True
    run_compute_stats: bool = True
    push_to_hub: bool = False

    video_backend: str = 'pyav'
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4


