from dataclasses import dataclass

from hydra_zen import MISSING

@dataclass
class DemonstrationRecorderConfig:
    root: str = MISSING
    dataset_repo_id: str = MISSING
    store_dataset: bool = True

    episode_time_s: float = 10.0
    num_episodes: int = 50
    task: str | None = None
    pretrained_algo_name_or_path: str | None = None
    algo_overrides: str | None = None

    resume: bool = False
    play_sounds: bool = True
    local_files_only: bool = False
    video: bool = True
    use_foot_switch: bool = True
    run_compute_stats: bool = True
    push_to_hub: bool = False

    video_backend: str = 'pyav'
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4


