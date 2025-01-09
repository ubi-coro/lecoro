import copy
from pathlib import Path
from typing import Callable, Dict, List, Literal

import torch
from av import logging
from omegaconf import OmegaConf
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset, CODEBASE_VERSION
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from torch.utils.data import Dataset

from coro.common.utils.hf_utils import create_branch, create_repo, create_lecoro_dataset_card, upload_folder
import coro.common.utils.obs_utils as ObsUtils
from coro.common.config_gen.observation import ObsConfig, ModalityConfig


# assuming this file is in the same directory

# modality specific transforms are done here (what was previously done in algo.preprocess_batch),
# used with _targets_ in configs assigned to modalities

# dataset features should contain actions and rewards -> I will make sure of that when recording demonstrations. stored in dataset_keys

# Lerobot dataset uses 'next.reward'?

# LambdaRL scheduler usually require the total number of steps -> extrapolation in algo_params.py


class LeCoroDataset(Dataset):
    """ creates a robomimic-esque dataset from a typical LeRobot dataset """

    def __init__(
            self,
            repo_id: str | list[str],

            # lerobot interface
            root: str | Path | None = None,
            episodes: dict | None = None,
            delta_timestamps: dict[list[float]] | None = None,
            drop_n_last_frames: int = 0,  # maybe get this from policy (would have to be in base config)
            tolerance_s: float = 1e-4,
            download_videos: bool = True,
            local_files_only: bool = False,
            video_backend: str | None = None,  # todo: find out which one worked on fafnir

            # prefered way of specifying the frame structure
            obs_config: ObsConfig | None = None,
            key_formatter: Callable | None = None,
            dataset_keys: tuple[str] | list[str] | None = None,  # keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            # robomimic way of specifying the frame structure
            obs_keys: bool | None = None,
    ):
        """
        A wrapper dataset that internally uses LeRobotDataset and produces data in a structure
        compatible with robomimic-style offline RL datasets.

        The returned batch will look like:
        {
          "obs": { "cam_low": torch.Tensor([...]), ... },
          "next_obs": { "cam_low": torch.Tensor([...]), ... },
          "goal_obs": { "cam_low": torch.Tensor([...]) },
          "actions": torch.Tensor([...]),
          "rewards": torch.Tensor([...])
        }

        - obs: Current observation keys derived from LeRobotDataset keys (flattened)
        - next_obs: Observations from the subsequent time step (if available), same format as obs
        - goal_obs: Observations from the last frame of the current episode
        - actions: If not provided by LeRobot, default to zeros
        - rewards: If not provided by LeRobot, default to zeros

        Args:
            repo_id (str): HF repo_id for the LeRobot dataset.
            root (str): Local path to store or load the dataset.
            episodes (list[int]): subset of episode indices to load, or None to load all.
            image_transforms (callable): transforms for images.
            expect_actions (bool): If True, we will try to extract actions from the dataset.
            expect_rewards (bool): If True, we will try to extract rewards from the dataset.
            default_action_dim (int): Action dimension if no actions are found.
            default_reward_dim (int): Reward dimension if no rewards are found.
            local_files_only (bool): If True, load dataset without remote downloads.
            download_videos (bool): If True, download videos if they are not available.
            video_backend (str): Video decoding backend.
        """
        self.obs_config = obs_config
        self.obs_keys = tuple(obs_keys) if obs_keys is not None else tuple()
        self.dataset_keys = tuple(dataset_keys) if obs_keys is not None else ('action', 'action_is_pad', 'rewards')
        self._group_mode: bool = obs_config is not None

        # maps observation keys, such as 'observation.images.cam_low' to a 2-tuple containing
        #   a new / stripped key under which the observation should be stored (such as just 'cam_low')
        #   whether this key belongs to an observation (would be false for dataset keys like 'rewards' or 'actions')
        #   -> not needed, because being an obs comes from the formatted key being in the obs config
        if key_formatter is None:
            key_formatter = lambda key: (key)
        self.key_formatter = key_formatter

        if self._group_mode:
            # store a list of groups (observation, goal, subgoal) and a mapping from keys to those group they are in
            self._key_to_groups = dict()
            self._expected_keys = []
            self._all_groups = []
            for group in self.obs_config.modalities:
                for keys in self.obs_config.modalities[group].values():
                    for key in keys:
                        if key in self._key_to_groups:
                            self._key_to_groups[key].append(group)
                        else:
                            self._key_to_groups[key] = [group]
                        self._expected_keys.append(key)

                    if group in ['obs', 'subgoals']:
                        self._all_groups.append('obs')
                    if group == 'goals':
                        self._all_groups.append('goal_obs')
            self._expected_keys = set(self._expected_keys)
            self._all_groups = set(self._all_groups)

        self._dataset_kwargs = {
            'repo_id': repo_id,
            'root': root,
            'episodes': episodes,
            'delta_timestamps': delta_timestamps,
            'tolerance_s': tolerance_s,
            'local_files_only': local_files_only,
            'download_videos': download_videos,
            'video_backend': video_backend
        }
        self.drop_n_last_frames = drop_n_last_frames

        self.dataset: LeRobotDataset = None
        self.meta: LeRobotDatasetMetadata = None
        self.shape_meta: dict = {}
        self._load_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        frame = self.dataset[idx]

        if 'goals' in self._all_groups:
            # fetch last frame of that episode
            ep_idx = frame["episode_index"].item()
            ep_end = self.dataset.episode_data_index["to"][ep_idx]
            goal_frame = self.dataset[ep_end]

        if self.obs_config is None and self.obs_keys is None:
            return {self.key_formatter(key): value for key, value in frame.items()}

        if self._group_mode:
            new_frame = {group: dict() for group in self._all_groups}
        else:
            new_frame = {}

        for key in frame:
            new_key = self.key_formatter(key)

            if self._group_mode and new_key in self._key_to_groups:
                groups = self._key_to_groups[new_key]

                if 'obs' in groups or 'subgoal' in groups:
                    new_frame['obs'][new_key] = ObsUtils.process_obs(frame[key], obs_key=key)

                if 'goal' in groups:
                    new_frame['goal_obs'][new_key] = ObsUtils.process_obs(goal_frame[key], obs_key=key)

            elif self.obs_keys is not None and new_key in self.obs_keys:
                new_frame[new_key] = ObsUtils.process_obs(frame[key], obs_key=key)

            elif new_key in self.dataset_keys:
                new_frame[new_key] = ObsUtils.process_obs(frame[key], obs_key=key)

        return new_frame

    @property
    def fps(self):
        return self.dataset.fps

    @property
    def shuffle(self):
        return self.drop_n_last_frames == 0

    def get_sampler(self):
        if self.drop_n_last_frames > 0:
            return EpisodeAwareSampler(
                self.dataset.episode_data_index,
                drop_n_last_frames=self.drop_n_last_frames,
                shuffle=True,
            )
        else:
            return None

    def set_delta_timestamps(self, delta_timestamps):
        self._dataset_kwargs['delta_timestamps'] = delta_timestamps
        self._load_dataset()

    def _get_shape_meta(self):
        # logic for turning hf_features into a flat key-only dictionary mapping to the expected shape for that key
        shape_meta = {}
        features = self.dataset.features

        if self.obs_config is None and self.obs_keys is None:
            return {self.key_formatter(key): self._get_processed_shape(value['shape'], self.key_formatter(key))
                    for key, value in features.items()}

        for key in self.dataset.features:
            new_key = self.key_formatter(key)
            shape = self.dataset.features[key]['shape']

            if (self._group_mode and new_key in self._key_to_groups) or \
                    (self.obs_keys is not None and new_key in self.obs_keys) or \
                    (new_key in self.dataset_keys):
                shape_meta[new_key] = self._get_processed_shape(shape, new_key)
        return shape_meta

    def _get_processed_shape(self, shape, key):
        # load modality specific shape processor and return the output
        return ObsUtils.get_processed_shape(ObsUtils.OBS_KEYS_TO_MODALITIES[key], shape)

    def _load_dataset(self):
        if isinstance(self._dataset_kwargs['repo_id'], str):
            self.dataset = LeRobotDataset(**self._dataset_kwargs)
        else:
            self.dataset = MultiLeRobotDataset(**self._dataset_kwargs)
        self.shape_meta = self._get_shape_meta()

        # handle normalization stats
        self.meta = self.dataset.meta
        self.meta.stats = {self.key_formatter(key): value for key, value in self.meta.stats.items()}

        if self.obs_config is None:
            return

        # obs config might contain dataset overwrites:
        for key in self.meta.stats:
            stats_dict = {}

            if key == 'action':  # output key
                if self.obs_config.decoder.normalization_stats is not None:
                    stats_dict = self.obs_config.decoder.normalization_stats

            elif key in self.shape_meta:  # any input key
                # per-modality dataset overwrite
                modality = ObsUtils.OBS_KEYS_TO_MODALITIES[key]
                if self.obs_config.encoder[modality].normalization_stats is not None:
                    stats_dict = self.obs_config.encoder[modality].normalization_stats

                # per-key dataset overwrites
                if key in self.obs_config.encoder_overwrites and self.obs_config.encoder_overwrites[key].normalization_stats is not None:
                    stats_dict = self.obs_config.encoder_overwrites[key].normalization_stats

            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                self.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

