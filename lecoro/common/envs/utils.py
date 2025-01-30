#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import requests
import threading
import time

import einops
import cv2
import numpy as np
import torch
from flask import Flask, Response
from torch import Tensor


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            img = torch.from_numpy(img)

            # sanity check that images are channel last
            h, w, c = img.shape[-3:]
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "... h w c -> ... c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float()
    return return_observations


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    return safe_goal_pos


class Displayer:
    def __init__(self, host="127.0.0.1", port=5000, target_height=300):
        """
        You can use the displayer to pass image information to separate process,
        which publishes your videos on a flask server you can customize (a little).
        Enables completely asynchronous live displays.

        Args:
            host (str): Host address for the Flask server.
            port (int): Port for the Flask server.
            target_height (int): Height to which all images will be resized.
        """
        self.host = host
        self.port = port
        self.target_height = target_height
        self.app = Flask(__name__)
        self.server_thread = None
        self.images_dict = {}  # Dictionary to hold images
        self.lock = threading.Lock()  # Lock for thread-safe updates

        # Define route to video feed
        self.app.add_url_rule('/video_feed', 'video_feed', self._video_feed, methods=["GET"])

        # Suppress most of Flask console output
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    def _concatenate_images(self):
        """
        Concatenate images horizontally from the dictionary.

        Returns:
            np.ndarray: Concatenated image.
        """
        with self.lock:  # Ensure thread-safe access to images_dict
            if not self.images_dict:
                return np.zeros((100, 100, 3), dtype=np.uint8)  # Return a blank image if no images exist.

            resized_images = []
            for label, img in self.images_dict.items():
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Add label to the image
                img = cv2.putText(
                    img.copy(), label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )
                # Resize image
                h, w, _ = img.shape
                scale = self.target_height / h
                new_width = int(w * scale)
                resized_img = cv2.resize(img, (new_width, self.target_height))
                resized_images.append(resized_img)

            # Concatenate images horizontally
            concatenated_image = np.hstack(resized_images)
            return concatenated_image

    def _generate_frames(self):
        """
        Generator function to yield concatenated frames for the video feed.
        """
        while True:
            concatenated_image = self._concatenate_images()
            _, buffer = cv2.imencode('.jpg', concatenated_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)  # Limit frame rate

    def _video_feed(self):
        """
        Video feed route for the Flask server.
        """
        return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def start_server(self):
        """
        Start the Flask server in a separate thread.
        """
        print(f"Video feed is available at: http://{self.host}:{self.port}/video_feed")
        self.server_thread = threading.Thread(
            target=self.app.run, kwargs={"host": self.host, "port": self.port, "debug": False, "threaded": True}, daemon=True
        ).start()

    def stop_server(self):
        """
        Stop the Flask server by making a request to the shutdown route.
        """
        try:
            requests.post(f"http://{self.host}:{self.port}/shutdown")
            if self.server_thread:
                self.server_thread.join()
            print("Server has been stopped.")
        except Exception as e:
            print(f"Failed to stop server: {e}")

    def update_images(self, images_dict):
        """
        Update the images to be displayed.

        Args:
            images_dict (dict): Dictionary of images to update. Keys are labels, and values are numpy arrays.
        """
        with self.lock:  # Ensure thread-safe update
            self.images_dict = images_dict.copy()
