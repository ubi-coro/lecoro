import threading
import logging
import time
import requests
from flask import Flask, Response
import numpy as np
import cv2

from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames_torchvision,
    encode_video_frames,
    get_video_info,
)


def correct_color(img):
    """
    Correct the color of the image by converting from BGR to RGB.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Corrected image in RGB format.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
                img = correct_color(img)

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
