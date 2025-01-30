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


