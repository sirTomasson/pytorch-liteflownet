import torch
import av
import cv2

import numpy as np

from run import estimate

def read_flow_file(file_path):
    """
    Read a flow file that was written with the format:
    - 4-byte header (80, 73, 69, 72) as uint8
    - 2 int32 values for width and height
    - Float32 flow data

    Parameters:
    file_path (str): Path to the flow file

    Returns:
    numpy.ndarray: Flow data with shape (2, height, width)
    """
    with open(file_path, 'rb') as f:
        # Read the header (4 bytes)
        header = np.fromfile(f, dtype=np.uint8, count=4)

        # Verify the header is correct (80, 73, 69, 72)
        if not np.array_equal(header, np.array([80, 73, 69, 72], np.uint8)):
            raise ValueError("Invalid flow file format - header mismatch")

        # Read the width and height (2 int32 values)
        width, height = np.fromfile(f, dtype=np.int32, count=2)

        # Read the flow data
        flow_data = np.fromfile(f, dtype=np.float32)

        # Reshape the data - original shape was (height, width, 2)
        # so we need to reshape and transpose to get (2, height, width)
        flow_data = flow_data.reshape(height, width, 2).transpose(2, 0, 1)

    return flow_data


def flow2rgb(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image where hue is direction and value is magnitude
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (direction)
    hsv[..., 1] = 255  # Saturation (full)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value (magnitude)

    # Convert HSV to RGB for visualization
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rgb2flow(rgb_image):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Extract components
    h = hsv[..., 0] * np.pi / 90.0  # Convert hue to angle (reverse of angle * 180 / np.pi / 2)
    s = hsv[..., 1]  # Saturation (not used in conversion)
    v = hsv[..., 2]  # Value represents magnitude

    # Create flow matrix with same shape as input but with 2 channels
    flow = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 2), dtype=np.float32)

    # Convert polar coordinates (angle and magnitude) back to cartesian (x,y)
    magnitude = v  # Assuming magnitude was normalized to 0-255 range
    flow[..., 0] = magnitude * np.cos(h)  # x component
    flow[..., 1] = magnitude * np.sin(h)  # y component

    return flow


def flowvid2rgb(flow_vid):
    return np.stack([flow2rgb(f) for f in flow_vid])


def read_video_file(video_path):
    container = av.open(video_path)
    frames = []

    for idx, frame in enumerate(container.decode(video=0)):
        frames.append(frame.to_ndarray(format='rgb24'))

    container.close()
    return np.stack(frames)


def preprocess_video(video):
    # convert RGB to BGR by reversing the last channel 🤯
    video = video[:, :, :, ::-1]
    # change T x H x W x C
    video = video.transpose(0, 3, 1, 2)
    # normalise 0-1
    return torch.tensor(video / 255, dtype=torch.float32)


def preprocess_batch(batch_video):
    return torch.stack([preprocess_video(v) for v in batch_video])


def estimate_flo_video(video):
    flo = []
    frame1 = video[0]
    for frame2 in video[1:]:
        frame2 = frame2, dtype = torch.float32
        flo_frame = estimate(frame1, frame2)
        flo.append(flo_frame.detach().cpu().numpy())
        frame1 = frame2

    return np.stack(flo)