import torch
import numpy as np
import av
from PIL import Image


def resize_frame(frame, target_size=(256, 256)):
    pil_image = Image.fromarray(frame)
    resized_image = pil_image.resize(target_size)
    return np.array(resized_image)


def open_video(file):
    container = av.open(file)
    video = []

    for frame in container.decode(video=0):
        # Convert frame to numpy array in RGB format
        rgb_image = frame.to_rgb().to_ndarray()
        rgb_image = resize_frame(rgb_image)  # ! resize processing
        video.append(rgb_image)

    container.close()
    return torch.from_numpy(np.stack(video))


def rescale_magvit_output(magvit_output):
    """
    [-1, 1] -> [0, 255]

    Important: clip to [0, 255]
    """
    rescaled_output = (magvit_output.detach().cpu() + 1) * 127.5
    clipped_output = torch.clamp(rescaled_output, 0, 255).to(dtype=torch.uint8)
    return clipped_output
