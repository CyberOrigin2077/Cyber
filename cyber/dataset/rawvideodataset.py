import logging
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset


# Create a package-level logger
logger = logging.getLogger(__name__)


class RawVideoDataset(Dataset):
    """
    a simple dataset used to retrieve raw frames from videos
    videos are stored in a directory with the following structure:
    dataset_path
    ├── video_0.mp4
    ├── video_1.mp4
    ...
    """

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.video_files = glob.glob(f"{dataset_path}/*.mp4")

    def __len__(self) -> int:
        """
        return the number of videos in the dataset
        """
        return len(self.video_files)

    def __getitem__(self, idx) -> np.ndarray:
        """
        get all frames from a single video, return them as a list of numpy arrays
        """
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return np.array(frames)
