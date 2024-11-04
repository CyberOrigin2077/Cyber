from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import logging


class RawImageDataset(Dataset):
    """
    This dataset is used to load raw image data from the disk.
    No labels are used in this dataset.
    """

    def __init__(self, dataset_path, transform=None):
        super().__init__()
        self.dataset_path = dataset_path
        # get all image file paths from the dataset
        supported_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
        self.image_files = []
        for ext in supported_extensions:
            self.image_files.extend(glob(f"{dataset_path}/*.{ext}", recursive=True))
        # if there are no image files, raise an error
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No image files found in {dataset_path}")
        # if there are unsupported files, raise a warning
        unsupported_files = [f for f in self.image_files if f.split(".")[-1] not in supported_extensions]
        if len(unsupported_files) > 0:
            logging.warning(f"Unsupported files found: {unsupported_files}")
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        pimg = Image.open(self.image_files[idx])
        if pimg.mode != "RGB":
            pimg = pimg.convert("RGB")
        if self.transform:
            timg = self.transform(pimg)
        else:
            timg = pimg
        return timg
