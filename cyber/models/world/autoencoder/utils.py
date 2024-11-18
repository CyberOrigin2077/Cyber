import torch
import os
import json
import logging
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm
from cyber.dataset import RawVideoDataset
from cyber.models.world import AutoEncoder
from cyber.models.world.autoencoder import VQModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_vqgan_new(config, ckpt_path=None):
    """
    Load a new VQGAN model and initialize it with the provided checkpoint file, if specified.

    Args:
    - config (dict or OmegaConf object): The configuration for the model, which includes the model's architecture and parameters.
    - ckpt_path (str, optional): The path to a pre-trained model checkpoint. Default is None.

    Returns:
    - model (VQModel): The initialized VQGAN model, set to evaluation mode (`eval()`).
    """
    model = VQModel(config)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        _, _ = model.load_state_dict(sd, strict=False)

    return model.eval()


def encode_videos(model: AutoEncoder, dataset: RawVideoDataset, dataset_dir: str, s=16, resize_width=256, resize_height=256):
    """
    encode videos into rawtokendataset compatible format

    Args:
    model: the model to use for encoding
    dataset: the dataset to encode
    dataset_dir: the dataset's directory

    files will be stored in a directory with the following structure:
    ├──dataset_path
        ├── video_0.mp4
        ├── video_1.mp4
        ...
    ├──compressed
        ├── metadata.json
        ├── segment_ids.bin
        ├── videos.bin

    """

    # Normalize
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize_width, resize_height)),
            transforms.ToTensor(),
        ]
    )
    save_dir = os.path.join(os.path.dirname(dataset_dir), "compressed")
    if os.path.exists(save_dir):
        logger.info("Skipping as it already exists in the output directory.")
        return
    else:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Created the output directory: {save_dir}")

    # Step1 generate segment_ids.bin
    total_frames = 0
    seg_ids = []
    logger.info(f"dataset : {dataset[0].shape[0]}")  # (num_frames, H, W, C)
    logger.info(f"dataset size: {dataset[0].shape}")  # (num_frames, H, W, C)
    logger.info(f"len(dataset): {len(dataset)}")

    for i in tqdm(range(len(dataset)), desc="Counting videos' frames", unit="video"):
        cur_frame_num = dataset[i].shape[0]
        seg_ids.extend([i] * cur_frame_num)
        total_frames += cur_frame_num

    logger.info(f"Total frames: {total_frames}")

    merged_id = os.path.join(save_dir, "segment_ids.bin")
    merged_ids = np.memmap(merged_id, dtype=np.uint32, mode="w+", shape=(total_frames,))

    for i in range(len(seg_ids)):
        merged_ids[i] = seg_ids[i]

    merged_ids.flush()
    del merged_ids
    logger.info(f"segment_ids saved to {merged_id}")

    # Step2 save metadata.json
    metadata = {
        "token_dtype": "uint32",
        "s": s,
        "h": resize_height // s,
        "w": resize_width // s,
        "vocab_size": 262144,
        "hz": 30,
        "tokenizer_ckpt": "imagenet_256_L.ckpt",
        "num_images": total_frames,
    }
    metadata_file = os.path.join(save_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=0)

    logger.info(f"Metadata saved to {metadata_file}")

    # Step3 save videos.bin
    tokens_file = os.path.join(save_dir, "videos.bin")
    mmap_tokens = np.memmap(tokens_file, dtype=np.uint32, mode="w+", shape=(total_frames, 16, 16))

    current_frame = 0
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing..."):
            color_image_data = dataset[idx]  # (num_frames, H, W, C)

            for frame_idx in tqdm(range(color_image_data.shape[0]), desc=f"Processing episode {idx}"):
                frame = color_image_data[frame_idx]  # (H, W, C)
                # todo: check float32
                frame_rgb = frame.astype("float32")
                frame_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
                frame_tensor = frame_tensor * 2 - 1  # Normalize to [-1, 1]

                tokens = model.encode(frame_tensor)
                mmap_tokens[current_frame] = tokens.cpu().numpy().squeeze()

                current_frame += 1
                torch.cuda.empty_cache()

    mmap_tokens.flush()
    del mmap_tokens
    logger.info(f"tokens saved to {tokens_file}")

    return tokens_file
