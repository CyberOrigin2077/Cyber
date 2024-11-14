import torch
import argparse
import logging

from omegaconf import OmegaConf
from cyber.models.world.autoencoder.utils import load_vqgan_new, encode_videos
from cyber.dataset import RawVideoDataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Encoding Script")
    parser.add_argument("--config_file", type=str, default="experiments/configs/models/world/openmagvit2.yaml", help="Path to the config file")
    parser.add_argument("--ckpt_path", type=str, default="experiments/checkpoints/imagenet_256_L.ckpt", help="Path to the checkpoint file")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory (this is required)")

    args = parser.parse_args()
    if args.config_file != "experiments/configs/models/world/openmagvit2.yaml" or args.ckpt_path != "experiments/checkpoints/imagenet_256_L.ckpt":
        logger.warning("Not using the default config file or checkpoint file! please adjust the params in the metadata file")

    configs = OmegaConf.load(args.config_file)
    model = load_vqgan_new(configs, args.ckpt_path)
    model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = RawVideoDataset(args.dataset_dir)

    encode_videos(model, dataset, args.dataset_dir)
