import cv2
from torchvision import transforms
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from cyber.models.world.autoencoder import VQModel
import argparse
from einops import rearrange


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = VQModel(config.model.init_args)
    # model = VQModel.from_config(config)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        _, _ = model.load_state_dict(sd, strict=False)
    return model.eval()


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def process_video(config_file, video_path, ckpt_path, save_dir, **kwargs):
    configs = OmegaConf.load(config_file)
    model = load_vqgan_new(configs, ckpt_path)
    model = model.to(device=DEVICE)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # output_path = os.path.join(args.save_dir, "direct.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))

    # save tokens
    tokens_file = os.path.join(save_dir, "tokens.bin")
    mmap_tokens = np.memmap(tokens_file, dtype=np.uint32, mode="w+", shape=(total_frames, 16, 16))

    # Normalize
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
            frame_tensor = frame_tensor * 2 - 1

            quant, _, _, _ = model.encode(frame_tensor)

            tokens = model.quantize.bits_to_indices(quant.permute(0, 2, 3, 1) > 0)
            mmap_tokens[frame_idx] = tokens.cpu().numpy().squeeze()

            # reconstructed = model.decode(quant)
            # reconstructed_image = custom_to_pil(reconstructed[0])
            # reconstructed_frame = cv2.cvtColor(np.array(reconstructed_image), cv2.COLOR_RGB2BGR)
            # out.write(reconstructed_frame)

    mmap_tokens.flush()

    cap.release()
    # out.release()


def reconstruct_video(config_file, tokens_path, ckpt_path, save_dir, fps, **kwargs):
    configs = OmegaConf.load(config_file)
    model = load_vqgan_new(configs, ckpt_path)
    model = model.to(device=DEVICE)

    tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
    num_frames = tokens.shape[0] // (16 * 16)
    tokens = tokens.reshape(num_frames, 16, 16)

    output_path = os.path.join(save_dir, "reconstructed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))

    reconstructed_frames = []

    with torch.no_grad():
        for frame_idx in tqdm(range(num_frames), desc="Reconstructing frames"):
            frame_tokens = torch.from_numpy(tokens[frame_idx : frame_idx + 1]).to(dtype=torch.int64, device=DEVICE)

            bhwc = (frame_tokens.shape[0], frame_tokens.shape[1], frame_tokens.shape[2], model.quantize.codebook_dim)
            quant = model.quantize.get_codebook_entry(rearrange(frame_tokens, "b h w -> b (h w)"), bhwc=bhwc).flip(1)

            reconstructed = model.decode(quant)
            reconstructed_frames.append(reconstructed)
            reconstructed_image = custom_to_pil(reconstructed[0])
            reconstructed_frame = cv2.cvtColor(np.array(reconstructed_image), cv2.COLOR_RGB2BGR)
            out.write(reconstructed_frame)

    out.release()
    return list(torch.cat(reconstructed_frames))


def process_video_from_args(args):
    return process_video(config_file=args.config_file, video_path=args.video_path, save_dir=args.save_dir, ckpt_path=args.ckpt_path)


def reconstruct_video_from_args(args):
    return reconstruct_video(config_file=args.config_file, save_dir=args.save_dir, ckpt_path=args.ckpt_path, tokens_path=args.tokens_path, fps=args.fps)


def get_args():
    parser = argparse.ArgumentParser(description="video processing")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--video_path", type=str, help="Path to input video")
    parser.add_argument("--tokens_path", type=str, help="Path to tokens.bin file")
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--mode", choices=["encode", "decode"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "encode":
        if args.video_path is None:
            raise ValueError("video_path is required for encode mode")
        process_video_from_args(args)

    else:  # decode
        if args.tokens_path is None:
            raise ValueError("tokens_path is required for decode mode")
        reconstruct_video_from_args(args)
