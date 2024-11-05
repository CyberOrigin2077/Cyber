#!/usr/bin/env python3

import math
import os
import imageio

import numpy as np
import torch
import torch.distributed.optim
import torch.utils.checkpoint
import torch.utils.data
import torchvision.transforms.v2.functional as transforms_f
from einops import rearrange
from tqdm.auto import tqdm

from cyber.models.world.autoencoder.magvit2.config import VQConfig
from cyber.models.world.autoencoder.magvit2.models.lfqgan import VQModel
from cyber.models.world.dynamic.genie.visualize import rescale_magvit_output


def export_to_mp4(frames: list, output_mp4_path: str, fps: int):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)

    # Ensure the output path ends with .mp4
    if not output_mp4_path.lower().endswith('.mp4'):
        output_mp4_path = f"{output_mp4_path}.mp4"

    # Convert PIL Images to NumPy arrays and ensure they are in uint8 format
    frames = [np.array(frame) for frame in frames]  # Convert to numpy array

    if not frames:
        raise ValueError("No frames to export")

    print(f"Exporting {len(frames)} frames of shape: {frames[0].shape}")

    try:
        with imageio.get_writer(output_mp4_path, fps=fps, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame.astype(np.uint8))  # Convert to uint8
        print(f"Video successfully exported to {output_mp4_path}")
    except Exception as e:
        print(f"Error exporting to MP4: {e}")
        # print("Attempting to save as individual frames instead.")
        # frame_dir = os.path.splitext(output_mp4_path)[0] + "_frames"
        # os.makedirs(frame_dir, exist_ok=True)
        # for i, frame in enumerate(frames):
        #     imageio.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.png"), frame.astype(np.uint8))
        # print(f"Frames saved to {frame_dir}")


def encode_and_save_tokens(video_data, ckpt_path, video_path):
    device = "cuda"
    dtype = torch.bfloat16

    model_config = VQConfig()
    model = VQModel(model_config, ckpt_path=ckpt_path)
    model = model.to(device=device, dtype=dtype)

    video_data = rearrange(video_data, 't h w c -> t c h w').to(dtype) / 127.5 - 1
    num_frames = video_data.shape[0]

    basename = os.path.splitext(os.path.basename(video_path))[0]
    tokens_file = f"output/tokens_{basename}.bin"
    os.makedirs('output', exist_ok=True)

    # Create memmap file
    mmap_tokens = np.memmap(tokens_file, dtype=np.uint32, mode="w+", shape=(num_frames, 16, 16))

    for i, frame in tqdm(enumerate(video_data), desc="Processing frames", total=num_frames):
        frame = frame.unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        with torch.no_grad():
            quant = model.encode(frame)[0]
            tokens = model.quantize.bits_to_indices(quant.permute(0, 2, 3, 1) > 0)
            mmap_tokens[i] = tokens.cpu().numpy().squeeze()

        torch.cuda.empty_cache()

    mmap_tokens.flush()
    print(f"Tokens saved to {tokens_file}")

    return tokens_file


def decode_tokens_to_video(tokens, ckpt_path, output_path, fps, batch_size=16, max_images=None):
    device = "cuda"
    dtype = torch.bfloat16

    model_config = VQConfig()
    model = VQModel(model_config, ckpt_path=ckpt_path)
    model = model.to(device=device, dtype=dtype)

    # Load tokens
    num_frames = tokens.shape[0] // (16 * 16)
    tokens = tokens.reshape(num_frames, 16, 16)

    @torch.no_grad()
    def decode_latents(video_data):
        decoded_imgs = []

        for shard_ind in tqdm(range(math.ceil(len(video_data) / batch_size)), desc="Decoding frames"):
            batch = torch.from_numpy(video_data[shard_ind * batch_size: (shard_ind + 1) * batch_size].astype(np.int64))
            # if model.use_ema:
            #     with model.ema_scope():
            #         quant = model.quantize.get_codebook_entry(rearrange(batch, "b h w -> b (h w)"),
            #                                                   bhwc=batch.shape + (model.quantize.codebook_dim,)).flip(1)
            #         decoded_imgs.append(rescale_magvit_output(model.decode(quant.to(device=device, dtype=dtype))))
            # else:
            quant = model.quantize.get_codebook_entry(rearrange(batch, "b h w -> b (h w)"),
                                                        bhwc=batch.shape + (model.quantize.codebook_dim,)).flip(1)  # noqa: RUF005
            decoded_imgs.append(rescale_magvit_output(model.decode(quant.to(device=device, dtype=dtype))))

            if max_images and len(decoded_imgs) * batch_size >= max_images:
                break

        return [transforms_f.to_pil_image(img) for img in torch.cat(decoded_imgs)]

    decoded_frames = decode_latents(tokens)

    import os

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    export_to_mp4(decoded_frames, output_path, fps)
    print(f"Decoded video saved to {output_path}")

    return decoded_frames
