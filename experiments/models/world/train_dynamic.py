# import logging
# logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from torch.utils.data import random_split

from cyber.models.world.dynamic import STMaskGIT
from cyber.models import CyberModule
from cyber.dataset import RawTokenDataset

import os
import time


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train the STMaskGIT model")
    # model and data configuration
    parser.add_argument("--config", type=str, default="configs/models/world/genie.yaml", help="Path to GENIE config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--window_size", type=int, default=16, help="Window size for the dataset, overrides config context window")
    parser.add_argument("--stride", type=int, default=15, help="Stride for the dataset e.g. 15 is equivalent to 2 fps")

    # hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size on each device")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")

    # training parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Path to save the checkpoints")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=1000, help="Save the checkpoint every n steps")
    parser.add_argument("--validate_every_n_steps", type=int, default=1000, help="Validate the model every n steps")
    parser.add_argument("--validate_steps", type=int, default=20, help="Number of steps to validate the model")

    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir):
    checkpoint = {"epoch": epoch, "step": step, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": loss}
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_step_{step}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def validate(model, val_dataloader, val_steps=0):
    model.eval()
    eval_loss = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Validating", leave=False)):
            for k, v in batch.items():
                batch[k] = v.to("cuda")
            loss = model.compute_training_loss(**batch)
            eval_loss.append(loss.item())
            if val_steps > 0 and i >= val_steps:
                break
    avg_val_loss = sum(eval_loss) / len(eval_loss)
    return avg_val_loss


def print_with_tqdm(pbar, message):
    tqdm.write(message)
    pbar.update(0)  # This refreshes the progress bar without advancing it


def main():
    # training parameters
    args = parse_args()

    # initialize the model
    genie_conf = OmegaConf.load(args.config)
    genie_conf.T = args.window_size
    model: CyberModule = STMaskGIT(genie_conf)  # 35M model

    # load the dataset
    pipette_data = RawTokenDataset(args.data_dir, window_size=args.window_size, stride=args.stride)

    # split the dataset into training and validation randomly
    train_dataset, val_dataset = random_split(pipette_data, [int(len(pipette_data) * 0.9), len(pipette_data) - int(len(pipette_data) * 0.9)])

    # data loader
    train_collator = model.get_train_collator()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_collator, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, collate_fn=train_collator, shuffle=True)  # shuffle here due to small val steps

    model.to("cuda")
    model.train()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # create dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f"{args.checkpoint_dir}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_graph = []
    for epoch in range(args.num_epochs):
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}/{args.num_epochs}", unit="step")
        for step, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to("cuda")
            loss = model.compute_training_loss(**batch)
            loss_graph.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

            # Validation after every `validate_every_n_steps`
            if (step + 1) % args.validate_every_n_steps == 0:
                print_with_tqdm(pbar, "Validating the model")
                val_loss = validate(model, val_dataloader, val_steps=args.validate_steps)
                print_with_tqdm(pbar, f"Epoch {epoch + 1} Step {step} Validation Loss: {val_loss:.4f}")

            # Save checkpoint every n steps
            if (step + 1) % args.checkpoint_every_n_steps == 0:
                print_with_tqdm(pbar, "Saving checkpoint")
                save_checkpoint(model, optimizer, epoch, step, loss.item(), checkpoint_dir)

    # plot the loss graph
    plt.plot(loss_graph)
    plt.title("Training Loss")
    plt.show()


if __name__ == "__main__":
    main()
