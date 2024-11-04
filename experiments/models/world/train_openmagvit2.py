import torch
from torch.utils.data import random_split, DataLoader
from omegaconf import OmegaConf
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from cyber.models.world.autoencoder import VQModel
from cyber.dataset import RawImageDataset


def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(output_dir / "training.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def setup_model(config, device):
    model = VQModel(config)
    model.to(device)
    if config.use_ema:
        from cyber.models.world.autoencoder.magvit2.modules.ema import LitEma

        model_ema = LitEma(model)
    return model, model_ema if config.use_ema else None


def setup_data(model, data_dir, config):
    dataset = RawImageDataset(data_dir)
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_collator = model.get_train_collator()

    train_dataloader = DataLoader(train_dataset, batch_size=config.get("batch_size", 2), collate_fn=train_collator, shuffle=True, num_workers=4)

    val_dataloader = DataLoader(val_dataset, batch_size=config.get("batch_size", 2), collate_fn=train_collator, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader


def setup_optimizers(model, learning_rate, scheduler_type, train_dataloader, config):
    opt_gen = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.quantize.parameters()), lr=learning_rate, betas=(0.5, 0.9)
    )

    opt_disc = torch.optim.Adam(model.loss.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    # Setup schedulers if specified in config
    scheduler_gen, scheduler_disc = None, None
    if scheduler_type != "None":
        steps_per_epoch = len(train_dataloader)
        _ = steps_per_epoch * config.get("num_epochs", 100)
        _ = steps_per_epoch * config.get("warmup_epochs", 1.0)

        # Add scheduler setup logic here as needed

    return opt_gen, opt_disc, scheduler_gen, scheduler_disc


def train_epoch(train_dataloader, model, opt_gen, opt_disc, device):
    model.train()
    epoch_gen_losses = []
    epoch_disc_losses = []

    pbar = tqdm(train_dataloader, desc="Training")

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        gen_loss, disc_loss = model.compute_training_loss(**batch)

        # Generator update
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        # Discriminator update
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        epoch_gen_losses.append(gen_loss.item())
        epoch_disc_losses.append(disc_loss.item())

        pbar.set_postfix({"gen_loss": f"{gen_loss.item():.4f}", "disc_loss": f"{disc_loss.item():.4f}"})

    return {
        "gen_loss": sum(epoch_gen_losses) / len(epoch_gen_losses),
        "disc_loss": sum(epoch_disc_losses) / len(epoch_disc_losses),
    }


def validate(val_dataloader, model, device):
    model.eval()
    val_gen_losses = []
    val_disc_losses = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            gen_loss, disc_loss = model.compute_training_loss(**batch)
            val_gen_losses.append(gen_loss.item())
            val_disc_losses.append(disc_loss.item())

    return {
        "val_gen_loss": sum(val_gen_losses) / len(val_gen_losses),
        "val_disc_loss": sum(val_disc_losses) / len(val_disc_losses),
    }


def main(config_path: str, data_dir: str, output_dir: str):
    config = OmegaConf.load(config_path)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(output_dir_path)

    model, _ = setup_model(config, device)

    train_dataloader, val_dataloader = setup_data(model, data_dir, config)

    learning_rate = config.learning_rate
    opt_gen, opt_disc, _, _ = setup_optimizers(model, learning_rate, config.scheduler_type, train_dataloader, config)

    num_epochs = config.get("num_epochs", 100)

    for epoch in range(num_epochs):
        train_metrics = train_epoch(train_dataloader, model, opt_gen, opt_disc, device)

        val_metrics = validate(val_dataloader, model, device)

        metrics = {**train_metrics, **val_metrics}

        logger.info(f"Epoch {epoch} - {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenMagVIT2 model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.config, args.data_dir, args.output_dir)
