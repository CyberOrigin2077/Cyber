import torch
import torch.nn.functional as F
import torchvision.transforms.transforms as visionTransforms
from einops import rearrange

from cyber.config.utils import instantiate_from_config
from contextlib import contextmanager
from collections import OrderedDict
from typing import Callable


from cyber.models.world.autoencoder.magvit2.modules.diffusionmodules.improved_model import Encoder, Decoder
from cyber.models.world.autoencoder.magvit2.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from cyber.models.world.autoencoder.magvit2.modules.vqvae.lookup_free_quantize import LFQ
from cyber.models.world.autoencoder.magvit2.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from cyber.models.world.autoencoder.magvit2.modules.ema import LitEma

from cyber.models.world import AutoEncoder

import PIL
import logging


class VQModel(AutoEncoder):
    def __init__(self, config):
        """
        Args:
            config: Configuration object containing all model parameters
        """
        super().__init__()

        ddconfig = config.get("ddconfig", {})
        lossconfig = config.get("lossconfig", {})

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        # Quantize
        self.quantize = LFQ(
            dim=config.embed_dim,
            codebook_size=config.n_embed,
            sample_minimization_weight=config.sample_minimization_weight,
            batch_maximization_weight=config.batch_maximization_weight,
            token_factorization=config.get("token_factorization", False),
            factorized_bits=config.get("factorized_bits", [9, 9]),
        )

        # Colorize
        if config.get("colorize_nlabels") is not None:
            assert isinstance(config.colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, config.colorize_nlabels, 1, 1))

        if config.get("monitor") is not None:
            self.monitor = config.monitor

        # ema
        self.use_ema = config.get("use_ema", False)
        stage = config.get("stage", None)
        if self.use_ema and stage is None:  # no need to construct EMA when training Transformer
            self.model_ema = LitEma(self)

        # load ckpt
        ckpt_path = config.get("ckpt_path")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=config.get("ignore_keys", []), stage=stage)
        self.global_step = 0

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        filter out the non-used keys
        """
        return {
            k: v
            for k, v in super().state_dict(*args, destination, prefix, keep_vars).items()
            if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)
        }

    def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer":  ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items():
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "")  # load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "")  # load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue
            else:  # also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
        missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)  # first stage
        print(f"Restored from {path}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor to latent code from the codebook.
        Downsampling factor is 2^(len(ch_mult)-1).

        Args:
            x(torch.Tensor): input image batch size(B, 3, H, W)

        Returns:
            torch.Tensor: quantized latent code size(B, H/Ds, W/Ds) Ds: downsampling factor
        """
        assert len(x.shape) == 4
        assert x.shape[1] == 3
        # check if the H and W are divisible by 2^(len(ch_mult)-1)
        h, w = x.shape[2:]
        Ds = int(2 ** (self.encoder.num_blocks - 1))
        if not (h % Ds == 0 and w % Ds == 0):
            logging.warning(
                "Input size is not divisible by 2^(len(ch_mult)-1),\
                            tokenization will work but sizes will not match"
            )
        quant, _, _, _ = self._encode(x)
        tokens = self.quantize.bits_to_indices(quant.permute(0, 2, 3, 1) > 0)
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent code to the image.

        Args:
            tokens(torch.Tensor): quantized latent code size(B, H, W)

        Returns:
            torch.Tensor: reconstructed image size(B, 3, H*Ds, W*Ds) Ds: downsampling factor
        """
        assert len(tokens.shape) == 3
        bhwc = (tokens.shape[0], tokens.shape[1], tokens.shape[2], self.quantize.codebook_dim)
        quant = self.quantize.get_codebook_entry(rearrange(tokens, "b h w -> b (h w)"), bhwc=bhwc).flip(1)
        reconstructed = self.decoder(quant)
        return reconstructed

    def _encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h, return_loss_breakdown=True)
        return quant, emb_loss, info, loss_breakdown

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decoder(quant_b)
        return dec

    def encode_and_decode(self, input):
        quant, diff, _, loss_break = self._encode(input)
        dec = self.decoder(quant)
        return dec, diff, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def compute_training_loss(self, images: torch.Tensor, **kwargs):
        """
        Compute the training loss for the the VQGAN model.
        Returns the generator loss and discriminator loss
        Args:
            images: torch.Tensor: input images size(B, 3, H, W)

        Returns:
            tuple(torch.Tensor, torch.Tensor): (generator loss, discriminator loss)
        """
        x = images
        xrec, eloss, loss_break = self.encode_and_decode(x)

        aeloss, _ = self.loss(eloss, loss_break, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
        discloss, _ = self.loss(eloss, loss_break, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")

        return aeloss, discloss

    def get_train_collator(self, overide_transform: visionTransforms.Compose = None, *args, **kwargs) -> Callable:
        if overide_transform is not None:

            def collate_fn(images: PIL.Image.Image) -> torch.Tensor:
                images = [overide_transform(image) for image in images]
                return {"images", torch.stack(images)}
        else:
            T = visionTransforms.Compose(
                [
                    visionTransforms.Resize((256, 256)),
                    visionTransforms.RandomHorizontalFlip(),
                    visionTransforms.ToTensor(),
                    visionTransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

            def collate_fn(images: PIL.Image.Image) -> torch.Tensor:
                images = [T(image) for image in images]
                return {"images": torch.stack(images)}

        return collate_fn

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.quantize.parameters()), lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))

        step_per_epoch = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
                opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min)
            )
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
                opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min)
            )
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self.encode_and_decode(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
