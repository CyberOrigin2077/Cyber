import random
import math
import os
import logging

import mup
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from omegaconf import OmegaConf


from cyber.models.world import DynamicModel
from cyber.models.world.dynamic.genie.factorization_utils import FactorizedEmbedding, factorize_labels, factorize_token_ids, unfactorize_token_ids
from cyber.models.world.dynamic.genie.st_transformer import STTransformerDecoder
from cyber.utils.module import load_statedict_from_file


def cosine_schedule(u):
    """u in [0, 1]"""
    if isinstance(u, torch.Tensor):
        cls = torch
    elif isinstance(u, float):
        cls = math
    else:
        raise NotImplementedError(f"Unexpected {type(u)=} {u=}")

    return cls.cos(u * cls.pi / 2)


class STMaskGIT(DynamicModel):
    # Next-Token prediction as done in https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, config):  # this now accepts omegaconf config
        super().__init__()
        self.h = self.w = math.isqrt(config.S)
        assert self.h**2 == config.S, "Expected S to be square"

        self.decoder = STTransformerDecoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            qk_norm=config.qk_norm,
            use_mup=config.use_mup,
            attn_drop=config.attn_drop,
            mlp_ratio=config.mlp_ratio,
            mlp_bias=config.mlp_bias,
            mlp_drop=config.mlp_drop,
        )

        self.pos_embed_TSC = torch.nn.Parameter(torch.zeros(1, config.T, config.S, config.d_model))
        self.mask_token_id = config.image_vocab_size

        self.token_embed = FactorizedEmbedding(  # also works for num_factored_vocabs = 1
            factored_vocab_size=config.factored_vocab_size,
            num_factored_vocabs=config.num_factored_vocabs,
            d_model=config.d_model,
            mask_token_id=self.mask_token_id,
        )

        cls = FixedMuReadout if config.use_mup else nn.Linear  # (Fixed)MuReadout might slow dow down compiled training?
        self.out_x_proj = cls(config.d_model, config.factored_vocab_size * config.num_factored_vocabs)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        frames_to_generate: int,
        return_logits: int = False,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates `frames_to_generate` frames given `x`.

        Args:
            x(torch.Tensor[torch.long]): Input tokens, size (B, T' * H * W) where T' = T - frames_to_generate.
            frames_to_generate: Number of frames to generate.
            return_logits: If True, will return the logits for each generated frame.
            maskgit_steps: Number of MaskGIT-style inference steps to take.
            temperature: Sampling temperature.

        Returns: (predicted_tokens, factored_logits) if `return_logits` else predicted_tokens
        """
        assert x.dtype is torch.long

        inputs_THW = rearrange(x.clone(), "b (t h w) -> b t h w", h=self.h, w=self.w)
        inputs_masked_THW = torch.cat(
            [inputs_THW, torch.full((x.size(0), frames_to_generate, self.h, self.w), self.mask_token_id, dtype=torch.long, device=x.device)], dim=1
        )

        all_factored_logits = []
        for timestep in range(inputs_THW.size(1), inputs_THW.size(1) + frames_to_generate):
            # could change sampling hparams
            sample_HW, factored_logits = self.maskgit_generate(inputs_masked_THW, timestep, maskgit_steps=maskgit_steps, temperature=temperature)
            inputs_masked_THW[:, timestep] = sample_HW
            all_factored_logits.append(factored_logits)

        predicted_tokens = rearrange(inputs_masked_THW, "B T H W -> B (T H W)")
        if return_logits:
            return predicted_tokens, torch.stack(all_factored_logits, dim=3)  # (b, factored_vocab_size, num_factored_vocabs, frames_to_generate, h, w)
        else:
            return predicted_tokens

    @staticmethod
    def init_mask(prompt_THW):
        # since we generate 1 image at a time, the mask should be for a single frame, not across all frames.
        T, H, W = prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        unmasked = torch.zeros(prompt_THW.size(0), H * W, dtype=torch.bool, device=prompt_THW.device)
        return unmasked

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW: torch.Tensor,
        out_t: int,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
        unmask_mode: str = "random",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs MaskGIT-style inference to predict frame `out_t`.

        Args:
            prompt_THW(torch.Tensor[torch.long]): Unfactorized token ids, size (B, T, H, W)
            out_t: Will return predicted unfactorized token ids for this frame.
                Should be >= 1 as the 0th frame is assumed to be given.
                Expects all future frames to be fully masked.
            maskgit_steps: The number of MaskGIT-style inference steps to take.
            temperature: Sampling temperature.
                In the factorized case, sampling is performed for each factorized vocabulary independently.
                If temperature is <= 1e-8, will be greedy (i.e. argmax) instead of actual sampling.
            unmask_mode: The method to determine tokens to unmask during each step of MaskGIT inference.
                Options:
                    - "greedy" for unmasking the most confident tokens, which is matches the original MaskGIT
                    - "random" for randomly choosing tokens to unmask
                "greedy" tends to copy the previous frame, so we default to "random" instead.

        Returns: (sample_HW, factored_logits)
            sample_HW: size (B, H, W) corresponding to predicted unfactorized token ids for frame `out_t`.
            factored_logits: size (B, factored_vocab_size, num_factored_vocabs, H, W).
        """
        assert prompt_THW.dtype is torch.long
        # assume we have pre-masked z{out_t}...zT with all masks
        assert out_t, "maskgit_generate requires out_t > 0"
        assert torch.all(prompt_THW[:, out_t:] == self.mask_token_id), f"when generating z{out_t}, frames {out_t} and later must be masked"

        bs, t, h, w = prompt_THW.size(0), prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)

        # this will be modified in place on each iteration of this loop
        unmasked = self.init_mask(prompt_THW)

        logits_CTHW = self.compute_logits(prompt_THW)
        logits_CHW = logits_CTHW[:, :, out_t]
        orig_logits_CHW = logits_CHW.clone()  # Return these original logits, not logits after partially sampling.
        for step in tqdm(range(maskgit_steps)):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:  # recompute logits with updated prompt
                logits_CHW = self.compute_logits(prompt_THW)[:, :, out_t]

            factored_logits = rearrange(
                logits_CHW,
                "b (num_vocabs vocab_size) h w -> b vocab_size num_vocabs h w",
                vocab_size=self.config.factored_vocab_size,
                num_vocabs=self.config.num_factored_vocabs,
            )

            factored_probs = torch.nn.functional.softmax(factored_logits, dim=1)

            samples_HW = torch.zeros((bs, h, w), dtype=torch.long, device=prompt_THW.device)
            confidences_HW = torch.ones((bs, h, w), dtype=torch.float, device=prompt_THW.device)
            for probs in factored_probs.flip(2).unbind(2):
                if temperature <= 1e-8:  # greedy sampling
                    sample = probs.argmax(dim=1)
                else:
                    # Categorical expects last dim to be channel dim
                    dist = torch.distributions.categorical.Categorical(probs=rearrange(probs, "b vocab_size ... -> b ... vocab_size") / temperature)
                    sample = dist.sample()
                samples_HW *= self.config.factored_vocab_size
                samples_HW += sample
                confidences_HW *= torch.gather(probs, 1, sample.unsqueeze(1)).squeeze(1)

            prev_unmasked = unmasked.clone()
            prev_img_flat = rearrange(prompt_THW[:, out_t], "B H W -> B (H W)")

            samples_flat = samples_HW.reshape(bs, self.config.S)

            if step != maskgit_steps - 1:  # skip masking for last maskgit step
                # use cosine mask scheduling function, n is how many of frame out_t to mask
                n = math.ceil(cosine_schedule((step + 1) / maskgit_steps) * self.config.S)

                if unmask_mode == "greedy":
                    # set the n patches with the least confidence to mask_token
                    confidences_flat = confidences_HW.reshape(bs, self.config.S)
                elif unmask_mode == "random":
                    # randomize confidences, so that patches are randomly masked
                    confidences_flat = torch.rand_like(confidences_HW).reshape(bs, self.config.S)
                    # not probability distribution anymore, but only relative order matters
                else:
                    raise NotImplementedError(f"Expected `unmask_mode` to be one of ['greedy', 'random'], " f"got {unmask_mode}")

                confidences_flat[unmasked] = torch.inf
                least_confident_tokens = torch.argsort(confidences_flat, dim=1)
                # unmask the (self.config.S - n) most confident tokens
                unmasked.scatter_(1, least_confident_tokens[:, n:], True)
                samples_flat.scatter_(1, least_confident_tokens[:, :n], self.mask_token_id)

            # copy previously unmasked values from prompt input into sample
            samples_flat[prev_unmasked] = prev_img_flat[prev_unmasked]
            samples_HW = samples_flat.reshape(-1, h, w)

            # feed back to iteratively decode
            prompt_THW[:, out_t] = samples_HW

        # Return the final sample and logits
        return samples_HW, rearrange(
            orig_logits_CHW,
            "B (num_vocabs vocab_size) H W -> B vocab_size num_vocabs H W",
            vocab_size=self.config.factored_vocab_size,
            num_vocabs=self.config.num_factored_vocabs,
            H=h,
            W=w,
        )

    def compute_loss_and_acc(self, logits_CTHW, targets_THW, relevant_mask_THW):
        # Video token prediction
        targets_THW = targets_THW.clone()
        logits_CTHW, targets_THW = logits_CTHW[:, :, 1:], targets_THW[:, 1:]  # first frame always unmasked
        factored_logits = rearrange(
            logits_CTHW,
            "b (num_vocabs vocab_size) t h w -> b vocab_size num_vocabs t h w",
            vocab_size=self.config.factored_vocab_size,
            num_vocabs=self.config.num_factored_vocabs,
        )
        factored_targets = factorize_labels(targets_THW)

        loss_THW = F.cross_entropy(factored_logits, factored_targets, reduction="none").sum(dim=1)
        acc_THW = (factored_logits.argmax(dim=1) == factored_targets).all(dim=1)

        # Compute the mean masked error.
        # Multiply loss values by mask instead of indexing them, more computationally efficient.
        num_masked_tokens = torch.sum(relevant_mask_THW)
        relevant_loss = torch.sum(loss_THW * relevant_mask_THW) / num_masked_tokens
        relevant_acc = torch.sum(acc_THW * relevant_mask_THW).float() / num_masked_tokens

        # only optimize on the masked/noised logits?
        return relevant_loss, relevant_acc

    def compute_logits(self, x_THW):
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        x_TS = rearrange(x_THW, "B T H W -> B T (H W)")
        x_TSC = self.token_embed(x_TS)

        # additive embeddings, using the same vocab space
        x_TSC = self.decoder(x_TSC + self.pos_embed_TSC)
        x_next_TSC = self.out_x_proj(x_TSC)

        logits_CTHW = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=self.h, W=self.w)
        return logits_CTHW

    def compute_training_loss(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Compute the training loss for the module given input tokens and labels.

        Args:
            x(torch.LongTensor): The input tokens, size (B, T * H * W)
            labels(torch.LongTensor): The target tokens, size (B, T * H * W)

        Returns:
            relevant_loss(torch.FloatTensor): The cross entropy loss of the input.
        """
        assert x.dtype is torch.long
        assert labels.dtype is torch.long

        T, H, W = self.config.T, self.h, self.w
        x_THW = rearrange(x, "B (T H W) -> B T H W", T=T, H=H, W=W)
        logits_CTHW = self.compute_logits(x_THW)

        labels = rearrange(labels, "B (T H W) -> B T H W", T=T, H=H, W=W)

        # Record the loss over masked tokens only to make it more comparable to LLM baselines
        relevant_mask = x_THW[:, 1:] == self.mask_token_id  # could also get mask of corrupted tokens by uncommenting line in `get_maskgit_collator`
        relevant_loss, _ = self.compute_loss_and_acc(logits_CTHW, labels, relevant_mask)

        # return ModelOutput(loss=relevant_loss, acc=relevant_acc, logits=logits_CTHW)
        return relevant_loss

    def init_weights(self):
        """Works with and without muP."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module.weight, "infshape"):  # muP
                    mup.normal_(module.weight, mean=0.0, std=std)
                else:
                    module.weight.data.normal_(mean=0.0, std=std)

                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def set_mup_shapes(self, rescale_params=False):
        base_config = self.config.shallow_copy()
        base_config.num_heads = 8
        base_config.d_model = 256  # currently hardcoding to this shape
        base_model = STMaskGIT(base_config)

        mup.set_base_shapes(self, base_model, rescale_params=rescale_params)

    @classmethod
    def from_pretrained(cls, path: str):
        """Extra logic for muP."""
        sd = load_statedict_from_file(os.path.join(path, "model.safetensors"))
        conf = OmegaConf.load(os.path.join(path, "config.yaml"))
        model = cls(conf)
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        if missing_keys or unexpected_keys:
            raise ValueError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        if model.config.use_mup:
            model.set_mup_shapes(rescale_params=False)
        logging.info(f"Restored pretrained model from {path}")
        return model

    def get_train_collator(self):
        return get_maskgit_collator(self.config)


def get_maskgit_collator(config):
    mask_token_id = config.image_vocab_size
    h = w = math.isqrt(config.S)

    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1

        input_ids = torch.stack([ex["input_ids"] for ex in features])
        device = input_ids.device
        x_THW = rearrange(input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T, h=h, w=w)
        x_THWC = factorize_token_ids(x_THW, config.num_factored_vocabs, config.factored_vocab_size)
        labels = x_THW.clone()

        # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
        r = torch.rand(x_THWC.size(), device=device)
        u01 = torch.rand((), device=device)
        random_patches_mask = r < config.max_corrupt_rate * u01
        random_values = torch.randint(low=0, high=config.factored_vocab_size, size=x_THWC.size(), dtype=torch.long, device=device)
        x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(0.25, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand((len(features), h, w, config.num_factored_vocabs), device=device)
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[:, first_masked_frame + i][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = 1

        mask = torch.zeros(1)
        c = 0
        while mask.max() == 0:  # We could get unlucky and mask no tokens?
            # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
            mask_prob_T = cosine_schedule(torch.rand(len(features), config.T - first_masked_frame, 1, 1))

            r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob_T
            c += 1

        if c > 1:
            print(f"Generated mask {c} > 1 times.")

        x_THW = unfactorize_token_ids(x_THWC, config.num_factored_vocabs, config.factored_vocab_size)
        x_THW[:, first_masked_frame:][mask] = mask_token_id

        return {
            "x": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
        }

    return collate_fn


class FixedMuReadout(mup.MuReadout):
    def forward(self, x):
        """
        Using `return super(mup.MuReadout, self).forward(self.output_mult * x / self.width_mult())` with `torch.compile`
        results in two divisions by `self.width_mult()` for some reason
        """
        # return F.linear(self.output_mult * x / self.width_mult(), self.weight, self.bias)  # equivalent
        return nn.Linear.forward(self, self.output_mult * x / self.width_mult())
