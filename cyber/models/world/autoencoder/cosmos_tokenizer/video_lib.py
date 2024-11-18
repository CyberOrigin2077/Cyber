# ruff: noqa: E402
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library for Causal Video Tokenizer inference."""

"""
Modified the Causal Video Tokenizer for inference.
Modified by CYBERORIGIN PTE. LTD. on 2024-11-18
"""

import torch
from typing import Any, Callable, Optional


from cyber.models.world.worldmodel import AutoEncoder
from cyber.models.world.autoencoder.cosmos_tokenizer.utils import (
    load_encoder_model,
    load_decoder_model,
)
from cyber.models.world.autoencoder.cosmos_tokenizer.networks import TokenizerModels


class CausalVideoTokenizer(AutoEncoder):
    def __init__(
        self,
        checkpoint_enc: Optional[str] = None,
        checkpoint_dec: Optional[str] = None,
        tokenizer_config: Optional[dict[str, Any]] = None,  # TODO: better config loading
        device: str = "cuda",
        dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._device = device
        # self._dtype = getattr(torch, dtype)
        if dtype is None:
            self._dtype = torch.float32
        else:
            self._dtype = getattr(torch, dtype)

        # must provide either checkpoint or checkpoint_enc and checkpoint_dec
        if checkpoint_enc is None or checkpoint_dec is None:
            assert tokenizer_config is not None, "tokenizer_config must be provided if checkpoints are not provided."
            tokenizer_name = tokenizer_config["name"]
            model = TokenizerModels[tokenizer_name].value(**tokenizer_config)
            self._enc_model = model.encoder_jit().to(self._device, self._dtype)
            self._dec_model = model.decoder_jit().to(self._device, self._dtype)
        else:
            assert checkpoint_dec is not None and checkpoint_enc is not None, "checkpoint_enc and checkpoint_dec must be provided together."
            self._enc_model = load_encoder_model(checkpoint_enc, tokenizer_config, device).to(self._dtype) if checkpoint_enc is not None else None
            self._dec_model = load_decoder_model(checkpoint_dec, tokenizer_config, device).to(self._dtype) if checkpoint_dec is not None else None

    def compute_training_loss(self, *args, **kwargs):
        """
        Compute the training loss for the module.
        For inference, this function is not used.
        """
        raise NotImplementedError("This model is for inference only.")

    def get_train_collator(self, *args, **kwargs) -> Callable:
        """
        Get the collator for the module.
        For inference, this function is not used.
        """
        raise NotImplementedError("This model is for inference only.")

    @torch.no_grad()
    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Encodes a numpy video into a CausalVideo latent or code.

        Args:
            input_tensor: The input tensor Bx3xTxHxW layout, range [-1..1].
        Returns:
            For causal continuous video (CV) tokenizer, the tuple contains:
                - The latent embedding, Bx16x(t)x(h)x(w), where the compression
                rate is (T/t x H/h x W/w), and channel dimension of 16.
            For causal discrete video (DV) tokenizer, the tuple contains:
              1) The indices, Bx(t)x(h)x(w), from a codebook of size 64K, which
                is formed by FSQ levels of (8,8,8,5,5,5).
              2) The discrete code, Bx6x(t)x(h)x(w), where the compression rate
                is again (T/t x H/h x W/w), and channel dimension of 6.
        """
        assert input_tensor.ndim == 5, "input video should be of 5D."
        if self._enc_model is None:
            raise ValueError("Encoder model is not initialized")

        output_latent = self._enc_model(input_tensor)
        if isinstance(output_latent, torch.Tensor):
            return output_latent
        return output_latent[:-1]

    @torch.no_grad()
    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Encodes a numpy video into a CausalVideo latent.

        Args:
            input_latent: The continuous latent Bx16xtxhxw for CV,
                        or the discrete indices Bxtxhxw for DV.
        Returns:
            The reconstructed tensor, layout [B,3,1+(T-1)*8,H*16,W*16] in range [-1..1].
        """
        assert input_latent.ndim >= 4, "input latent should be of 5D for continuous and 4D for discrete."
        if self._dec_model is None:
            raise ValueError("Decoder model is not initialized")

        return self._dec_model(input_latent)
