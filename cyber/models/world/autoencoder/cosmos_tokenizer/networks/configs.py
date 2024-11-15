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
"""The default image and video tokenizer configs."""

"""
Modified costom tokenizer configs for the cosmos video tokenizer.
"""

from cyber.models.world.autoencoder.cosmos_tokenizer.modules import (
    ContinuousFormulation,
    DiscreteQuantizer,
    Encoder3DType,
    Decoder3DType,
)

continuous_video = dict(  # noqa: C408
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    latent_channels=16,
    z_channels=16,
    z_factor=1,
    num_groups=1,
    legacy_mode=False,
    spatial_compression=8,
    temporal_compression=8,
    formulation=ContinuousFormulation.AE.name,
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="CV",
)

discrete_video = dict(  # noqa: C408
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    z_channels=16,
    z_factor=1,
    num_groups=1,
    legacy_mode=False,
    spatial_compression=16,
    temporal_compression=8,
    quantizer=DiscreteQuantizer.FSQ.name,
    embedding_dim=6,
    levels=[8, 8, 8, 5, 5, 5],
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="DV",
)
