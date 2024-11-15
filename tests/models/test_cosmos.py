import pytest
import torch
import numpy as np
from cyber.models.world.autoencoder import CausalVideoTokenizer
from cyber.models.world.autoencoder.cosmos_tokenizer.utils import (
    pad_video_batch,
    numpy2tensor,
    read_video,
    resize_video,
    tensor2numpy,
    unpad_video_batch
)
from cyber.models.world.autoencoder.cosmos_tokenizer.networks.configs import discrete_video
from tests.models.utils import reseed_everything

class TestCosmosTokenizer:
    @classmethod
    def setup_class(cls):

        reseed_everything()

        cls.model = CausalVideoTokenizer(tokenizer_config=discrete_video)
        cls.model.to("cuda")

        video_path = "experiments/notebooks/demo_data/video_1s.mp4"
        video = read_video(video_path)
        video = resize_video(video)
        cls.batch_video = video[np.newaxis, ...]
        
        cls.num_frames = cls.batch_video.shape[1]
        cls.temporal_window = 17

    @classmethod
    def teardown_class(cls):
        del cls.model

    def test_encode_decode_process(self):
        tokens_list = []
        crop_regions = []

        for idx in range(0, (self.num_frames - 1) // self.temporal_window + 1):
            start, end = idx * self.temporal_window, (idx + 1) * self.temporal_window
            input_video = self.batch_video[:, start:end, ...]

            padded_input_video, crop_region = pad_video_batch(input_video)
            input_tensor = numpy2tensor(
                padded_input_video, dtype=torch.float32, device="cuda"
            )

            tokens = self.model.encode(input_tensor)[0]  # [1, 3, 16, 16]

            tokens_list.append(tokens.squeeze(0))  # [3, 16, 16]
            crop_regions.append(crop_region)

        all_tokens = torch.cat(tokens_list, dim=0)

        expected_token_shape = (6, 16, 16)
        assert all_tokens.shape == expected_token_shape, \
            f"Expected token shape {expected_token_shape}, got {all_tokens.shape}"
        
        reference_tokens = torch.load("tests/fixtures/tensors/cosmos_tokens.pth")
        reference_tokens = reference_tokens.to(all_tokens.device)

        assert torch.allclose(all_tokens, reference_tokens, rtol=1e-1, atol=1e-1), \
            "Generated tokens do not match reference tokens"
        
        num_windows = all_tokens.shape[0] // 3
        output_video_list = []
        for idx in range(num_windows):
            start_idx = idx * 3
            current_tokens = all_tokens[start_idx:start_idx + 3]
            current_tokens = current_tokens.unsqueeze(0)

            current_crop_region = crop_regions[idx]

            # output_tensor = decoder_DV.decode(current_tokens)
            output_tensor = self.model.decode(current_tokens)
            padded_output_video = tensor2numpy(output_tensor)
            output_video = unpad_video_batch(padded_output_video, current_crop_region)
            
            output_video_list.append(output_video)

        result_video_DV = np.concatenate(output_video_list, axis=1)
        expected_video_shape = (1, 30, 256, 256, 3)
        assert result_video_DV.shape == expected_video_shape, \
            f"Expected video shape {expected_video_shape}, got {result_video_DV.shape}"