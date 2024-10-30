from cyber.models.world.dynamic import STMaskGIT
from omegaconf import OmegaConf
import torch

from tests.models.utils import reseed_everything

import pytest
import sys

class TestGenie:
    @classmethod
    def setup_class(cls):
        reseed_everything()
        genie_conf = OmegaConf.load("experiments/configs/models/world/genie.yaml")
        cls.model = STMaskGIT(genie_conf)  # 35M model
        cls.model.init_weights() # deterministic initialization due to fixed seed
        cls.model.to("cuda")
        cls.input = torch.load("tests/fixtures/tensors/tokens.pth").to("cuda")

    @classmethod
    def teardown_class(cls):
        del cls.model

    def test_train_loss(self):
        self.model.train()
        input_copy = self.input.clone().detach()
        input_copy[0,3000:] = 262144
        loss = self.model.compute_training_loss(input_copy, self.input)
        assert loss.item() == pytest.approx(12.558472633361816, rel=sys.float_info.epsilon*2)
    
    def test_inference(self):
        self.model.eval()
        prediction = self.model(self.input[:,:4096-256], 1)
        assert prediction.shape == (1, 4096)
        # load prediction from file
        expected_prediction = torch.load("tests/fixtures/tensors/genie_prediction.pth")
        assert torch.allclose(prediction, expected_prediction)

