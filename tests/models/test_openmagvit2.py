from cyber.models.world.autoencoder import VQModel
from cyber.models.world import autoencoder
from omegaconf import OmegaConf
import torch

from tests.models.utils import reseed_everything

import pytest

class TestOpenMagVit2:
    @classmethod
    def setup_class(cls):
        reseed_everything()
        omvit2_conf = OmegaConf.load("experiments/configs/models/world/openmagvit2.yaml")
        cls.model:autoencoder = VQModel(omvit2_conf)
        cls.model.to("cuda")
        cls.input = torch.load("tests/fixtures/tensors/img_tensor.pth").to("cuda")

    @classmethod
    def teardown_class(cls):
        del cls.model

    def test_train_loss(self):
        self.model.train()
        loss1,loss2 = self.model(self.input)
        assert loss1.item() == pytest.approx(0.8149874210357666, rel=1e-5)
        assert loss2.item() == pytest.approx(1.3878211975097656, rel=1e-5)
    
    def test_inference(self):
        self.model.eval()
        tokens = self.model(self.input)
        assert tokens.shape == (1, 16,16), f"tokens shape is {tokens.shape}"
        # load prediction from file
        expected_tokens = torch.load("tests/fixtures/tensors/openmagvit2_encoded_tokens.pth")
        assert tokens.equal(expected_tokens.to("cuda")), f"encoded tokens are not equal"
        reconstructed = self.model.decode(expected_tokens)
        assert reconstructed.shape == (1, 3, 256, 256), f"reconstructed shape is {reconstructed.shape}"
        # load prediction from file
        expected_reconstructed = torch.load("tests/fixtures/tensors/openmagvit2_reconstructed_image.pth")
        assert torch.allclose(reconstructed, expected_reconstructed.to("cuda"), atol=1e-5), f"reconstructed image is not equal"