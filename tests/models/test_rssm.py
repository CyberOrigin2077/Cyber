from cyber.models.world.dynamic import RecurrentStateSpaceModel
from cyber.models.world import DynamicModel
from omegaconf import OmegaConf
import torch

from tests.models.utils import reseed_everything

import pytest

class TestRSSM:
    @classmethod
    def setup_class(cls):
        rssm_conf = OmegaConf.load("experiments/configs/models/world/rssm.yaml")
        reseed_everything()
        cls.model:DynamicModel = RecurrentStateSpaceModel(**rssm_conf)
        cls.pred_steps = 5
        cls.input = {
            "initial": torch.randn(1, rssm_conf.hidden_size+rssm_conf.latent_size),
            "actions": torch.randn(1, cls.pred_steps, rssm_conf.action_size),
            "observations": torch.randn(1, cls.pred_steps, 3, 64, 64)
        }
        
    @classmethod
    def teardown_class(cls):
        del cls.model

    def test_train_loss(self):
        reseed_everything(42)
        self.model.train()
        loss = self.model(self.input)
        assert loss.item() == pytest.approx(16.0120, rel=1e-5)
    
    def test_inference(self):
        reseed_everything(42)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.input, self.pred_steps, use_mle=False)
        assert predictions.shape == (1, self.pred_steps,self.model.hidden_size+self.model.latent_size), \
        f"predictions shape is {predictions.shape}"
        # load prediction from file
        expected_pred_1 = torch.load("tests/fixtures/tensors/rssm_prediction_stochastic_1.pth")
        assert torch.allclose(predictions, expected_pred_1, atol=1e-5), f"predictions are not equal"
        with torch.no_grad():
            predictions = self.model(self.input, self.pred_steps, use_mle=False)
        expected_pred_2 = torch.load("tests/fixtures/tensors/rssm_prediction_stochastic_2.pth")
        assert torch.allclose(predictions, expected_pred_2, atol=1e-5), f"predictions are not equal"
        with torch.no_grad():
            predictions = self.model(self.input, self.pred_steps, use_mle=True)
        expected_pred_3 = torch.load("tests/fixtures/tensors/rssm_prediction_deterministic.pth")
        assert torch.allclose(predictions, expected_pred_3, atol=1e-5), f"predictions are not equal"