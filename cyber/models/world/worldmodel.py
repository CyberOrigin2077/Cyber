from cyber.models import CyberModule

from abc import abstractmethod
from typing import Any


class DynamicModel(CyberModule):
    """
    template for dynamic models
    """

    @abstractmethod
    def forward_method(self, inputs: Any, frames_to_generate: int, *args, **kwargs):
        """
        prototype for forward pass through the dynamic model.

        Args:
        input(any): the input to the dynamic model, usually the output of an encoder.
        frames_to_generate(int): number of new frames to generate

        Returns:
        torch.Tensor: generated frames
        """
        pass
