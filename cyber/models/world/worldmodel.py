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


class AutoEncoder(CyberModule):
    """
    template for dynamic models
    """

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        prototype for the encoder.

        Args:
        TODO

        Returns:

        """
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        """
        prototype for the decoder.

        Args:
        TODO

        Returns:

        """
        pass

    def forward_method(self, *args, **kwargs):
        """
        a forward pass through the autoencoder is the same as encoding.

        """
        return self.encode(*args, **kwargs)
