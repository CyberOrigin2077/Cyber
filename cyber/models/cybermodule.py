from abc import ABC, abstractmethod
from typing import TypeVar, Callable
import torch

T = TypeVar("T", bound="CyberModule")


class CyberModule(ABC, torch.nn.Module):
    """
    All modules provided by this package should inherit from this class.
    This class defines the common interface for all modules so that they can be easily used.
    Currently subclasses this module only interfaces with torch (by inheritting from torch.nn.module),
    but in the future they may be extended to support other frameworks.
    """

    _forward: Callable

    def __init__(self):
        super().__init__()
        self._forward = self.compute_training_loss  # nn.Module defaults to training mode

    def forward(self, *args, **kwargs):
        """
        forward is stateful.
        When run in 'train' mode, returns a default loss value.
        When run in 'eval' mode, returns the output of the module.
        """
        return self._forward(*args, **kwargs)

    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.

        In addition to Pytorch's default behavior, this method also sets the
        forward method to the compute_training_loss method when in training mode.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            CyberModule: self
        """
        torch.nn.Module.train(self)
        if mode:
            self._forward = self.compute_training_loss
        else:
            self._forward = self.forward_method
        return self

    def forward_method(self, *args, **kwargs):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forword_method" function')

    @abstractmethod
    def compute_training_loss(self, *args, **kwargs):
        """
        Compute the training loss for the module.
        Cyber modules should provide default loss functions to simplify training.
        """
        pass

    @abstractmethod
    def get_train_collator(self, *args, **kwargs) -> Callable:
        """
        Get the collator for the module.
        Cyber modules should provide a default collator that's compatible
        with compute_training_loss to simplify training.
        """
        pass

    # @classmethod
    # @abstractmethod
    # def from_pretrained(cls, *args, **kwargs):
    #     '''
    #     Load the module from a pretrained checkpoint.
    #     Cyber modules should provide a method to load weights and configs, ideally from a single url.
    #     '''
    #     pass
