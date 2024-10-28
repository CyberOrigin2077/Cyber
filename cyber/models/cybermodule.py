from abc import ABC, abstractmethod
import torch


class CyberModule(ABC, torch.nn.Module):
    """
    All modules provided by this package should inherit from this class.
    This class defines the common interface for all modules so that they can be easily used.
    Currently subclasses this module only interfaces with torch (by inheritting from torch.nn.module),
    but in the future they may be extended to support other frameworks.
    """

    @abstractmethod
    def compute_training_loss(self, *args, **kwargs):
        """
        Compute the training loss for the module.
        Cyber modules should provide default loss functions to simplify training.
        """
        pass

    @abstractmethod
    def get_train_collator(self, *args, **kwargs):
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
