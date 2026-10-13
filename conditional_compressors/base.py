from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy as np
import torch


Model = TypeVar("Model")


class ConditionalCompressor(Generic[Model], ABC):
    """Abstract base class for conditional compressors.

    This class defines the interface for conditional compressors,
    which are used to compress one model given others.
    """

    @abstractmethod
    def compress(
        self, model: Model, given_models: list[Model]
    ) -> np.ndarray | torch.Tensor:
        """Compress the model given some other models, and output the compression length.

        Args:
            model (Model): A batch of models to compress.
            given_models (list[Model]): A list of batched models that the compression is conditioned on.

        Returns:
            np.ndarray | torch.Tensor: The compression length of the model given the other models, batched.
        """
        pass
