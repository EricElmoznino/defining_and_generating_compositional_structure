from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy as np
import torch


Model = TypeVar("Model")


class TwoPartCode(Generic[Model], ABC):
    """Abstract base class for two-part codes.

    This class defines the interface for two-part codes, which are used to
    compress data into a structured model and unstructured random component.
    """

    def prepare_for_decomposition(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Attach compressor-specific metadata before decomposition, if needed."""
        return data

    @abstractmethod
    def compress(
        self, data: np.ndarray | torch.Tensor
    ) -> tuple[Model, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Compress the data into a structured model and unstructured random component.

        Args:
            data (np.ndarray | torch.Tensor): The data to compress, with a leading batch dimension.

        Returns:
            tuple[Model, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]: A tuple containing a batch of structured models, their complexities, and the complexities of the unstructured random components.
        """
        pass
