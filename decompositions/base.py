from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import torch


# First element of the tuple is a tuple of indices of the parts being joined,
# the second element is the index of the resulting part,
# the third element is the size of the resulting part.
DecompositionEdge = tuple[tuple[int, ...], int, int]


class Decomposition(ABC):
    @abstractmethod
    def parts(
        self, obj: np.ndarray | torch.Tensor
    ) -> Iterator[np.ndarray | torch.Tensor]:
        """Enumerate the parts of the decomposition (nodes in the decomposition tree).

        Args:
            obj (np.ndarray | torch.Tensor): The object to decompose.

        Yields:
            Iterator[np.ndarray | torch.Tensor]: An iterator over the parts of the decomposition.
        """
        pass

    @abstractmethod
    def joins(self, obj: np.ndarray | torch.Tensor) -> Iterator[DecompositionEdge]:
        """Enumerate the joins of the decomposition (edges in the decomposition tree).

        Args:
            obj (np.ndarray | torch.Tensor): The object to decompose.

        Yields:
            Iterator[DecompositionEdge]: An iterator over the joins of the decomposition.
        """
        pass
