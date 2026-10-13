from abc import ABC, abstractmethod
from typing import TypeVar

import torch
import torch.nn as nn

from conditional_compressors.base import ConditionalCompressor
from decompositions.base import Decomposition, DecompositionEdge
from two_part_codes.base import TwoPartCode


class Reward(nn.Module, ABC):
    """Abstract base class for rewards."""

    @abstractmethod
    def __call__(self, obj: torch.Tensor) -> torch.Tensor:
        pass

    def curriculum_step(self, step: int) -> None:
        """Update any curriculum learning parameters based on the current training step.

        Args:
            step (int): Current training step.
        """
        pass


Model = TypeVar("Model")
JoinToComposition = dict[DecompositionEdge, torch.Tensor]


class DecompositionReward(Reward, ABC):

    def __init__(
        self,
        two_part_code: TwoPartCode[Model],
        conditional_compressor: ConditionalCompressor[Model],
        decomposition: Decomposition,
        normalize_by_part_size: bool = False,
        normalize_reward: bool = False,
        normalization_quantile: float = 0.95,
        obj_shape: tuple[int, ...] | None = None,
        obj_num_symbols: int | None = None,
    ) -> None:
        """Reward defined from my compositionality metric applied on a recursive decomposition of the object.

        Args:
            two_part_code (TwoPartCode[Model]): Two-part-code model fitting `m_x` from `x`.
            conditional_compressor (ConditionalCompressor[Model]): Conditional compressor of one model given others `K(m_x | m_{x_a}, m_{x_b}, ...)`.
            decomposition (Decomposition): How to do the recursive decomposition of the object.
            normalize_by_part_size (bool, optional): Whether or not to normalize `K(m_x | m_{x_a}, m_{x_b}, ...)` terms by len(x). Defaults to False.
            normalize_reward (bool, optional): Whether or not to subtract the reward of a typical random object. Defaults to False.
            normalization_quantile (float, optional): Which quantile of the random object reward to use for normalization. Defaults to 0.95.
            obj_shape (tuple[int, ...] | None, optional): The shape of the object (excluding batch dimension), necessary when using reward normalization. Defaults to None.
            obj_num_symbols (int | None, optional): Number of symbols in the object, necessary when using reward normalization. Defaults to None.
        """
        super().__init__()

        self.two_part_code = two_part_code
        self.conditional_compressor = conditional_compressor
        self.decomposition = decomposition
        self.normalize_by_part_size = normalize_by_part_size

        self.random_obj_reward = 0.0
        if normalize_reward:
            assert (obj_shape is not None) and (
                obj_num_symbols is not None
            ), "To use centered rewards, the shape and number of symbols of the objects must be provided."
            self.random_obj_reward = (
                self(torch.randint(0, obj_num_symbols, (1000, *obj_shape)))
                .quantile(normalization_quantile)
                .item()
            )

    def __call__(
        self,
        obj: torch.Tensor,
        return_compositions: bool = False,
        return_models: bool = False,
    ) -> (
        tuple[torch.Tensor]
        | tuple[torch.Tensor, JoinToComposition]
        | tuple[torch.Tensor, list[Model]]
        | tuple[torch.Tensor, JoinToComposition, list[Model]]
    ):
        decomposition_obj = self.two_part_code.prepare_for_decomposition(obj)
        part_models = [
            self.two_part_code.compress(part)[0]
            for part in self.decomposition.parts(decomposition_obj)
        ]

        def compress(children_part_idxs, parent_part_idx, part_size):
            if not self.normalize_by_part_size:
                part_size = 1
            return (
                self.conditional_compressor.compress(
                    part_models[parent_part_idx],
                    [part_models[i] for i in children_part_idxs],
                )
                / part_size
            )

        compositions = {
            (children_part_idxs, parent_part_idx): compress(
                children_part_idxs, parent_part_idx, part_size
            )
            for children_part_idxs, parent_part_idx, part_size in self.decomposition.joins(
                decomposition_obj
            )
        }

        reward = self.compositions_to_reward(compositions, obj.shape[1:])
        reward -= self.random_obj_reward
        reward = torch.clamp(reward, min=0.0)

        if return_compositions and return_models:
            return reward, compositions, part_models
        elif return_compositions:
            return reward, compositions
        elif return_models:
            return reward, part_models
        return reward

    @abstractmethod
    def compositions_to_reward(
        self,
        compositions: JoinToComposition,
        obj_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Map the compositionality scores K(m_x | m_{x_a}, ..., m_{x_n}) values to a reward.

        Args:
            compositions (JoinToComposition): A dictionary mapping decomposition edges (tuples of source node indices, sink node index) to their batched compositionality scores (batch of scalars).
            obj_shape (tuple[int, ...]): Shape of the object that was decomposed (excluding batch dimension).

        Returns:
            torch.Tensor: Batch of scalar rewards for each object in the batch.
        """
        pass
