import numpy as np
import torch

from conditional_compressors.base import ConditionalCompressor
from decompositions.hierarchical import HierarchicalDecomposition
from rewards.base import DecompositionReward, JoinToComposition, Model
from two_part_codes.base import TwoPartCode


class DecompositionLevelMeanReward(DecompositionReward):

    def __init__(
        self,
        two_part_code: TwoPartCode[Model],
        conditional_compressor: ConditionalCompressor[Model],
        decomposition: HierarchicalDecomposition,
        normalize_by_part_size: bool = False,
        normalize_reward: bool = False,
        obj_shape: tuple[int, ...] | None = None,
        obj_num_symbols: int | None = None,
        curriculum_steps: int = 0,
        p: float = 1.0,
    ) -> None:
        """Reward defined from my compositionality metric applied on a hierarchically recursive decomposition of the object.

        Args:
            two_part_code (TwoPartCode[Model]): Two-part-code model fitting `m_x` from `x`.
            conditional_compressor (ConditionalCompressor[Model]): Conditional compressor of one model given others `K(m_x | m_{x_a}, m_{x_b}, ...)`.
            decomposition (HierarchicalDecomposition): How to do the recursive decomposition of the object.
            normalize_by_part_size (bool, optional): Whether or not to normalize `K(m_x | m_{x_a}, m_{x_b}, ...)` terms by len(x). Defaults to False.
            normalize_reward (bool, optional): Whether or not to subtract the reward of a typical random object. Defaults to False.
            normalization_quantile (float, optional): Which quantile of the random object reward to use for normalization. Defaults to 0.95.
            obj_shape (tuple[int, ...] | None, optional): The shape of the object (excluding batch dimension), necessary when using reward normalization. Defaults to None.
            obj_num_symbols (int | None, optional): Number of symbols in the object, necessary when using reward normalization. Defaults to None.
            curriculum_steps (int, optional): Number of curriculum steps to use. Defaults to 0.
            p (float, optional): The L-p norm to use when aggregating level rewards. Defaults to 1.0 (mean).
        """
        assert isinstance(decomposition, HierarchicalDecomposition)
        super().__init__(
            two_part_code=two_part_code,
            conditional_compressor=conditional_compressor,
            decomposition=decomposition,
            normalize_by_part_size=normalize_by_part_size,
            normalize_reward=normalize_reward,
            obj_shape=obj_shape,
            obj_num_symbols=obj_num_symbols,
        )
        self.curriculum_steps = curriculum_steps
        self.current_step = 0
        self.p = p

    def compositions_to_reward(
        self,
        compositions: JoinToComposition,
        obj_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Map the compositionality scores K(m_x | m_{x_a}, ..., m_{x_n}) values to a reward.

        Args:
            compositions (JoinToComposition): A dictionary mapping decomposition edges (tuples of source node indices, sink node index) to their batched compositionality scores (batch of scalars).
            obj_shape (tuple[int, ...]): Shape of the object that was decomposed.

        Returns:
            torch.Tensor: Batch of scalar rewards for each object in the batch.
        """
        level_rewards = self.compositions_to_level_rewards(
            compositions=compositions,
            obj_shape=obj_shape,
        )
        if self.curriculum_steps > 0:
            weights = self.weights_for_current_curriculum_step(
                num_layers=level_rewards.shape[1]
            ).to(level_rewards.device)
            reward = (
                torch.linalg.vector_norm(
                    level_rewards * weights[None], ord=self.p, dim=1
                )
                * level_rewards.shape[1]
                / ((level_rewards.shape[1] ** (1 / self.p)))
            )
        else:
            reward = torch.linalg.vector_norm(level_rewards, ord=self.p, dim=1) / (
                level_rewards.shape[1] ** (1 / self.p)
            )
        return reward

    def compositions_to_level_rewards(
        self,
        compositions: JoinToComposition,
        obj_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Map the compositionality scores K(m_x | m_{x_a}, ..., m_{x_n}) values to rewards at each level of the hierarchy.

        Args:
            compositions (JoinToComposition): A dictionary mapping decomposition edges (tuples of source node indices, sink node index) to their batched compositionality scores (batch of scalars).
            obj_shape (tuple[int, ...]): Shape of the object that was decomposed (excluding batch dimension).

        Returns:
            torch.Tensor: Batch of rewards for each object in the batch (axis 0) and at each level of the hierarchy (axis 1).
        """
        assert isinstance(self.decomposition, HierarchicalDecomposition)
        num_parts_per_level = self.decomposition.num_parts_per_level_from_obj_shape(
            obj_shape
        )
        num_join_levels = len(num_parts_per_level) - 1
        if num_join_levels <= 0:
            raise ValueError(
                "Hierarchical decomposition has no joins for the provided object shape."
            )

        level_cutoffs = np.cumsum(num_parts_per_level[:num_join_levels])
        level_rewards = [[] for _ in range(num_join_levels)]
        for (_, sink_idx), join_composition in compositions.items():
            level = int(np.searchsorted(level_cutoffs, sink_idx, side="right") - 1)
            level_rewards[level].append(join_composition)

        level_rewards = [
            torch.stack(level_reward, dim=0).mean(dim=0)
            for level_reward in level_rewards
        ]
        level_rewards = torch.stack(level_rewards, dim=1)
        return level_rewards

    def curriculum_step(self, step):
        self.current_step = step

    def weights_for_current_curriculum_step(self, num_layers: int) -> torch.Tensor:
        """Gradually increases the weights of later layers in the hierarchy over the course of training."""
        t = self.current_step / self.curriculum_steps
        t = min(max(t, 1e-6), 1.0)  # From epsilon (one-hot at index 0) to 1.0 (uniform)
        r = 1 - (1 - t) ** 2
        alpha = -torch.log(torch.tensor(r))
        logits = -alpha * torch.arange(num_layers, dtype=torch.float32)
        w = torch.softmax(logits, dim=0)
        normalization = w[0] * num_layers
        w = (
            w / normalization
        )  # So that the reward contributions of all layers are strictly increasing from 0.0 to 1.0 / num_layers
        return w
