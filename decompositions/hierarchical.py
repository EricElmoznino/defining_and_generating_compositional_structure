import math
from numbers import Integral
from typing import Iterator, Sequence

import numpy as np
import torch

from decompositions.base import Decomposition, DecompositionEdge


class HierarchicalDecomposition(Decomposition):
    """Hierarchical decomposition over one or more spatial dimensions.

    Backward-compatible default behavior is unchanged for strings:
    - `num_spatial_dims=1`, `num_channel_dims=0`,
    - scalar `min_part_size` and `branching_factor`.

    For tensors with spatial + channel dimensions (e.g. images), set
    `num_spatial_dims` and `num_channel_dims` so only spatial axes are split.
    """

    def __init__(
        self,
        min_part_size: int | tuple[int, ...] = 1,
        branching_factor: int | tuple[int, ...] = 2,
        num_spatial_dims: int = 1,
        num_channel_dims: int = 0,
    ):
        """Initialize hierarchical decomposition settings.

        Args:
            min_part_size (int | tuple[int, ...]): Minimum part size per spatial
                axis. If int, the same value is used for all spatial axes.
            branching_factor (int | tuple[int, ...]): Number of child parts per
                axis at each level. If int, the same value is used for all axes.
            num_spatial_dims (int, optional): Number of spatial axes in the
                input (excluding batch and channels). Defaults to 1.
            num_channel_dims (int, optional): Number of trailing non-spatial axes
                that should not be split (e.g., 1 for a channel axis). Defaults to 0.
        """
        if num_spatial_dims < 1:
            raise ValueError("num_spatial_dims must be >= 1.")
        if num_channel_dims < 0:
            raise ValueError("num_channel_dims must be >= 0.")

        self.min_part_size = min_part_size
        self.branching_factor = branching_factor
        self.num_spatial_dims = int(num_spatial_dims)
        self.num_channel_dims = int(num_channel_dims)

        self._min_part_size = self._normalize_per_dim(
            value=min_part_size, name="min_part_size"
        )
        self._branching_factor = self._normalize_per_dim(
            value=branching_factor, name="branching_factor"
        )
        if any(v <= 0 for v in self._min_part_size):
            raise ValueError("All min_part_size values must be > 0.")
        if any(v <= 1 for v in self._branching_factor):
            raise ValueError("All branching_factor values must be > 1.")

    @property
    def min_part_size_per_dim(self) -> tuple[int, ...]:
        return self._min_part_size

    @property
    def branching_factor_per_dim(self) -> tuple[int, ...]:
        return self._branching_factor

    def _normalize_per_dim(
        self, value: int | tuple[int, ...], name: str
    ) -> tuple[int, ...]:
        if isinstance(value, Integral):
            return (int(value),) * self.num_spatial_dims
        if isinstance(value, Sequence):
            if len(value) != self.num_spatial_dims:
                raise ValueError(
                    f"{name} tuple must have length num_spatial_dims={self.num_spatial_dims}, "
                    f"got {len(value)}."
                )
            if not all(isinstance(v, Integral) for v in value):
                raise TypeError(f"All {name} tuple values must be integers.")
            return tuple(int(v) for v in value)
        raise TypeError(
            f"{name} must be an int or tuple[int, ...], got {type(value).__name__}."
        )

    def _spatial_shape_from_full_shape(
        self,
        full_shape: tuple[int, ...],
        has_batch_dim: bool,
    ) -> tuple[int, ...]:
        expected_rank = (
            (1 if has_batch_dim else 0)
            + self.num_spatial_dims
            + self.num_channel_dims
        )
        assert len(full_shape) == expected_rank, (
            f"Expected rank={expected_rank} "
            f"(batch={has_batch_dim}, spatial={self.num_spatial_dims}, channel_dims={self.num_channel_dims}), "
            f"got shape={full_shape}."
        )
        start = 1 if has_batch_dim else 0
        end = start + self.num_spatial_dims
        return tuple(int(v) for v in full_shape[start:end])

    def _num_levels_from_spatial_shape(self, spatial_shape: tuple[int, ...]) -> int:
        levels_per_dim = []
        for axis, size in enumerate(spatial_shape):
            min_size = self._min_part_size[axis]
            branch = self._branching_factor[axis]
            assert (
                size >= min_size
            ), f"Spatial size at axis {axis} must be >= min_part_size[{axis}]={min_size}."
            assert (
                size % min_size == 0
            ), f"Spatial size at axis {axis} must be divisible by min_part_size[{axis}]={min_size}."
            ratio = size / min_size
            levels_minus_one = math.log(ratio, branch)
            assert levels_minus_one.is_integer(), (
                f"Spatial size at axis {axis} must equal "
                f"min_part_size[{axis}] * branching_factor[{axis}]**k for some integer k."
            )
            levels_per_dim.append(int(levels_minus_one) + 1)

        assert all(level == levels_per_dim[0] for level in levels_per_dim), (
            "All spatial dimensions must imply the same number of hierarchical levels. "
            f"Got levels per dim: {levels_per_dim}."
        )
        return int(levels_per_dim[0])

    @staticmethod
    def _flat_index(coords: tuple[int, ...], shape: tuple[int, ...]) -> int:
        # Row-major flattening: last axis varies fastest.
        idx = 0
        for c, dim in zip(coords, shape):
            idx = idx * dim + c
        return int(idx)

    def _part_shape_for_level(self, level: int) -> tuple[int, ...]:
        return tuple(
            self._min_part_size[axis] * (self._branching_factor[axis] ** level)
            for axis in range(self.num_spatial_dims)
        )

    def parts(
        self, obj: np.ndarray | torch.Tensor
    ) -> Iterator[np.ndarray | torch.Tensor]:
        spatial_shape = self._spatial_shape_from_full_shape(
            tuple(obj.shape), has_batch_dim=True
        )
        num_levels = self._num_levels_from_spatial_shape(spatial_shape)
        channel_slices = (slice(None),) * self.num_channel_dims

        for level in range(num_levels):
            part_shape = self._part_shape_for_level(level)
            grid_shape = tuple(
                spatial // part for spatial, part in zip(spatial_shape, part_shape)
            )
            for grid_coord in np.ndindex(*grid_shape):
                spatial_slices = tuple(
                    slice(coord * part_size, (coord + 1) * part_size)
                    for coord, part_size in zip(grid_coord, part_shape)
                )
                index = (slice(None), *spatial_slices, *channel_slices)
                part = obj[index]
                yield part

    def joins(self, obj: np.ndarray | torch.Tensor) -> Iterator[DecompositionEdge]:
        spatial_shape = self._spatial_shape_from_full_shape(
            tuple(obj.shape), has_batch_dim=True
        )
        num_levels = self._num_levels_from_spatial_shape(spatial_shape)
        num_nodes_per_level = [
            int(
                np.prod(
                    tuple(
                        spatial // part
                        for spatial, part in zip(
                            spatial_shape, self._part_shape_for_level(level)
                        )
                    ),
                    dtype=np.int64,
                )
            )
            for level in range(num_levels)
        ]
        level_start_idx = [0]
        for n in num_nodes_per_level[:-1]:
            level_start_idx.append(level_start_idx[-1] + n)

        for level in range(num_levels - 1):
            child_part_shape = self._part_shape_for_level(level)
            parent_part_shape = self._part_shape_for_level(level + 1)
            child_grid_shape = tuple(
                spatial // part
                for spatial, part in zip(spatial_shape, child_part_shape)
            )
            parent_grid_shape = tuple(
                spatial // part
                for spatial, part in zip(spatial_shape, parent_part_shape)
            )

            child_level_start = level_start_idx[level]
            parent_level_start = level_start_idx[level + 1]
            part_size = int(np.prod(parent_part_shape, dtype=np.int64))

            for parent_grid_coord in np.ndindex(*parent_grid_shape):
                parent_flat = self._flat_index(parent_grid_coord, parent_grid_shape)
                parent_idx = parent_level_start + parent_flat
                origin_idxs = []
                for child_offset in np.ndindex(*self._branching_factor):
                    child_grid_coord = tuple(
                        parent_grid_coord[axis] * self._branching_factor[axis]
                        + child_offset[axis]
                        for axis in range(self.num_spatial_dims)
                    )
                    child_flat = self._flat_index(child_grid_coord, child_grid_shape)
                    origin_idxs.append(child_level_start + child_flat)
                yield tuple(origin_idxs), parent_idx, part_size

    def num_levels(self, obj: np.ndarray | torch.Tensor) -> int:
        spatial_shape = self._spatial_shape_from_full_shape(
            tuple(obj.shape), has_batch_dim=True
        )
        return self._num_levels_from_spatial_shape(spatial_shape)

    def num_levels_from_obj_shape(self, obj_shape: tuple[int, ...]) -> int:
        spatial_shape = self._spatial_shape_from_full_shape(
            tuple(obj_shape), has_batch_dim=False
        )
        return self._num_levels_from_spatial_shape(spatial_shape)

    def num_parts_per_level_from_obj_shape(
        self, obj_shape: tuple[int, ...]
    ) -> list[int]:
        spatial_shape = self._spatial_shape_from_full_shape(
            tuple(obj_shape), has_batch_dim=False
        )
        num_levels = self._num_levels_from_spatial_shape(spatial_shape)
        return [
            int(
                np.prod(
                    tuple(
                        spatial // part
                        for spatial, part in zip(
                            spatial_shape, self._part_shape_for_level(level)
                        )
                    ),
                    dtype=np.int64,
                )
            )
            for level in range(num_levels)
        ]


def _run_spot_checks() -> None:
    print("Running spot checks for HierarchicalDecomposition...")

    # 1D string compatibility check.
    string = torch.arange(2 * 8).reshape(2, 8)
    dec_1d = HierarchicalDecomposition(min_part_size=1, branching_factor=2)
    parts_1d = list(dec_1d.parts(string))
    joins_1d = list(dec_1d.joins(string))
    assert dec_1d.num_levels(string) == 4
    assert len(parts_1d) == 8 + 4 + 2 + 1
    assert len(joins_1d) == 4 + 2 + 1
    assert parts_1d[0].shape == (2, 1)
    assert parts_1d[8].shape == (2, 2)
    assert parts_1d[-1].shape == (2, 8)
    assert joins_1d[0][0] == (0, 1) and joins_1d[0][1] == 8 and joins_1d[0][2] == 2
    print("  [OK] 1D behavior matches previous hierarchical decomposition semantics.")

    # 2D image-style decomposition check.
    image = torch.arange(2 * 4 * 4 * 3).reshape(2, 4, 4, 3)
    dec_2d = HierarchicalDecomposition(
        min_part_size=(1, 1),
        branching_factor=(2, 2),
        num_spatial_dims=2,
        num_channel_dims=1,
    )
    parts_2d = list(dec_2d.parts(image))
    joins_2d = list(dec_2d.joins(image))
    assert dec_2d.num_levels(image) == 3
    assert dec_2d.num_levels_from_obj_shape((4, 4, 3)) == 3
    assert dec_2d.num_parts_per_level_from_obj_shape((4, 4, 3)) == [16, 4, 1]
    assert len(parts_2d) == 16 + 4 + 1
    assert len(joins_2d) == 4 + 1
    assert parts_2d[0].shape == (2, 1, 1, 3)
    assert parts_2d[16].shape == (2, 2, 2, 3)
    assert parts_2d[-1].shape == (2, 4, 4, 3)
    assert len(joins_2d[0][0]) == 4
    assert joins_2d[0][2] == 4  # parent part has shape 2x2
    assert joins_2d[-1][2] == 16  # root part has shape 4x4
    print("  [OK] 2D decomposition splits spatial axes and preserves channel axis.")

    # Scalar broadcasting across dimensions.
    dec_broadcast = HierarchicalDecomposition(
        min_part_size=1,
        branching_factor=2,
        num_spatial_dims=2,
        num_channel_dims=1,
    )
    assert dec_broadcast.min_part_size_per_dim == (1, 1)
    assert dec_broadcast.branching_factor_per_dim == (2, 2)
    print("  [OK] Scalar min_part_size/branching_factor broadcast across spatial dims.")

    print("All spot checks passed.")


if __name__ == "__main__":
    _run_spot_checks()
