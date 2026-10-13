from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CPPNCoordinateFrame:
    """Placement of a CPPN-modeled crop inside a shared full-image grid."""

    origin: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    full_spatial_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        num_dims = len(self.spatial_shape)
        if num_dims == 0:
            raise ValueError("CPPN coordinate frames require at least one dimension.")
        if len(self.origin) != num_dims or len(self.full_spatial_shape) != num_dims:
            raise ValueError("CPPN coordinate frame tuples must have matching lengths.")
        for axis, (origin, size, full_size) in enumerate(
            zip(self.origin, self.spatial_shape, self.full_spatial_shape)
        ):
            if origin < 0:
                raise ValueError(f"Coordinate-frame origin[{axis}] must be >= 0.")
            if size <= 0 or full_size <= 0:
                raise ValueError("Coordinate-frame spatial sizes must be > 0.")
            if origin + size > full_size:
                raise ValueError(
                    f"Coordinate frame exceeds full image at axis {axis}: "
                    f"origin={origin}, size={size}, full_size={full_size}."
                )

    @classmethod
    def root(cls, spatial_shape: tuple[int, ...]) -> "CPPNCoordinateFrame":
        return cls(
            origin=(0,) * len(spatial_shape),
            spatial_shape=spatial_shape,
            full_spatial_shape=spatial_shape,
        )

    def crop(self, spatial_slices: tuple[slice, ...]) -> "CPPNCoordinateFrame":
        if len(spatial_slices) != len(self.spatial_shape):
            raise ValueError("Crop slices must match the coordinate-frame rank.")

        cropped_origin = []
        cropped_shape = []
        for axis, (selector, size) in enumerate(
            zip(spatial_slices, self.spatial_shape)
        ):
            start, stop, step = selector.indices(size)
            if step != 1:
                raise ValueError(
                    f"Coordinate-frame crop step must be 1 at axis {axis}."
                )
            if stop <= start:
                raise ValueError("Coordinate-frame crops must be non-empty.")
            cropped_origin.append(self.origin[axis] + start)
            cropped_shape.append(stop - start)

        return CPPNCoordinateFrame(
            origin=tuple(cropped_origin),
            spatial_shape=tuple(cropped_shape),
            full_spatial_shape=self.full_spatial_shape,
        )


@dataclass(frozen=True)
class CPPNCoordinateData:
    """Tensor-like CPPN input that preserves its global frame when cropped."""

    data: np.ndarray | torch.Tensor
    coordinate_frame: CPPNCoordinateFrame

    def __post_init__(self) -> None:
        num_spatial_dims = len(self.coordinate_frame.spatial_shape)
        if self.data.ndim < num_spatial_dims + 2:
            raise ValueError(
                "CPPN coordinate data must have batch, spatial, and channel dimensions."
            )
        spatial_shape = tuple(int(v) for v in self.data.shape[1 : 1 + num_spatial_dims])
        if spatial_shape != self.coordinate_frame.spatial_shape:
            raise ValueError(
                "CPPN coordinate-data shape does not match its coordinate frame."
            )

    @classmethod
    def root(cls, data: np.ndarray | torch.Tensor) -> "CPPNCoordinateData":
        if data.ndim < 3:
            raise ValueError(
                "CPPN coordinate data must have shape "
                "(batch, *spatial_dims, channels)."
            )
        spatial_shape = tuple(int(v) for v in data.shape[1:-1])
        return cls(
            data=data,
            coordinate_frame=CPPNCoordinateFrame.root(spatial_shape),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __getitem__(self, index: object) -> "CPPNCoordinateData":
        if not isinstance(index, tuple):
            index = (index,)
        if Ellipsis in index:
            raise ValueError("Ellipsis indexing is not supported for CPPN coordinate data.")
        if len(index) > self.data.ndim:
            raise IndexError("Too many indices for CPPN coordinate data.")

        padded_index = (*index, *((slice(None),) * (self.data.ndim - len(index))))
        if not isinstance(padded_index[0], slice):
            raise ValueError("CPPN coordinate-data cropping must preserve the batch axis.")

        num_spatial_dims = len(self.coordinate_frame.spatial_shape)
        spatial_slices = padded_index[1 : 1 + num_spatial_dims]
        if not all(isinstance(selector, slice) for selector in spatial_slices):
            raise ValueError("CPPN coordinate-data cropping must preserve spatial axes.")

        return CPPNCoordinateData(
            data=self.data[padded_index],
            coordinate_frame=self.coordinate_frame.crop(spatial_slices),
        )


def coordinate_grid_from_frame(
    coordinate_frame: CPPNCoordinateFrame,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build raw coordinates for a crop in its full-image normalized frame."""

    axes = []
    for origin, size, full_size in zip(
        coordinate_frame.origin,
        coordinate_frame.spatial_shape,
        coordinate_frame.full_spatial_shape,
    ):
        if full_size <= 1:
            axes.append(torch.zeros(size, device=device, dtype=dtype))
        else:
            indices = torch.arange(origin, origin + size, device=device, dtype=dtype)
            axes.append(2.0 * indices / float(full_size - 1) - 1.0)
    mesh = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack(mesh, dim=-1)
    return coords.reshape(-1, len(coordinate_frame.spatial_shape))
