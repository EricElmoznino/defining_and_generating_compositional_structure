import torch
import torch.nn.functional as F

from rewards.base import Reward


class ToyRewardModes(Reward):
    """A toy reward for quick debugging with a couple of modes."""

    def __init__(self, vocab_size: int, string_length: int, n_modes: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.string_length = string_length
        self.n_modes = n_modes
        self.register_buffer(
            "modes",
            torch.randint(0, vocab_size, (n_modes, string_length), dtype=torch.float32),
        )

    def __call__(self, obj: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(obj.float(), self.modes, p=0)
        closest_mode = distances.argmin(dim=1)
        rewards = torch.exp(-distances[torch.arange(distances.shape[0]), closest_mode])
        return rewards


class ToyRewardCPPNBranchingNetwork(Reward):
    """A toy reward for sparse branching CPPN structures.

    A foreground network starts at a small source gate on the left edge. Reward is
    earned by growing the source-connected component toward many small terminal
    gates on the top, right, and bottom edges. There is no target interior layout:
    different trunks, branches, and detours can reach the same terminals.
    """

    def __init__(
        self,
        vocab_size: int,
        spatial_shape: tuple[int, int],
        n_channels: int = 1,
        path_token: int | None = None,
        source_gate_size: int = 9,
        terminal_gate_size: int = 5,
        terminal_gates_per_side: int = 5,
        path_density_target: float = 0.14,
        max_path_density: float = 0.34,
        approach_radius_fraction: float = 0.22,
        coverage_grid_size: int = 6,
        connectivity_steps: int | None = None,
        terminal_hit_weight: float = 0.40,
        terminal_approach_weight: float = 0.25,
        coverage_weight: float = 0.20,
        connected_foreground_weight: float = 0.10,
        source_weight: float = 0.05,
    ) -> None:
        super().__init__()
        assert vocab_size > 1, "vocab_size must be > 1."
        assert len(spatial_shape) == 2, (
            "ToyRewardCPPNBranchingNetwork expects 2D objects."
        )
        assert n_channels > 0, "n_channels must be > 0."
        assert source_gate_size > 0, "source_gate_size must be > 0."
        assert terminal_gate_size > 0, "terminal_gate_size must be > 0."
        assert terminal_gates_per_side > 0, "terminal_gates_per_side must be > 0."
        assert 0.0 < path_density_target < max_path_density < 1.0, (
            "Require 0 < path_density_target < max_path_density < 1."
        )
        assert approach_radius_fraction > 0.0, (
            "approach_radius_fraction must be > 0."
        )
        assert coverage_grid_size > 0, "coverage_grid_size must be > 0."
        assert (
            connectivity_steps is None or connectivity_steps > 0
        ), "connectivity_steps must be > 0."

        if path_token is None:
            path_token = vocab_size - 1
        assert 0 <= path_token < vocab_size, "path_token must be in the vocabulary."

        weights = torch.tensor(
            [
                terminal_hit_weight,
                terminal_approach_weight,
                coverage_weight,
                connected_foreground_weight,
                source_weight,
            ],
            dtype=torch.float32,
        )
        assert torch.all(weights >= 0.0).item(), (
            "All reward weights must be non-negative."
        )
        assert (weights.sum() > 0.0).item(), "At least one reward weight must be > 0."

        self.vocab_size = vocab_size
        self.spatial_shape = tuple(spatial_shape)
        self.n_channels = n_channels
        self.path_token = path_token
        self.path_density_target = path_density_target
        self.max_path_density = max_path_density
        self.approach_radius_fraction = approach_radius_fraction
        self.coverage_grid_size = coverage_grid_size
        self.connectivity_steps = connectivity_steps

        normalized_weights = weights / weights.sum()
        self.terminal_hit_weight = float(normalized_weights[0])
        self.terminal_approach_weight = float(normalized_weights[1])
        self.coverage_weight = float(normalized_weights[2])
        self.connected_foreground_weight = float(normalized_weights[3])
        self.source_weight = float(normalized_weights[4])

        source_mask, terminal_masks, terminal_distances = self._build_gate_buffers(
            spatial_shape=spatial_shape,
            source_gate_size=source_gate_size,
            terminal_gate_size=terminal_gate_size,
            terminal_gates_per_side=terminal_gates_per_side,
        )
        self.register_buffer("source_mask", source_mask)
        self.register_buffer("terminal_masks", terminal_masks)
        self.register_buffer("terminal_distances", terminal_distances)

    def __call__(self, obj: torch.Tensor) -> torch.Tensor:
        assert obj.ndim == 4, "Expected input shape (batch, height, width, channels)."
        assert (
            tuple(obj.shape[1:3]) == self.spatial_shape
        ), f"Expected spatial shape {self.spatial_shape}, got {tuple(obj.shape[1:3])}."
        assert (
            obj.shape[3] == self.n_channels
        ), f"Expected {self.n_channels} channels, got {obj.shape[3]}."

        path = obj[..., 0].long() == self.path_token
        path_float = path.float()
        source_mask = self.source_mask.to(device=path.device)
        terminal_masks = self.terminal_masks.to(device=path.device)
        terminal_distances = self.terminal_distances.to(device=path.device)

        connected = self._connected_from_source(path_float, source_mask)

        source_score = (path_float * source_mask).amax(dim=(1, 2))
        path_count = path_float.sum(dim=(1, 2))
        connected_count = connected.sum(dim=(1, 2))
        connected_foreground_score = torch.where(
            path_count > 0.0,
            connected_count / path_count.clamp(min=1.0),
            torch.zeros_like(path_count),
        )

        terminal_hits = (
            connected[:, None] * terminal_masks[None]
        ).amax(dim=(2, 3))
        terminal_hit_score = terminal_hits.mean(dim=1)

        connected_for_distance = connected[:, None].bool()
        distance_fill = torch.full_like(
            terminal_distances[None],
            float("inf"),
        )
        connected_distances = torch.where(
            connected_for_distance,
            terminal_distances[None],
            distance_fill,
        )
        min_terminal_distances = connected_distances.amin(dim=(2, 3))
        approach_radius = (
            max(self.spatial_shape) * self.approach_radius_fraction
        )
        terminal_approach_score = (
            1.0 - min_terminal_distances / approach_radius
        ).clamp(min=0.0, max=1.0)
        terminal_approach_score = terminal_approach_score.mean(dim=1)

        connected_tiles = F.adaptive_max_pool2d(
            connected[:, None],
            output_size=(self.coverage_grid_size, self.coverage_grid_size),
        )
        coverage_score = connected_tiles.mean(dim=(1, 2, 3))

        density_score = self._density_score(path_float)
        structure_score = (
            self.terminal_hit_weight * terminal_hit_score
            + self.terminal_approach_weight * terminal_approach_score
            + self.coverage_weight * coverage_score
            + self.connected_foreground_weight * connected_foreground_score
            + self.source_weight * source_score
        )
        return density_score * structure_score

    def _connected_from_source(
        self,
        path_float: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, height, width = path_float.shape
        active = path_float[:, None]
        connected = active * source_mask[None, None]

        n_steps = self.connectivity_steps
        if n_steps is None:
            n_steps = height + width

        for _ in range(n_steps):
            connected = F.max_pool2d(
                connected,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            connected = connected * active

        return connected[:, 0]

    def _density_score(self, path_float: torch.Tensor) -> torch.Tensor:
        density = path_float.mean(dim=(1, 2))
        too_sparse_score = (density / self.path_density_target).clamp(max=1.0)
        too_dense_score = (
            1.0
            - (density - self.path_density_target).clamp(min=0.0)
            / (self.max_path_density - self.path_density_target)
        ).clamp(min=0.0, max=1.0)
        return torch.sqrt(too_sparse_score * too_dense_score)

    @staticmethod
    def _build_gate_buffers(
        spatial_shape: tuple[int, int],
        source_gate_size: int,
        terminal_gate_size: int,
        terminal_gates_per_side: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = spatial_shape
        source_radius = max(0, source_gate_size // 2)
        terminal_radius = max(0, terminal_gate_size // 2)
        edge_depth = max(1, min(height, width) // 32)

        source_mask = torch.zeros(height, width, dtype=torch.float32)
        source_center = height // 2
        y0 = max(0, source_center - source_radius)
        y1 = min(height, source_center + source_radius + 1)
        source_mask[y0:y1, :edge_depth] = 1.0

        terminal_bounds: list[tuple[int, int, int, int]] = []

        def add_terminal(y_start: int, y_end: int, x_start: int, x_end: int) -> None:
            terminal_bounds.append(
                (
                    max(0, y_start),
                    min(height, y_end),
                    max(0, x_start),
                    min(width, x_end),
                )
            )

        top_xs = torch.linspace(
            0.18 * (width - 1),
            0.86 * (width - 1),
            terminal_gates_per_side,
        ).round().long()
        right_ys = torch.linspace(
            0.10 * (height - 1),
            0.90 * (height - 1),
            terminal_gates_per_side,
        ).round().long()
        bottom_xs = torch.linspace(
            0.28 * (width - 1),
            0.96 * (width - 1),
            terminal_gates_per_side,
        ).round().long()

        for x_center in top_xs.tolist():
            add_terminal(
                0,
                edge_depth,
                x_center - terminal_radius,
                x_center + terminal_radius + 1,
            )
        for y_center in right_ys.tolist():
            add_terminal(
                y_center - terminal_radius,
                y_center + terminal_radius + 1,
                width - edge_depth,
                width,
            )
        for x_center in bottom_xs.tolist():
            add_terminal(
                height - edge_depth,
                height,
                x_center - terminal_radius,
                x_center + terminal_radius + 1,
            )

        terminal_masks = []
        terminal_distances = []
        y_coords = torch.arange(height).view(height, 1)
        x_coords = torch.arange(width).view(1, width)

        for y0, y1, x0, x1 in terminal_bounds:
            mask = torch.zeros(height, width, dtype=torch.float32)
            mask[y0:y1, x0:x1] = 1.0
            terminal_masks.append(mask)

            dy = (y0 - y_coords).clamp(min=0) + (y_coords - (y1 - 1)).clamp(min=0)
            dx = (x0 - x_coords).clamp(min=0) + (x_coords - (x1 - 1)).clamp(min=0)
            terminal_distances.append((dy + dx).float())

        return (
            source_mask,
            torch.stack(terminal_masks, dim=0),
            torch.stack(terminal_distances, dim=0),
        )
