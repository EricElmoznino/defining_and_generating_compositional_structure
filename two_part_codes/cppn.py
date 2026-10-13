import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cppn_coordinates import (
    CPPNCoordinateData,
    CPPNCoordinateFrame,
    coordinate_grid_from_frame,
)
from lib.helpers import (
    lookup_coordinate_batch,
    positional_encode,
    select_active_coordinate_indices,
)
from lib.independent_model_fitting import (
    BatchedCoordinateMLP,
    adamw_step_batched_,
    build_activation,
    clip_batched_grad_norm_,
    gather_batched_targets,
    init_batched_adamw_state,
    iter_sample_coordinate_chunks,
    parse_cppn_activation_names,
)
from lib.type_aliases import CPPNModel
from two_part_codes.base import TwoPartCode


def _coordinate_grid(
    spatial_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axes = []
    for size in spatial_shape:
        if size <= 1:
            axes.append(torch.zeros(size, device=device, dtype=dtype))
        else:
            axes.append(torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype))
    mesh = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack(mesh, dim=-1)
    return coords.reshape(-1, len(spatial_shape))


def _cross_entropy_sum_bits_per_sample(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    num_samples = logits.shape[0]
    loss_nats = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    )
    loss_nats = loss_nats.view(num_samples, -1).sum(dim=1)
    return loss_nats / math.log(2.0)


def _empty_fit_logs() -> dict[str, list[float]]:
    return {
        "step": [],
        "train_bits_per_symbol": [],
        "val_bits_per_symbol": [],
        "lr": [],
        "samples_seen": [],
    }


def _append_fit_log(
    logs: dict[str, list[float]],
    step: float,
    train_bits: float,
    val_bits: float,
    lr: float,
    samples_seen: float,
) -> None:
    logs["step"].append(float(step))
    logs["train_bits_per_symbol"].append(float(train_bits))
    logs["val_bits_per_symbol"].append(float(val_bits))
    logs["lr"].append(float(lr))
    logs["samples_seen"].append(float(samples_seen))


class _CoordinateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        vocab_size: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.activation_names = parse_cppn_activation_names(activation)
        self.layer_dim = hidden_dim * len(self.activation_names)
        self.input_layer = nn.Linear(input_dim, self.layer_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.layer_dim, self.layer_dim)
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(self.layer_dim, num_channels * vocab_size)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(coords)
        x = self.dropout(self.activation(x))
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
        logits = self.output_layer(x)
        return logits.view(coords.shape[0], self.num_channels, self.vocab_size)


class _ConstantMarginalModel(nn.Module):
    def __init__(self, channel_logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("channel_logits", channel_logits)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.channel_logits.unsqueeze(0).expand(coords.shape[0], -1, -1)


@dataclass
class CPPNFitResult:
    model: nn.Module
    model_complexity: float
    data_complexity: float
    logs: dict[str, list[float]]
    summary: dict[str, Any]


class CPPNMLPTwoPartCode(TwoPartCode[CPPNModel]):
    """Two-part code that fits one coordinate-MLP (CPPN) per object."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        use_positional_encoding: bool = False,
        positional_encoding_bands: int = 4,
        include_raw_coordinates: bool = True,
        use_shared_coordinates: bool = False,
        validation_fraction: float = 0.15,
        validation_mode: Literal["random", "block"] = "random",
        min_validation_points: int = 16,
        max_steps: int = 300,
        min_steps: int = 10,
        eval_every: int = 5,
        patience_evals: int = 8,
        min_delta_bits: float = 1e-4,
        lr: float = 3e-3,
        min_lr_ratio: float = 0.1,
        lr_scheduler: Literal["none", "cosine"] = "cosine",
        weight_decay: float = 1e-4,
        coordinate_batch_size: int | None = None,
        collective_coordinate_batch_size: int | None = None,
        gradient_clip_norm: float | None = 1.0,
        marginal_pseudocount: float = 1e-3,
        min_relative_improvement_over_constant: float = 0.02,
        min_absolute_improvement_bits: float = 0.02,
        model_complexity_mode: Literal[
            "excess_train_bits_auc", "samples_seen"
        ] = "excess_train_bits_auc",
        complexity_samples_scale: float = 1.0,
        constant_model_complexity: float = 0.0,
        device: torch.device | str | None = None,
        training_dtype: torch.dtype = torch.float32,
        seed: int | None = None,
        store_models_on_cpu: bool = True,
    ) -> None:
        """Initialize a CPPN two-part compressor for discrete coordinate-indexed tensors.

        Training loop pseudocode (per object):
            1. Flatten object into coordinate/target pairs.
            2. Build normalized coordinates, optionally apply frequency encoding.
            3. Split coordinates into train/validation sets (`random` or contiguous `block`).
            4. Build constant baseline model from train-set channel marginals.
            5. Train MLP with cross-entropy on train coordinates:
               a. Take one full or minibatch coordinate batch.
               b. Forward -> CE loss -> backward -> optimizer step.
               c. Optional gradient clipping and LR scheduler step.
            6. Every `eval_every` steps, compute full train/val bits-per-symbol.
            7. Track best checkpoint by `val_bits`.
               Stop when no improvement for `patience_evals` evaluations
               (after `min_steps`) or at `max_steps`.
            8. Restore best checkpoint.
            9. Compare best trained model vs constant baseline on validation:
               keep trained model only if both absolute and relative improvements
               exceed thresholds; otherwise use constant model.
           10. Compute model complexity proxy:
               - `samples_seen`: selected_samples_seen * scale
               - `excess_train_bits_auc`: area over samples-seen of
                 max(0, train_bits - selected_train_bits), scaled.
           11. Return selected model, model complexity proxy, and total-data NLL bits.

        Args:
            vocab_size (int): Number of discrete values each channel can take.
            hidden_dim (int, optional): Hidden width of the coordinate MLP.
            num_hidden_layers (int, optional): Number of hidden layers in the MLP.
            activation (str, optional): Space-separated hidden activation functions.
                Supported names are `identity`, `sigmoid`, `tanh`, `sin`, `gauss`,
                and `silu`. Each hidden layer has
                `num_activation_functions * hidden_dim` units.
            dropout (float, optional): Dropout probability on hidden activations.
            use_positional_encoding (bool, optional): Whether to apply NeRF-style
                sinusoidal coordinate encoding.
            positional_encoding_bands (int, optional): Number of frequency bands
                used when positional encoding is enabled.
            include_raw_coordinates (bool, optional): If True, concatenate raw
                normalized coordinates with sinusoidal features.
            use_shared_coordinates (bool, optional): If True, preserve one
                full-image normalized coordinate system across hierarchical
                crops. Defaults to False.
            validation_fraction (float, optional): Fraction of coordinates held out
                for validation and early stopping.
            validation_mode (Literal["random", "block"], optional): Validation split
                strategy (`random` index sampling or contiguous `block` holdout).
            min_validation_points (int, optional): Minimum number of held-out points.
            max_steps (int, optional): Maximum gradient steps.
            min_steps (int, optional): Minimum steps before patience-based stopping.
            eval_every (int, optional): Evaluate train/val metrics every N steps.
            patience_evals (int, optional): Number of non-improving evaluations
                before early stopping.
            min_delta_bits (float, optional): Minimum score improvement required to
                reset patience.
            lr (float, optional): AdamW learning rate.
            min_lr_ratio (float, optional): Minimum LR ratio for cosine annealing.
            lr_scheduler (Literal["none", "cosine"], optional): LR scheduling mode.
            weight_decay (float, optional): AdamW weight decay.
            coordinate_batch_size (int | None, optional): Minibatch size in number
                of coordinates; `None` uses full-batch updates.
            collective_coordinate_batch_size (int | None, optional): Maximum number
                of sample-coordinate pairs evaluated in one vectorized forward chunk.
                This bounds GPU memory without changing each CPPN's coordinate
                minibatch size. `None` processes all active samples at once.
            gradient_clip_norm (float | None, optional): Global gradient-norm clip;
                set to `None` to disable clipping.
            marginal_pseudocount (float, optional): Additive smoothing when fitting
                constant marginal baseline channel distributions.
            min_relative_improvement_over_constant (float, optional): Minimum
                relative validation improvement required to keep trained CPPN.
            min_absolute_improvement_bits (float, optional): Minimum absolute
                validation improvement (bits/symbol) required to keep trained CPPN.
            model_complexity_mode (Literal["excess_train_bits_auc", "samples_seen"], optional):
                Model complexity proxy mode. `excess_train_bits_auc` uses a
                prequential-like cumulative excess training bits proxy.
                `samples_seen` uses selected training samples seen.
            complexity_samples_scale (float, optional): Scale factor applied to
                the selected model complexity proxy from `model_complexity_mode`.
            constant_model_complexity (float, optional): Complexity assigned to the
                constant marginal fallback model.
            device (torch.device | str | None, optional): Training device. If None,
                uses CUDA when available, else CPU.
            training_dtype (torch.dtype, optional): Floating dtype for coordinates
                and model parameters.
            seed (int | None, optional): Base seed for per-sample RNG.
            store_models_on_cpu (bool, optional): If True, move returned models to CPU.
        """
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if validation_fraction < 0.0 or validation_fraction >= 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if min_steps < 0:
            raise ValueError("min_steps must be >= 0")
        if eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if patience_evals < 1:
            raise ValueError("patience_evals must be >= 1")
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if min_lr_ratio <= 0.0 or min_lr_ratio > 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        if lr_scheduler not in {"none", "cosine"}:
            raise ValueError("lr_scheduler must be one of: none, cosine")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        activation_names = parse_cppn_activation_names(activation)
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if coordinate_batch_size is not None and coordinate_batch_size < 1:
            raise ValueError("coordinate_batch_size must be >= 1")
        if (
            collective_coordinate_batch_size is not None
            and collective_coordinate_batch_size < 1
        ):
            raise ValueError("collective_coordinate_batch_size must be >= 1")
        if marginal_pseudocount < 0.0:
            raise ValueError("marginal_pseudocount must be >= 0")
        if min_absolute_improvement_bits < 0.0:
            raise ValueError("min_absolute_improvement_bits must be >= 0")
        if model_complexity_mode not in {"excess_train_bits_auc", "samples_seen"}:
            raise ValueError(
                "model_complexity_mode must be one of: "
                "'excess_train_bits_auc', 'samples_seen'"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.activation_names = activation_names
        self.dropout = dropout
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_bands = positional_encoding_bands
        self.include_raw_coordinates = include_raw_coordinates
        self.use_shared_coordinates = use_shared_coordinates
        self.validation_fraction = validation_fraction
        self.validation_mode = validation_mode
        self.min_validation_points = min_validation_points
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.eval_every = eval_every
        self.patience_evals = patience_evals
        self.min_delta_bits = min_delta_bits
        self.lr = lr
        self.min_lr_ratio = min_lr_ratio
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.coordinate_batch_size = coordinate_batch_size
        self.collective_coordinate_batch_size = collective_coordinate_batch_size
        self.gradient_clip_norm = gradient_clip_norm
        self.marginal_pseudocount = marginal_pseudocount
        self.min_relative_improvement_over_constant = (
            min_relative_improvement_over_constant
        )
        self.min_absolute_improvement_bits = min_absolute_improvement_bits
        self.model_complexity_mode = model_complexity_mode
        self.complexity_samples_scale = complexity_samples_scale
        self.constant_model_complexity = constant_model_complexity
        self.device = device
        self.training_dtype = training_dtype
        self.seed = seed
        self.store_models_on_cpu = store_models_on_cpu

    def _sample_rng(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    def prepare_for_decomposition(
        self, data: np.ndarray | torch.Tensor | CPPNCoordinateData
    ) -> np.ndarray | torch.Tensor | CPPNCoordinateData:
        if not self.use_shared_coordinates:
            return data
        if isinstance(data, CPPNCoordinateData):
            return data
        return CPPNCoordinateData.root(data)

    def _unwrap_data(
        self, data: np.ndarray | torch.Tensor | CPPNCoordinateData
    ) -> tuple[np.ndarray | torch.Tensor, CPPNCoordinateFrame | None]:
        if isinstance(data, CPPNCoordinateData):
            if not self.use_shared_coordinates:
                raise ValueError(
                    "CPPN coordinate metadata requires use_shared_coordinates=True."
                )
            return data.data, data.coordinate_frame

        if not self.use_shared_coordinates:
            return data, None
        return data, CPPNCoordinateData.root(data).coordinate_frame

    def _build_coordinate_grid(
        self,
        spatial_shape: tuple[int, ...],
        coordinate_frame: CPPNCoordinateFrame | None,
    ) -> torch.Tensor:
        if coordinate_frame is None:
            return _coordinate_grid(
                spatial_shape, device=self.device, dtype=self.training_dtype
            )
        if coordinate_frame.spatial_shape != spatial_shape:
            raise ValueError(
                "CPPN coordinate frame does not match the modeled spatial shape."
            )
        return coordinate_grid_from_frame(
            coordinate_frame, device=self.device, dtype=self.training_dtype
        )

    def _encode_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.use_positional_encoding:
            return coords
        return positional_encode(
            coords,
            num_bands=self.positional_encoding_bands,
            include_raw=self.include_raw_coordinates,
        )

    def _train_val_split(
        self,
        spatial_shape: tuple[int, ...],
        num_points: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_points <= 1 or self.validation_fraction <= 0.0:
            idx = torch.arange(num_points, dtype=torch.long)
            return idx, idx

        val_count = int(round(num_points * self.validation_fraction))
        val_count = max(self.min_validation_points, val_count)
        val_count = min(max(1, val_count), num_points - 1)

        if self.validation_mode == "random":
            perm = rng.permutation(num_points)
            val_idx_np = perm[:val_count]
            train_idx_np = perm[val_count:]
            return (
                torch.from_numpy(train_idx_np.astype(np.int64)),
                torch.from_numpy(val_idx_np.astype(np.int64)),
            )

        if self.validation_mode != "block":
            raise ValueError("validation_mode must be either 'random' or 'block'")

        # Hold out one contiguous block to discourage coordinate memorization.
        num_dims = len(spatial_shape)
        side_scale = self.validation_fraction ** (1.0 / max(1, num_dims))
        block_sizes = [
            max(1, min(size, int(round(size * side_scale)))) for size in spatial_shape
        ]
        starts = []
        for size, block_size in zip(spatial_shape, block_sizes):
            max_start = size - block_size
            if max_start <= 0:
                starts.append(0)
            else:
                starts.append(int(rng.integers(0, max_start + 1)))

        mask = torch.zeros(spatial_shape, dtype=torch.bool)
        slices = tuple(
            slice(start, start + block_size)
            for start, block_size in zip(starts, block_sizes)
        )
        mask[slices] = True
        val_idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)

        if val_idx.numel() == 0:
            perm = rng.permutation(num_points)
            val_idx = torch.from_numpy(perm[:val_count].astype(np.int64))
        elif val_idx.numel() > val_count:
            chosen = rng.choice(val_idx.cpu().numpy(), size=val_count, replace=False)
            val_idx = torch.from_numpy(chosen.astype(np.int64))
        elif val_idx.numel() < val_count:
            val_set = set(val_idx.cpu().numpy().tolist())
            remaining = np.array(
                [i for i in range(num_points) if i not in val_set], dtype=np.int64
            )
            needed = val_count - val_idx.numel()
            if remaining.size > 0:
                extra = rng.choice(
                    remaining, size=min(needed, remaining.size), replace=False
                )
                val_idx = torch.cat(
                    [val_idx, torch.from_numpy(extra.astype(np.int64))], dim=0
                )

        all_idx = torch.arange(num_points, dtype=torch.long)
        train_mask = torch.ones(num_points, dtype=torch.bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]

        if train_idx.numel() == 0:
            train_idx = val_idx[:1]
            val_idx = val_idx[1:]
        if val_idx.numel() == 0:
            val_idx = train_idx

        return train_idx, val_idx

    def _complexity_from_logs_excess_train_bits_auc(
        self,
        logs: dict[str, list[float]],
        selected_samples_seen: int,
        selected_train_bits: float,
        num_channels: int,
    ) -> float:
        # Integrate max(0, train_bits - selected_train_bits) over samples seen.
        # `samples_seen` is counted in coordinates, so multiply by channels to
        # approximate bits over symbols.
        if selected_samples_seen <= 0:
            return 0.0

        samples = logs["samples_seen"]
        train_bits = logs["train_bits_per_symbol"]
        if len(samples) == 0:
            return 0.0

        area = 0.0
        prev_x = 0.0
        prev_y = max(0.0, float(train_bits[0]) - selected_train_bits)

        for x, y_raw in zip(samples, train_bits):
            x = float(x)
            if x > selected_samples_seen:
                y = prev_y
                dx = max(0.0, float(selected_samples_seen) - prev_x)
                area += 0.5 * (prev_y + y) * dx
                break

            y = max(0.0, float(y_raw) - selected_train_bits)
            dx = max(0.0, x - prev_x)
            area += 0.5 * (prev_y + y) * dx
            prev_x = x
            prev_y = y

        return area * float(max(1, num_channels))

    def _estimate_channel_logits_batched(self, targets: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(targets, num_classes=self.vocab_size).to(torch.float32)
        counts = one_hot.sum(dim=1)
        counts = counts + self.marginal_pseudocount
        probs = counts / counts.sum(dim=-1, keepdim=True)
        return probs.log()

    @staticmethod
    def _constant_sum_bits_per_sample(
        channel_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = channel_logits.unsqueeze(1).expand(-1, targets.shape[1], -1, -1)
        log_probs = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
        return -log_probs.squeeze(-1).sum(dim=(1, 2)) / math.log(2.0)

    @staticmethod
    def _constant_mean_bits_per_sample(
        channel_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        total_bits = CPPNMLPTwoPartCode._constant_sum_bits_per_sample(
            channel_logits, targets
        )
        return total_bits / float(targets.shape[1] * targets.shape[2])

    def _eval_batched_model_bits_per_sample(
        self,
        model: BatchedCoordinateMLP,
        sample_indices: torch.Tensor,
        coord_indices: torch.Tensor,
        encoded_coords: torch.Tensor,
        targets: torch.Tensor,
        *,
        mean: bool,
    ) -> torch.Tensor:
        model.eval()
        num_samples = int(sample_indices.numel())
        if num_samples == 0:
            return torch.empty(0, device=targets.device, dtype=torch.float32)

        num_coords = int(coord_indices.shape[1])
        total_bits = torch.zeros(
            num_samples, device=targets.device, dtype=torch.float32
        )
        selected_coord_indices = select_active_coordinate_indices(
            coord_indices, sample_indices
        )

        with torch.no_grad():
            for (
                sample_start,
                sample_end,
                coord_start,
                coord_end,
            ) in iter_sample_coordinate_chunks(
                num_samples, num_coords, self.collective_coordinate_batch_size
            ):
                chunk_samples = sample_indices[sample_start:sample_end]
                chunk_coord_indices = selected_coord_indices[
                    sample_start:sample_end, coord_start:coord_end
                ]
                chunk_coords = lookup_coordinate_batch(
                    encoded_coords, chunk_coord_indices
                )
                chunk_targets = gather_batched_targets(
                    targets, chunk_samples, chunk_coord_indices
                )
                logits = model(chunk_samples, chunk_coords)
                total_bits[sample_start:sample_end] += (
                    _cross_entropy_sum_bits_per_sample(logits, chunk_targets)
                    .detach()
                    .to(torch.float32)
                )

        if not mean:
            return total_bits
        return total_bits / float(num_coords * targets.shape[-1])

    def _backward_batched_training_loss(
        self,
        model: BatchedCoordinateMLP,
        sample_indices: torch.Tensor,
        coord_indices: torch.Tensor,
        encoded_coords: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        num_samples = int(sample_indices.numel())
        num_coords = int(coord_indices.shape[1])
        num_channels = int(targets.shape[-1])
        denom = float(num_coords * num_channels)

        for (
            sample_start,
            sample_end,
            coord_start,
            coord_end,
        ) in iter_sample_coordinate_chunks(
            num_samples, num_coords, self.collective_coordinate_batch_size
        ):
            chunk_samples = sample_indices[sample_start:sample_end]
            chunk_coord_indices = coord_indices[
                sample_start:sample_end, coord_start:coord_end
            ]
            chunk_coords = lookup_coordinate_batch(encoded_coords, chunk_coord_indices)
            chunk_targets = gather_batched_targets(
                targets, chunk_samples, chunk_coord_indices
            )
            logits = model(chunk_samples, chunk_coords)
            loss_nats = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                chunk_targets.reshape(-1),
                reduction="none",
            )
            loss_nats = loss_nats.view(sample_end - sample_start, -1).sum(dim=1)
            loss = (loss_nats / denom).sum()
            loss.backward()

    def _lr_after_scheduler_step(self, steps: torch.Tensor) -> torch.Tensor:
        if self.lr_scheduler == "none":
            return torch.full_like(steps, float(self.lr), dtype=torch.float32)
        if self.lr_scheduler == "cosine":
            steps_float = steps.to(torch.float32)
            eta_min = self.lr * self.min_lr_ratio
            return eta_min + 0.5 * (self.lr - eta_min) * (
                1.0 + torch.cos(math.pi * steps_float / float(max(1, self.max_steps)))
            )
        raise ValueError("lr_scheduler must be one of: none, cosine")

    def _extract_single_model_from_batched_state(
        self,
        state: dict[str, torch.Tensor],
        sample_idx: int,
        input_dim: int,
        num_channels: int,
        target_device: torch.device,
    ) -> nn.Module:
        model = _CoordinateMLP(
            input_dim=input_dim,
            num_channels=num_channels,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
        ).to(device=target_device, dtype=self.training_dtype)

        state_dict: dict[str, torch.Tensor] = {
            "input_layer.weight": state["input_weight"][sample_idx]
            .detach()
            .to(target_device)
            .clone(),
            "input_layer.bias": state["input_bias"][sample_idx]
            .detach()
            .to(target_device)
            .clone(),
            "output_layer.weight": state["output_weight"][sample_idx]
            .detach()
            .to(target_device)
            .clone(),
            "output_layer.bias": state["output_bias"][sample_idx]
            .detach()
            .to(target_device)
            .clone(),
        }
        for layer_idx in range(self.num_hidden_layers - 1):
            state_dict[f"hidden_layers.{layer_idx}.weight"] = (
                state[f"hidden_weight_{layer_idx}"][sample_idx]
                .detach()
                .to(target_device)
                .clone()
            )
            state_dict[f"hidden_layers.{layer_idx}.bias"] = (
                state[f"hidden_bias_{layer_idx}"][sample_idx]
                .detach()
                .to(target_device)
                .clone()
            )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def fit_with_logs(
        self, data: np.ndarray | torch.Tensor | CPPNCoordinateData
    ) -> list[CPPNFitResult]:
        """Fit one independent CPPN per sample with a vectorized batched trainer."""
        raw_data, coordinate_frame = self._unwrap_data(data)
        if isinstance(raw_data, np.ndarray):
            tensor_data = torch.from_numpy(raw_data)
        else:
            tensor_data = raw_data

        if tensor_data.ndim < 3:
            raise ValueError(
                "Data must have shape (batch, *spatial_dims, channels). "
                f"Got shape={tuple(tensor_data.shape)}."
            )

        batch_size = int(tensor_data.shape[0])
        if batch_size == 0:
            return []

        spatial_shape = tuple(int(v) for v in tensor_data.shape[1:-1])
        num_channels = int(tensor_data.shape[-1])
        if num_channels < 1:
            raise ValueError("The channel dimension must be >= 1.")

        device = self.device
        tensor_data = tensor_data.to(device=device, non_blocking=False)
        if torch.is_floating_point(tensor_data):
            rounded = tensor_data.round()
            if not torch.allclose(tensor_data, rounded):
                raise ValueError(
                    "CPPN compression expects discrete integer-valued tensors."
                )
            tensor_data = rounded

        targets = tensor_data.to(torch.long).reshape(batch_size, -1, num_channels)
        if targets.shape[1] < 1:
            raise ValueError("The spatial dimensions must contain at least one point.")
        if targets.min().item() < 0 or targets.max().item() >= self.vocab_size:
            raise ValueError(
                f"Data values must be in [0, {self.vocab_size - 1}] for vocab_size={self.vocab_size}."
            )

        coords = self._build_coordinate_grid(spatial_shape, coordinate_frame)
        encoded_coords = self._encode_coordinates(coords)
        num_points = int(encoded_coords.shape[0])
        input_dim = int(encoded_coords.shape[-1])

        coordinate_rng = self._sample_rng(0)
        train_idx_cpu, val_idx_cpu = self._train_val_split(
            spatial_shape, num_points, coordinate_rng
        )
        train_idx_np = train_idx_cpu.numpy()
        train_idx_row = train_idx_cpu.to(device=device)
        val_idx_row = val_idx_cpu.to(device=device)
        train_idx = train_idx_row.unsqueeze(0).expand(batch_size, -1)
        val_idx = val_idx_row.unsqueeze(0).expand(batch_size, -1)
        num_train = int(train_idx_row.numel())
        num_val = int(val_idx_row.numel())
        all_sample_indices = torch.arange(batch_size, device=device, dtype=torch.long)

        train_targets = gather_batched_targets(targets, all_sample_indices, train_idx)
        val_targets = gather_batched_targets(targets, all_sample_indices, val_idx)
        baseline_logits = self._estimate_channel_logits_batched(train_targets).to(
            device=device, dtype=self.training_dtype
        )
        baseline_val_bits = self._constant_mean_bits_per_sample(
            baseline_logits, val_targets
        ).to(torch.float32)
        del train_targets, val_targets

        logs = [_empty_fit_logs() for _ in range(batch_size)]

        model = BatchedCoordinateMLP(
            batch_size=batch_size,
            input_dim=input_dim,
            num_channels=num_channels,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
        ).to(device=device, dtype=self.training_dtype)
        named_parameters = model.batched_named_parameters()
        adamw_state = init_batched_adamw_state(named_parameters)
        best_state = {
            name: torch.empty_like(parameter.detach())
            for name, parameter in named_parameters
        }

        best_score = torch.full((batch_size,), float("inf"), device=device)
        best_val_bits = torch.full((batch_size,), float("inf"), device=device)
        best_train_bits = torch.full((batch_size,), float("inf"), device=device)
        best_step = torch.zeros(batch_size, device=device, dtype=torch.long)
        best_samples_seen = torch.zeros(batch_size, device=device, dtype=torch.long)
        num_evals_without_improve = torch.zeros(
            batch_size, device=device, dtype=torch.long
        )
        samples_seen = torch.zeros(batch_size, device=device, dtype=torch.long)
        steps_ran = torch.zeros(batch_size, device=device, dtype=torch.long)
        optimizer_steps = torch.zeros(batch_size, device=device, dtype=torch.long)
        current_lr = torch.full(
            (batch_size,), float(self.lr), device=device, dtype=torch.float32
        )
        active_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

        if (
            self.coordinate_batch_size is None
            or self.coordinate_batch_size >= num_train
        ):
            per_sample_batch_size = num_train
        else:
            per_sample_batch_size = int(self.coordinate_batch_size)
        per_sample_batch_size = max(1, per_sample_batch_size)

        for step in range(1, self.max_steps + 1):
            active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break

            model.train()
            if per_sample_batch_size >= num_train:
                batch_coord_idx = train_idx[:1].expand(int(active_indices.numel()), -1)
            else:
                chosen_np = coordinate_rng.choice(
                    train_idx_np,
                    size=per_sample_batch_size,
                    replace=num_train < per_sample_batch_size,
                )
                batch_coord_row = torch.from_numpy(chosen_np.astype(np.int64)).to(
                    device=device
                )
                batch_coord_idx = batch_coord_row.unsqueeze(0).expand(
                    int(active_indices.numel()), -1
                )

            model.zero_grad(set_to_none=True)
            self._backward_batched_training_loss(
                model=model,
                sample_indices=active_indices,
                coord_indices=batch_coord_idx,
                encoded_coords=encoded_coords,
                targets=targets,
            )
            if self.gradient_clip_norm is not None:
                clip_batched_grad_norm_(
                    named_parameters,
                    active_indices,
                    float(self.gradient_clip_norm),
                )

            optimizer_steps[active_indices] += 1
            adamw_step_batched_(
                named_parameters=named_parameters,
                state=adamw_state,
                active_indices=active_indices,
                lr_values=current_lr.index_select(0, active_indices),
                step_counts=optimizer_steps.index_select(0, active_indices),
                weight_decay=self.weight_decay,
            )
            current_lr[active_indices] = self._lr_after_scheduler_step(
                optimizer_steps.index_select(0, active_indices)
            ).to(device=device)
            model.zero_grad(set_to_none=True)

            samples_seen[active_indices] += int(batch_coord_idx.shape[1])
            steps_ran[active_indices] = int(step)

            if step % self.eval_every != 0 and step != 1 and step != self.max_steps:
                continue

            train_bits = self._eval_batched_model_bits_per_sample(
                model=model,
                sample_indices=active_indices,
                coord_indices=train_idx,
                encoded_coords=encoded_coords,
                targets=targets,
                mean=True,
            )
            val_bits = self._eval_batched_model_bits_per_sample(
                model=model,
                sample_indices=active_indices,
                coord_indices=val_idx,
                encoded_coords=encoded_coords,
                targets=targets,
                mean=True,
            )
            lr_values = current_lr.index_select(0, active_indices)

            active_list = active_indices.detach().cpu().tolist()
            train_bits_list = train_bits.detach().cpu().tolist()
            val_bits_list = val_bits.detach().cpu().tolist()
            lr_values_list = lr_values.detach().cpu().tolist()
            samples_seen_list = (
                samples_seen.index_select(0, active_indices).detach().cpu().tolist()
            )
            for local_pos, sample_idx in enumerate(active_list):
                _append_fit_log(
                    logs[sample_idx],
                    step,
                    train_bits_list[local_pos],
                    val_bits_list[local_pos],
                    lr_values_list[local_pos],
                    samples_seen_list[local_pos],
                )

            previous_best = best_score.index_select(0, active_indices)
            first_eval = best_step.index_select(0, active_indices) == 0
            improved = first_eval | (val_bits < previous_best - self.min_delta_bits)
            if improved.any():
                improved_indices = active_indices[improved]
                best_score[improved_indices] = val_bits[improved]
                best_val_bits[improved_indices] = val_bits[improved]
                best_train_bits[improved_indices] = train_bits[improved]
                best_step[improved_indices] = int(step)
                best_samples_seen[improved_indices] = samples_seen[improved_indices]
                num_evals_without_improve[improved_indices] = 0
                for name, parameter in named_parameters:
                    best_state[name][
                        improved_indices
                    ] = parameter.detach().index_select(0, improved_indices)

            if (~improved).any():
                not_improved_indices = active_indices[~improved]
                num_evals_without_improve[not_improved_indices] += 1

            should_stop = (step >= self.min_steps) & (
                num_evals_without_improve.index_select(0, active_indices)
                >= self.patience_evals
            )
            if should_stop.any():
                stopped_indices = active_indices[should_stop]
                active_mask[stopped_indices] = False
                steps_ran[stopped_indices] = int(step)

        with torch.no_grad():
            for name, parameter in named_parameters:
                parameter.copy_(best_state[name])

        trained_val_bits = best_val_bits.to(torch.float32)
        abs_improvement = baseline_val_bits - trained_val_bits
        rel_improvement = torch.where(
            baseline_val_bits < self.min_absolute_improvement_bits,
            torch.zeros_like(abs_improvement),
            abs_improvement / torch.clamp(baseline_val_bits.abs(), min=1e-3),
        )
        use_constant = (
            rel_improvement < self.min_relative_improvement_over_constant
        ) | (abs_improvement < self.min_absolute_improvement_bits)

        raw_model_complexity = torch.full(
            (batch_size,),
            float(self.constant_model_complexity),
            device=device,
            dtype=torch.float32,
        )
        model_complexity = raw_model_complexity.clone()
        selected_train_bits = torch.where(
            use_constant, baseline_val_bits, best_train_bits.to(torch.float32)
        )
        selected_val_bits = torch.where(
            use_constant, baseline_val_bits, trained_val_bits
        )
        selected_samples_seen = torch.where(
            use_constant,
            torch.zeros_like(best_samples_seen),
            best_samples_seen,
        )
        selected_step = torch.where(
            use_constant, torch.zeros_like(best_step), best_step
        )

        use_constant_list = use_constant.detach().cpu().tolist()
        best_samples_seen_list = best_samples_seen.detach().cpu().tolist()
        best_train_bits_list = best_train_bits.detach().cpu().tolist()
        for sample_idx in range(batch_size):
            if bool(use_constant_list[sample_idx]):
                continue
            if self.model_complexity_mode == "samples_seen":
                raw_value = float(best_samples_seen_list[sample_idx])
            else:
                raw_value = self._complexity_from_logs_excess_train_bits_auc(
                    logs=logs[sample_idx],
                    selected_samples_seen=int(best_samples_seen_list[sample_idx]),
                    selected_train_bits=float(best_train_bits_list[sample_idx]),
                    num_channels=num_channels,
                )
            raw_model_complexity[sample_idx] = float(raw_value)
            model_complexity[sample_idx] = float(
                raw_value * self.complexity_samples_scale
            )

        data_complexity = torch.empty(batch_size, device=device, dtype=torch.float32)
        constant_indices = torch.nonzero(use_constant, as_tuple=False).squeeze(-1)
        if constant_indices.numel() > 0:
            data_complexity[constant_indices] = self._constant_sum_bits_per_sample(
                baseline_logits.index_select(0, constant_indices),
                targets.index_select(0, constant_indices),
            ).to(torch.float32)

        trained_indices = torch.nonzero(~use_constant, as_tuple=False).squeeze(-1)
        if trained_indices.numel() > 0:
            all_coord_idx = (
                torch.arange(num_points, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            data_complexity[trained_indices] = self._eval_batched_model_bits_per_sample(
                model=model,
                sample_indices=trained_indices,
                coord_indices=all_coord_idx,
                encoded_coords=encoded_coords,
                targets=targets,
                mean=False,
            )

        target_device = torch.device("cpu") if self.store_models_on_cpu else device
        best_step_list = best_step.detach().cpu().tolist()
        steps_ran_list = steps_ran.detach().cpu().tolist()
        selected_step_list = selected_step.detach().cpu().tolist()
        samples_seen_list = samples_seen.detach().cpu().tolist()
        selected_samples_seen_list = selected_samples_seen.detach().cpu().tolist()
        baseline_val_bits_list = baseline_val_bits.detach().cpu().tolist()
        trained_val_bits_list = trained_val_bits.detach().cpu().tolist()
        selected_train_bits_list = selected_train_bits.detach().cpu().tolist()
        selected_val_bits_list = selected_val_bits.detach().cpu().tolist()
        rel_improvement_list = rel_improvement.detach().cpu().tolist()
        abs_improvement_list = abs_improvement.detach().cpu().tolist()
        raw_model_complexity_list = raw_model_complexity.detach().cpu().tolist()
        model_complexity_list = model_complexity.detach().cpu().tolist()
        data_complexity_list = data_complexity.detach().cpu().tolist()
        results: list[CPPNFitResult] = []
        for sample_idx in range(batch_size):
            if bool(use_constant_list[sample_idx]):
                final_model: nn.Module = _ConstantMarginalModel(
                    baseline_logits[sample_idx].detach().to(target_device).clone()
                ).to(target_device)
            else:
                final_model = self._extract_single_model_from_batched_state(
                    state=best_state,
                    sample_idx=sample_idx,
                    input_dim=input_dim,
                    num_channels=num_channels,
                    target_device=target_device,
                )
            final_model.eval()

            summary = {
                "best_step": int(best_step_list[sample_idx]),
                "steps_ran": int(steps_ran_list[sample_idx]),
                "selected_step": int(selected_step_list[sample_idx]),
                "samples_seen": int(samples_seen_list[sample_idx]),
                "selected_samples_seen": int(selected_samples_seen_list[sample_idx]),
                "num_train_points": int(num_train),
                "num_val_points": int(num_val),
                "used_constant_model": bool(use_constant_list[sample_idx]),
                "baseline_val_bits_per_symbol": float(
                    baseline_val_bits_list[sample_idx]
                ),
                "trained_val_bits_per_symbol": float(trained_val_bits_list[sample_idx]),
                "selected_train_bits_per_symbol": float(
                    selected_train_bits_list[sample_idx]
                ),
                "selected_val_bits_per_symbol": float(
                    selected_val_bits_list[sample_idx]
                ),
                "relative_improvement_over_constant": float(
                    rel_improvement_list[sample_idx]
                ),
                "absolute_improvement_over_constant": float(
                    abs_improvement_list[sample_idx]
                ),
                "model_complexity_mode": self.model_complexity_mode,
                "raw_model_complexity": float(raw_model_complexity_list[sample_idx]),
                "model_complexity": float(model_complexity_list[sample_idx]),
                "data_complexity": float(data_complexity_list[sample_idx]),
            }

            results.append(
                CPPNFitResult(
                    model=final_model,
                    model_complexity=float(model_complexity_list[sample_idx]),
                    data_complexity=float(data_complexity_list[sample_idx]),
                    logs=logs[sample_idx],
                    summary=summary,
                )
            )
        return results

    def compress(
        self, data: np.ndarray | torch.Tensor | CPPNCoordinateData
    ) -> tuple[CPPNModel, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Compress batched discrete tensors with independently fit CPPNs."""
        raw_data, coordinate_frame = self._unwrap_data(data)
        return_numpy = isinstance(raw_data, np.ndarray)
        out_device = raw_data.device if isinstance(raw_data, torch.Tensor) else None

        fit_results = self.fit_with_logs(data)
        models = [r.model for r in fit_results]
        model_complexities = [r.model_complexity for r in fit_results]
        data_complexities = [r.data_complexity for r in fit_results]

        if return_numpy:
            model_bits = np.asarray(model_complexities, dtype=np.float32)
            data_bits = np.asarray(data_complexities, dtype=np.float32)
        else:
            model_bits = torch.tensor(
                model_complexities,
                dtype=torch.float32,
                device=out_device,
            )
            data_bits = torch.tensor(
                data_complexities,
                dtype=torch.float32,
                device=out_device,
            )
        shape = tuple(int(v) for v in raw_data.shape[1:])
        if coordinate_frame is None:
            model: CPPNModel = (models, shape)
        else:
            model = (models, shape, coordinate_frame)
        return model, model_bits, data_bits


def _build_teacher_cppn(
    input_dim: int,
    vocab_size: int,
    mode: Literal["simple", "compositional"],
) -> _CoordinateMLP:
    teacher = _CoordinateMLP(
        input_dim=input_dim,
        num_channels=1,
        vocab_size=vocab_size,
        hidden_dim=24 if mode == "compositional" else 16,
        num_hidden_layers=3 if mode == "compositional" else 2,
        activation="tanh",
        dropout=0.0,
    )
    with torch.no_grad():
        for p in teacher.parameters():
            p.zero_()

        teacher.input_layer.weight[0, 0] = 3.0
        if input_dim > 1:
            teacher.input_layer.weight[1, 1] = 3.0
        else:
            teacher.input_layer.weight[1, 0] = -2.5

        if mode == "simple":
            teacher.input_layer.weight[2, 0] = 2.0
            teacher.input_layer.weight[3, 1 if input_dim > 1 else 0] = -2.0
            h0 = teacher.hidden_layers[0]
            h0.weight[0, 0] = 1.8
            h0.weight[1, 1] = 1.8
            h0.weight[2, 2] = 1.5
            h0.weight[3, 3] = 1.5
            h0.weight[4, 0] = 1.2
            h0.weight[4, 1] = -1.2
            out = teacher.output_layer
            out.weight[0, 0] = 2.2
            out.weight[0, 4] = 1.0
            out.weight[1, 1] = 2.2
            out.weight[1, 4] = -1.0
            out.weight[2, 2] = 2.0
            out.weight[2, 3] = 1.0
            out.weight[3, 0] = -1.0
            out.weight[3, 1] = -1.0
            out.weight[3, 2] = -1.0
            out.weight[3, 3] = -1.0
        else:
            teacher.input_layer.weight[2, min(2, input_dim - 1)] = 2.5
            teacher.input_layer.weight[3, min(3, input_dim - 1)] = -2.5
            teacher.input_layer.weight[4, min(4, input_dim - 1)] = 2.0
            teacher.input_layer.weight[5, min(5, input_dim - 1)] = 2.0

            h0 = teacher.hidden_layers[0]
            h1 = teacher.hidden_layers[1]
            h0.weight[0, 0] = 1.6
            h0.weight[0, 2] = 1.2
            h0.weight[1, 1] = 1.6
            h0.weight[1, 3] = 1.2
            h0.weight[2, 4] = 1.5
            h0.weight[3, 5] = 1.5
            h0.weight[4, 2] = 1.0
            h0.weight[4, 3] = -1.0
            h0.weight[5, 0] = 0.8
            h0.weight[5, 1] = 0.8

            h1.weight[0, 0] = 1.4
            h1.weight[0, 2] = 1.0
            h1.weight[1, 1] = 1.4
            h1.weight[1, 3] = 1.0
            h1.weight[2, 4] = 1.3
            h1.weight[3, 5] = 1.3
            h1.weight[4, 0] = 1.1
            h1.weight[4, 1] = -1.1
            h1.weight[5, 2] = 1.0
            h1.weight[5, 3] = -1.0

            out = teacher.output_layer
            out.weight[0, 0] = 2.2
            out.weight[0, 4] = 0.8
            out.weight[1, 1] = 2.2
            out.weight[1, 4] = -0.8
            out.weight[2, 2] = 2.0
            out.weight[2, 5] = 0.9
            out.weight[3, 3] = 2.0
            out.weight[3, 5] = -0.9

    teacher.eval()
    return teacher


def _generate_teacher_data(
    spatial_shape: tuple[int, ...],
    vocab_size: int,
    mode: Literal["simple", "compositional"],
    use_positional_encoding: bool = True,
) -> torch.Tensor:
    coords = _coordinate_grid(
        spatial_shape, device=torch.device("cpu"), dtype=torch.float32
    )
    if mode == "simple" or not use_positional_encoding:
        encoded = coords
    else:
        encoded = positional_encode(coords, num_bands=4, include_raw=True)
    teacher = _build_teacher_cppn(encoded.shape[-1], vocab_size=vocab_size, mode=mode)
    with torch.no_grad():
        logits = teacher(encoded)[:, 0, :]
    labels = logits.argmax(dim=-1).reshape(*spatial_shape, 1)
    return labels.to(torch.long)


def _run_spot_checks() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    shared_data = CPPNCoordinateData.root(torch.zeros(1, 4, 6, 1, dtype=torch.long))
    cropped_data = shared_data[:, 1:3, 2:5, :]
    singleton_data = shared_data[:, 2:3, 4:5, :]
    shared_root_coords = coordinate_grid_from_frame(
        shared_data.coordinate_frame,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).reshape(4, 6, 2)
    cropped_coords = coordinate_grid_from_frame(
        cropped_data.coordinate_frame,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    singleton_coords = coordinate_grid_from_frame(
        singleton_data.coordinate_frame,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_cropped_coords = shared_root_coords[1:3, 2:5].reshape(-1, 2)
    assert torch.equal(shared_root_coords[0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(shared_root_coords[-1, -1], torch.tensor([1.0, 1.0]))
    assert torch.equal(cropped_coords, expected_cropped_coords)
    assert torch.equal(singleton_coords[0], shared_root_coords[2, 4])
    assert torch.equal(
        positional_encode(cropped_coords, num_bands=3, include_raw=True),
        positional_encode(expected_cropped_coords, num_bands=3, include_raw=True),
    )
    print("Shared CPPN coordinate-frame spot checks passed.")

    spatial_shape = (20, 20)
    vocab_size = 4

    constant = torch.zeros(*spatial_shape, 1, dtype=torch.long)
    random_sample = torch.randint(0, vocab_size, (*spatial_shape, 1), dtype=torch.long)
    simple_teacher = _generate_teacher_data(
        spatial_shape=spatial_shape, vocab_size=vocab_size, mode="simple"
    )
    compositional_teacher = _generate_teacher_data(
        spatial_shape=spatial_shape, vocab_size=vocab_size, mode="compositional"
    )

    data = torch.stack(
        [constant, random_sample, simple_teacher, compositional_teacher], dim=0
    )

    compressor = CPPNMLPTwoPartCode(
        vocab_size=vocab_size,
        hidden_dim=48,
        num_hidden_layers=3,
        activation="tanh",
        use_positional_encoding=True,
        positional_encoding_bands=4,
        include_raw_coordinates=True,
        validation_fraction=0.2,
        validation_mode="block",
        max_steps=220,
        min_steps=10,
        eval_every=5,
        patience_evals=8,
        lr=4e-3,
        weight_decay=1e-4,
        coordinate_batch_size=None,
        min_relative_improvement_over_constant=0.02,
        complexity_samples_scale=1.0,
        device="cpu",
        seed=13,
    )

    results = compressor.fit_with_logs(data)
    names = ["constant", "random", "teacher_simple", "teacher_compositional"]

    print("CPPN two-part compression spot checks")
    print("-" * 88)
    print(
        f"{'sample':<24} {'const?':<8} {'model_c':>10} {'data_c':>12} "
        f"{'rel_imp':>10} {'steps':>8}"
    )
    for name, result in zip(names, results):
        s = result.summary
        print(
            f"{name:<24} {str(s['used_constant_model']):<8} "
            f"{s['model_complexity']:>10.1f} {s['data_complexity']:>12.1f} "
            f"{s['relative_improvement_over_constant']:>10.3f} {s['selected_step']:>8d}"
        )

    constant_res, random_res, simple_res, comp_res = results

    assert constant_res.summary[
        "used_constant_model"
    ], "Constant data should select the coordinate-independent marginal model."
    assert random_res.summary[
        "used_constant_model"
    ], "Random data should usually not justify a trained CPPN over the constant baseline."
    assert (
        simple_res.summary["relative_improvement_over_constant"] > 0.05
    ), "Teacher-generated simple structure should improve validation compression."
    assert (
        comp_res.summary["relative_improvement_over_constant"] > 0.05
    ), "Teacher-generated compositional structure should improve validation compression."
    assert (
        simple_res.summary["model_complexity"] > random_res.summary["model_complexity"]
    ), "Structured data should require higher model complexity proxy than random data."
    assert (
        comp_res.summary["model_complexity"] > random_res.summary["model_complexity"]
    ), "Structured compositional data should require higher model complexity proxy than random data."

    print("All spot checks passed.")


if __name__ == "__main__":
    _run_spot_checks()
