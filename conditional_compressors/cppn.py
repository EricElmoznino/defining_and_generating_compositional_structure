import copy
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conditional_compressors.base import ConditionalCompressor
from lib.cppn_coordinates import CPPNCoordinateFrame, coordinate_grid_from_frame
from lib.helpers import (
    lookup_coordinate_batch,
    positional_encode,
    select_active_coordinate_indices,
    uses_shared_coordinate_indices,
)
from lib.independent_model_fitting import (
    BatchedModuleGroup,
    BatchedModuleStackUnavailable,
    adamw_step_batched_module_groups_,
    build_batched_module_groups,
    clip_batched_module_group_grad_norm_,
    find_batched_module_group_for_sample,
    flatten_batched_module_groups,
    iter_sample_coordinate_chunks,
    restore_best_batched_module_group_state,
    save_best_batched_module_group_state,
    set_batched_module_group_mode,
    zero_batched_module_group_grads,
)
from lib.type_aliases import CPPNModel


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


@dataclass
class _CoordinateEncodingSpec:
    use_positional_encoding: bool
    positional_encoding_bands: int
    include_raw_coordinates: bool
    input_dim: int


@dataclass
class CPPNConditionalFitResult:
    model: list[nn.Module]
    model_complexity: float
    residual_complexity: float
    total_conditional_complexity: float
    logs: dict[str, list[float]]
    summary: dict[str, Any]


def _unpack_cppn_model(
    model: CPPNModel,
) -> tuple[list[nn.Module], tuple[int, ...], CPPNCoordinateFrame | None]:
    if len(model) == 2:
        modules, shape = model
        return modules, shape, None
    if len(model) == 3:
        modules, shape, coordinate_frame = model
        if not isinstance(coordinate_frame, CPPNCoordinateFrame):
            raise ValueError("CPPN model coordinate metadata must be a coordinate frame.")
        return modules, shape, coordinate_frame
    raise ValueError("CPPN models must contain modules, shape, and optional frame.")


def _empty_conditional_fit_logs() -> dict[str, list[float]]:
    return {
        "step": [],
        "train_distill_bits_per_symbol": [],
        "val_distill_bits_per_symbol": [],
        "lr": [],
        "samples_seen": [],
    }


def _append_conditional_fit_log(
    logs: dict[str, list[float]],
    step: float,
    train_bits: float,
    val_bits: float,
    lr: float,
    samples_seen: float,
) -> None:
    logs["step"].append(float(step))
    logs["train_distill_bits_per_symbol"].append(float(train_bits))
    logs["val_distill_bits_per_symbol"].append(float(val_bits))
    logs["lr"].append(float(lr))
    logs["samples_seen"].append(float(samples_seen))


def _coords_for_group_positions(
    coords: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if coords.ndim == 2:
        return coords.unsqueeze(0).expand(int(positions.numel()), -1, -1)
    return coords.index_select(0, positions)


_CONDITIONAL_FIT_SUMMARY_FIELDS: tuple[tuple[str, type], ...] = (
    ("best_step", int),
    ("steps_ran", int),
    ("selected_step", int),
    ("samples_seen", int),
    ("selected_samples_seen", int),
    ("num_train_points", int),
    ("num_val_points", int),
    ("baseline_val_distill_bits_per_symbol", float),
    ("trained_val_distill_bits_per_symbol", float),
    ("selected_train_distill_bits_per_symbol", float),
    ("selected_val_distill_bits_per_symbol", float),
    ("absolute_improvement_over_baseline", float),
    ("relative_improvement_over_baseline", float),
    ("used_fallback_baseline", bool),
    ("model_complexity_mode", str),
    ("raw_model_complexity", float),
    ("model_complexity", float),
    ("residual_complexity", float),
    ("total_conditional_complexity", float),
    ("num_components", int),
    ("num_trainable_components", int),
    ("num_non_trainable_components", int),
)


def _build_conditional_fit_summary(**values: Any) -> dict[str, Any]:
    return {
        key: caster(values[key])
        for key, caster in _CONDITIONAL_FIT_SUMMARY_FIELDS
    }


class CPPNRoutedMixtureConditionalCompressor(ConditionalCompressor[CPPNModel]):
    """Conditional CPPN compressor via routed block-mixture fine-tuning.

    This compressor estimates conditional model cost
    `L_cond^g(m_parent | m_child_1, ..., m_child_k)` for CPPN models produced
    by the CPPN two-part code.

    The student is a routed mixture of child components: each parent-grid
    coordinate is mapped to exactly one block component (row-major block order).
    Legacy models query children in block-local normalized coordinates and the
    teacher in parent-global normalized coordinates. Models carrying a shared
    coordinate frame query both teacher and children in the same full-image
    normalized coordinates. Training minimizes `KL(q_teacher || q_student)`
    over coordinates/channels.

    The final conditional cost is:
        `L_cond = L_model_cond + L_resid_cond`,
    where `L_model_cond` is an adaptation-complexity proxy and `L_resid_cond`
    is full-grid KL bits between teacher and selected student. As in the two-part
    compressor, adapted weights are only kept if they beat the unadapted routed
    baseline on validation by configured absolute and relative margins.
    """

    def __init__(
        self,
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
        min_relative_improvement_over_constant: float = 0.02,
        min_absolute_improvement_bits: float = 0.02,
        model_complexity_mode: Literal[
            "excess_train_bits_auc", "samples_seen"
        ] = "excess_train_bits_auc",
        complexity_samples_scale: float = 1.0,
        device: torch.device | str | None = None,
        training_dtype: torch.dtype = torch.float32,
        seed: int | None = None,
        store_models_on_cpu: bool = True,
    ) -> None:
        """Initialize the routed-mixture CPPN conditional compressor.

        Training/selection pseudocode (per batch item):
            1. Build parent-grid teacher coordinates and routed child coordinates.
            2. Query teacher on global coordinates and cache teacher log-probs.
            3. Build unadapted routed baseline from child models.
            4. Split coordinates into train/validation (`random` or contiguous `block`).
            5. If any child component is trainable, optimize routed student to
               minimize mean `KL(q_teacher || q_student)` on train coordinates.
            6. Evaluate train/val distillation bits every `eval_every`, keep best
               checkpoint by minimum validation distillation bits, and early-stop
               with patience after `min_steps`.
            7. Compare trained checkpoint vs baseline on validation bits; keep the
               trained model only if both absolute and relative improvements exceed
               thresholds, otherwise fallback to baseline.
            8. Compute:
               - `L_model_cond`: adaptation complexity proxy
                 (`excess_train_bits_auc` or `samples_seen`, scaled).
               - `L_resid_cond`: full-grid KL bits between teacher and selected student.
               - `L_cond = L_model_cond + L_resid_cond`.

        Args:
            validation_fraction (float, optional): Fraction of coordinates used for
                validation/model selection.
            validation_mode (Literal["random", "block"], optional): Validation split
                strategy.
            min_validation_points (int, optional): Minimum held-out coordinates.
            max_steps (int, optional): Maximum optimization steps.
            min_steps (int, optional): Minimum steps before patience stopping.
            eval_every (int, optional): Evaluation period in optimization steps.
            patience_evals (int, optional): Number of non-improving evaluations
                before stopping.
            min_delta_bits (float, optional): Minimum validation improvement to
                reset patience.
            lr (float, optional): AdamW learning rate.
            min_lr_ratio (float, optional): Minimum LR ratio for cosine scheduling.
            lr_scheduler (Literal["none", "cosine"], optional): LR scheduling mode.
            weight_decay (float, optional): AdamW weight decay.
            coordinate_batch_size (int | None, optional): Minibatch size over
                coordinates; `None` means full-batch.
            collective_coordinate_batch_size (int | None, optional): Maximum number
                of sample-coordinate pairs evaluated in one vectorized forward chunk.
                This bounds GPU memory without changing each CPPN's coordinate
                minibatch size. `None` processes all active samples at once.
            gradient_clip_norm (float | None, optional): Global gradient clipping.
            min_relative_improvement_over_constant (float, optional): Minimum
                relative validation improvement over baseline required to keep
                adaptation.
            min_absolute_improvement_bits (float, optional): Minimum absolute
                validation improvement over baseline required to keep adaptation.
            model_complexity_mode (Literal["excess_train_bits_auc", "samples_seen"], optional):
                Complexity proxy for `L_model_cond`.
            complexity_samples_scale (float, optional): Scale applied to model
                complexity proxy.
            device (torch.device | str | None, optional): Training device. If None,
                uses CUDA when available else CPU.
            training_dtype (torch.dtype, optional): Floating dtype for coordinates
                and parameters.
            seed (int | None, optional): Base RNG seed for per-sample splits.
            store_models_on_cpu (bool, optional): If True, returned selected student
                components are moved to CPU.
        """
        super().__init__()
        if validation_fraction < 0.0 or validation_fraction >= 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if min_steps < 0:
            raise ValueError("min_steps must be >= 0.")
        if eval_every < 1:
            raise ValueError("eval_every must be >= 1.")
        if patience_evals < 1:
            raise ValueError("patience_evals must be >= 1.")
        if lr <= 0.0:
            raise ValueError("lr must be > 0.")
        if min_lr_ratio <= 0.0 or min_lr_ratio > 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1].")
        if (
            collective_coordinate_batch_size is not None
            and collective_coordinate_batch_size < 1
        ):
            raise ValueError("collective_coordinate_batch_size must be >= 1.")
        if min_absolute_improvement_bits < 0.0:
            raise ValueError("min_absolute_improvement_bits must be >= 0.")
        if model_complexity_mode not in {"excess_train_bits_auc", "samples_seen"}:
            raise ValueError(
                "model_complexity_mode must be one of: "
                "'excess_train_bits_auc', 'samples_seen'."
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

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
        self.min_relative_improvement_over_constant = (
            min_relative_improvement_over_constant
        )
        self.min_absolute_improvement_bits = min_absolute_improvement_bits
        self.model_complexity_mode = model_complexity_mode
        self.complexity_samples_scale = complexity_samples_scale
        self.device = device
        self.training_dtype = training_dtype
        self.seed = seed
        self.store_models_on_cpu = store_models_on_cpu

    def _sample_rng(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + idx)

    @staticmethod
    def _copy_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    @staticmethod
    def _first_input_dim(module: nn.Module) -> int | None:
        input_layer = getattr(module, "input_layer", None)
        if isinstance(input_layer, nn.Linear):
            return int(input_layer.in_features)
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                return int(submodule.in_features)
        return None

    @staticmethod
    def _infer_coordinate_encoding(
        num_spatial_dims: int, input_dim: int
    ) -> _CoordinateEncodingSpec:
        if input_dim == num_spatial_dims:
            return _CoordinateEncodingSpec(
                use_positional_encoding=False,
                positional_encoding_bands=0,
                include_raw_coordinates=True,
                input_dim=input_dim,
            )
        if input_dim % num_spatial_dims != 0:
            raise ValueError(
                f"Cannot infer coordinate encoding: input_dim={input_dim}, "
                f"num_spatial_dims={num_spatial_dims}."
            )
        ratio = input_dim // num_spatial_dims
        if ratio >= 2 and ratio % 2 == 0:
            return _CoordinateEncodingSpec(
                use_positional_encoding=True,
                positional_encoding_bands=ratio // 2,
                include_raw_coordinates=False,
                input_dim=input_dim,
            )
        if ratio >= 3 and (ratio - 1) % 2 == 0:
            return _CoordinateEncodingSpec(
                use_positional_encoding=True,
                positional_encoding_bands=(ratio - 1) // 2,
                include_raw_coordinates=True,
                input_dim=input_dim,
            )
        raise ValueError(
            f"Unsupported coordinate feature dimension: input_dim={input_dim}, "
            f"num_spatial_dims={num_spatial_dims}."
        )

    @staticmethod
    def _encode_coordinates(
        coords: torch.Tensor, spec: _CoordinateEncodingSpec
    ) -> torch.Tensor:
        if not spec.use_positional_encoding:
            return coords
        return positional_encode(
            coords,
            num_bands=spec.positional_encoding_bands,
            include_raw=spec.include_raw_coordinates,
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
            raise ValueError("validation_mode must be either 'random' or 'block'.")

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
        if selected_samples_seen <= 0:
            return 0.0
        samples = logs["samples_seen"]
        train_bits = logs["train_distill_bits_per_symbol"]
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

    def _lr_after_scheduler_step(self, steps: torch.Tensor) -> torch.Tensor:
        if self.lr_scheduler == "none":
            return torch.full_like(steps, float(self.lr), dtype=torch.float32)
        if self.lr_scheduler == "cosine":
            steps_float = steps.to(torch.float32)
            eta_min = self.lr * self.min_lr_ratio
            return eta_min + 0.5 * (self.lr - eta_min) * (
                1.0 + torch.cos(math.pi * steps_float / float(max(1, self.max_steps)))
            )
        raise ValueError("lr_scheduler must be one of: none, cosine.")

    @staticmethod
    def _infer_encoding_for_modules(
        modules: list[nn.Module],
        num_spatial_dims: int,
    ) -> _CoordinateEncodingSpec:
        input_dims = []
        for module in modules:
            input_dim = CPPNRoutedMixtureConditionalCompressor._first_input_dim(module)
            if input_dim is not None:
                input_dims.append(int(input_dim))
        if len(input_dims) > 0 and any(v != input_dims[0] for v in input_dims):
            raise ValueError(
                "Teacher and given CPPN components have inconsistent input dimensions."
            )
        if len(input_dims) == 0:
            return _CoordinateEncodingSpec(
                use_positional_encoding=False,
                positional_encoding_bands=0,
                include_raw_coordinates=True,
                input_dim=num_spatial_dims,
            )
        return CPPNRoutedMixtureConditionalCompressor._infer_coordinate_encoding(
            num_spatial_dims=num_spatial_dims,
            input_dim=input_dims[0],
        )

    def _forward_teacher_log_probs_batched(
        self,
        teacher_groups: list[BatchedModuleGroup],
        sample_indices: torch.Tensor,
        coord_indices: torch.Tensor,
        encoded_teacher_coords: torch.Tensor,
        num_channels: int,
        expected_vocab_size: int | None = None,
    ) -> torch.Tensor:
        coords = lookup_coordinate_batch(encoded_teacher_coords, coord_indices)
        logits: torch.Tensor | None = None

        for group in teacher_groups:
            rows, positions, _ = group.rows_for_sample_indices(sample_indices)
            if positions.numel() == 0:
                continue
            group_logits = group.forward_rows(
                rows,
                _coords_for_group_positions(coords, positions),
                use_initial=False,
            )
            if group_logits.ndim != 4:
                raise ValueError(
                    "Teacher output must have shape (batch, points, channels, vocab)."
                )
            if int(group_logits.shape[2]) != int(num_channels):
                raise ValueError(
                    "Teacher channel mismatch: "
                    f"expected {num_channels}, got {group_logits.shape[2]}."
                )
            vocab_size = int(group_logits.shape[3])
            if expected_vocab_size is not None and vocab_size != expected_vocab_size:
                raise ValueError(
                    "Teacher vocabulary mismatch: "
                    f"expected {expected_vocab_size}, got {vocab_size}."
                )
            if logits is None:
                logits = torch.empty(
                    (
                        int(sample_indices.numel()),
                        int(coord_indices.shape[1]),
                        int(num_channels),
                        vocab_size,
                    ),
                    device=coords.device,
                    dtype=group_logits.dtype,
                )
            elif int(logits.shape[-1]) != vocab_size:
                raise ValueError("Teacher vocabulary sizes must match across samples.")
            logits[positions] = group_logits.to(logits.dtype)

        if logits is None:
            raise ValueError("No teacher module matched the requested samples.")
        return F.log_softmax(logits, dim=-1)

    def _forward_routed_components_batched(
        self,
        component_groups_by_block: list[list[BatchedModuleGroup]],
        sample_indices: torch.Tensor,
        coord_indices: torch.Tensor,
        encoded_local_coords: torch.Tensor,
        block_ids: torch.Tensor,
        num_channels: int,
        vocab_size: int,
        *,
        use_initial: bool,
    ) -> torch.Tensor:
        num_samples = int(sample_indices.numel())
        num_coords = int(coord_indices.shape[1])
        shared_coord_indices = uses_shared_coordinate_indices(coord_indices)
        coord_index_row = (
            coord_indices if coord_indices.ndim == 1 else coord_indices[0]
        )
        coord_block_ids = (
            block_ids.index_select(0, coord_index_row)
            if shared_coord_indices
            else block_ids[coord_indices]
        )
        logits = torch.empty(
            (num_samples, num_coords, num_channels, vocab_size),
            device=encoded_local_coords.device,
            dtype=encoded_local_coords.dtype,
        )
        filled = torch.zeros(
            (num_samples, num_coords),
            device=encoded_local_coords.device,
            dtype=torch.bool,
        )

        for block_id_tensor in torch.unique(coord_block_ids, sorted=True):
            block_id = int(block_id_tensor.item())
            if block_id < 0 or block_id >= len(component_groups_by_block):
                raise ValueError(
                    f"Invalid routed block id={block_id} "
                    f"for {len(component_groups_by_block)} components."
                )
            if shared_coord_indices:
                block_mask_row = coord_block_ids == block_id
                coord_positions = torch.nonzero(
                    block_mask_row, as_tuple=False
                ).squeeze(-1)
                if coord_positions.numel() == 0:
                    continue
                selected_coord_indices = coord_index_row.index_select(
                    0, coord_positions
                )
                shared_coord_batch = encoded_local_coords.index_select(
                    0, selected_coord_indices
                )

                for group in component_groups_by_block[block_id]:
                    group_rows, group_positions, _ = group.rows_for_sample_indices(
                        sample_indices
                    )
                    if group_positions.numel() == 0:
                        continue
                    group_logits = group.forward_rows(
                        group_rows,
                        shared_coord_batch.unsqueeze(0).expand(
                            int(group_positions.numel()), -1, -1
                        ),
                        use_initial=use_initial,
                    )
                    if group_logits.ndim != 4:
                        raise ValueError(
                            "CPPN component output must have shape "
                            "(batch, points, channels, vocab)."
                        )
                    if int(group_logits.shape[2]) != int(num_channels):
                        raise ValueError(
                            "Component channel mismatch: "
                            f"expected {num_channels}, got {group_logits.shape[2]}."
                        )
                    if int(group_logits.shape[3]) != int(vocab_size):
                        raise ValueError(
                            "Component vocabulary mismatch: "
                            f"expected {vocab_size}, got {group_logits.shape[3]}."
                        )

                    sample_positions = group_positions.unsqueeze(1).expand(
                        -1, int(coord_positions.numel())
                    )
                    coord_positions_expanded = coord_positions.unsqueeze(0).expand(
                        int(group_positions.numel()), -1
                    )
                    logits[sample_positions, coord_positions_expanded] = (
                        group_logits.to(logits.dtype)
                    )
                    filled[sample_positions, coord_positions_expanded] = True
                continue

            block_mask = coord_block_ids == block_id
            for group in component_groups_by_block[block_id]:
                group_rows, group_positions, _ = group.rows_for_sample_indices(
                    sample_indices
                )
                if group_positions.numel() == 0:
                    continue

                group_block_mask = block_mask.index_select(0, group_positions)
                group_counts = group_block_mask.sum(dim=1)
                nonempty_rows = torch.nonzero(
                    group_counts > 0, as_tuple=False
                ).squeeze(-1)
                if nonempty_rows.numel() == 0:
                    continue

                active_group_rows = group_rows.index_select(0, nonempty_rows)
                active_positions = group_positions.index_select(0, nonempty_rows)
                active_mask = group_block_mask.index_select(0, nonempty_rows)
                active_counts = group_counts.index_select(0, nonempty_rows)
                max_count = int(active_counts.max().item())
                nonzero_positions = torch.nonzero(active_mask, as_tuple=False)
                pair_rows = nonzero_positions[:, 0]
                coord_positions = nonzero_positions[:, 1]
                rank_positions = (
                    active_mask.to(torch.long).cumsum(dim=1) - 1
                )[pair_rows, coord_positions]
                active_coord_indices = coord_indices.index_select(0, active_positions)
                selected_coord_indices = active_coord_indices[
                    pair_rows, coord_positions
                ]
                coord_batch = torch.zeros(
                    (
                        int(active_positions.numel()),
                        max_count,
                        int(encoded_local_coords.shape[-1]),
                    ),
                    device=encoded_local_coords.device,
                    dtype=encoded_local_coords.dtype,
                )
                coord_batch[pair_rows, rank_positions] = (
                    encoded_local_coords.index_select(0, selected_coord_indices)
                )

                group_logits = group.forward_rows(
                    active_group_rows,
                    coord_batch,
                    use_initial=use_initial,
                )
                if group_logits.ndim != 4:
                    raise ValueError(
                        "CPPN component output must have shape "
                        "(batch, points, channels, vocab)."
                    )
                if int(group_logits.shape[2]) != int(num_channels):
                    raise ValueError(
                        "Component channel mismatch: "
                        f"expected {num_channels}, got {group_logits.shape[2]}."
                    )
                if int(group_logits.shape[3]) != int(vocab_size):
                    raise ValueError(
                        "Component vocabulary mismatch: "
                        f"expected {vocab_size}, got {group_logits.shape[3]}."
                    )

                sample_positions = active_positions.index_select(0, pair_rows)
                logits[sample_positions, coord_positions] = group_logits[
                    pair_rows, rank_positions
                ].to(logits.dtype)
                filled[sample_positions, coord_positions] = True

        if not bool(filled.all().item()):
            raise ValueError("Every routed coordinate must map to one component.")
        return logits

    def _eval_batched_distill_bits_per_sample(
        self,
        component_groups_by_block: list[list[BatchedModuleGroup]],
        teacher_groups: list[BatchedModuleGroup],
        sample_indices: torch.Tensor,
        coord_indices: torch.Tensor,
        encoded_teacher_coords: torch.Tensor,
        encoded_local_coords: torch.Tensor,
        block_ids: torch.Tensor,
        num_channels: int,
        vocab_size: int,
        *,
        use_initial_components: bool,
        mean: bool,
    ) -> torch.Tensor:
        num_samples = int(sample_indices.numel())
        if num_samples == 0:
            return torch.empty(
                0, device=encoded_teacher_coords.device, dtype=torch.float32
            )

        component_groups = flatten_batched_module_groups(component_groups_by_block)
        set_batched_module_group_mode(component_groups, False)
        set_batched_module_group_mode(teacher_groups, False)

        num_coords = int(coord_indices.shape[1])
        total_bits = torch.zeros(
            num_samples, device=encoded_teacher_coords.device, dtype=torch.float32
        )
        selected_coord_indices = select_active_coordinate_indices(
            coord_indices, sample_indices
        )

        with torch.no_grad():
            for sample_start, sample_end, coord_start, coord_end in iter_sample_coordinate_chunks(
                num_samples, num_coords, self.collective_coordinate_batch_size
            ):
                chunk_samples = sample_indices[sample_start:sample_end]
                chunk_coord_indices = selected_coord_indices[
                    sample_start:sample_end, coord_start:coord_end
                ]
                teacher_log_probs = self._forward_teacher_log_probs_batched(
                    teacher_groups=teacher_groups,
                    sample_indices=chunk_samples,
                    coord_indices=chunk_coord_indices,
                    encoded_teacher_coords=encoded_teacher_coords,
                    num_channels=num_channels,
                    expected_vocab_size=vocab_size,
                )
                student_logits = self._forward_routed_components_batched(
                    component_groups_by_block=component_groups_by_block,
                    sample_indices=chunk_samples,
                    coord_indices=chunk_coord_indices,
                    encoded_local_coords=encoded_local_coords,
                    block_ids=block_ids,
                    num_channels=num_channels,
                    vocab_size=vocab_size,
                    use_initial=use_initial_components,
                )
                kl_nats = self._kl_nats_from_log_probs(
                    teacher_log_probs=teacher_log_probs,
                    student_logits=student_logits,
                )
                total_bits[sample_start:sample_end] += (
                    kl_nats.sum(dim=(1, 2)).detach().to(torch.float32)
                    / math.log(2.0)
                )

        if not mean:
            return total_bits
        return total_bits / float(num_coords * num_channels)

    def _backward_batched_distill_loss(
        self,
        component_groups_by_block: list[list[BatchedModuleGroup]],
        teacher_groups: list[BatchedModuleGroup],
        sample_indices: torch.Tensor,
        coord_indices: torch.Tensor,
        encoded_teacher_coords: torch.Tensor,
        encoded_local_coords: torch.Tensor,
        block_ids: torch.Tensor,
        num_channels: int,
        vocab_size: int,
    ) -> None:
        num_samples = int(sample_indices.numel())
        num_coords = int(coord_indices.shape[1])
        denom = float(num_coords * num_channels)

        for sample_start, sample_end, coord_start, coord_end in iter_sample_coordinate_chunks(
            num_samples, num_coords, self.collective_coordinate_batch_size
        ):
            chunk_samples = sample_indices[sample_start:sample_end]
            chunk_coord_indices = coord_indices[
                sample_start:sample_end, coord_start:coord_end
            ]
            with torch.no_grad():
                teacher_log_probs = self._forward_teacher_log_probs_batched(
                    teacher_groups=teacher_groups,
                    sample_indices=chunk_samples,
                    coord_indices=chunk_coord_indices,
                    encoded_teacher_coords=encoded_teacher_coords,
                    num_channels=num_channels,
                    expected_vocab_size=vocab_size,
                )
            student_logits = self._forward_routed_components_batched(
                component_groups_by_block=component_groups_by_block,
                sample_indices=chunk_samples,
                coord_indices=chunk_coord_indices,
                encoded_local_coords=encoded_local_coords,
                block_ids=block_ids,
                num_channels=num_channels,
                vocab_size=vocab_size,
                use_initial=False,
            )
            kl_nats = self._kl_nats_from_log_probs(
                teacher_log_probs=teacher_log_probs,
                student_logits=student_logits,
            )
            loss = (kl_nats.sum(dim=(1, 2)) / denom).sum()
            if loss.requires_grad:
                loss.backward()

    @staticmethod
    def _validate_cppn_models(
        model: CPPNModel, given_models: list[CPPNModel]
    ) -> tuple[
        int,
        tuple[int, ...],
        int,
        tuple[int, ...],
        CPPNCoordinateFrame | None,
    ]:
        teacher_models, parent_shape, parent_frame = _unpack_cppn_model(model)
        if len(parent_shape) < 2:
            raise ValueError("Parent model shape must be (*spatial_dims, channels).")
        batch_size = int(len(teacher_models))
        if len(given_models) == 0:
            raise ValueError(
                "given_models must be non-empty for CPPN conditional compression."
            )

        parent_spatial_shape = tuple(int(v) for v in parent_shape[:-1])
        num_channels = int(parent_shape[-1])
        if (
            parent_frame is not None
            and parent_frame.spatial_shape != parent_spatial_shape
        ):
            raise ValueError("Parent CPPN coordinate frame does not match model shape.")

        unpacked_given_models = [_unpack_cppn_model(given) for given in given_models]
        first_child_shape = unpacked_given_models[0][1]
        if len(first_child_shape) != len(parent_shape):
            raise ValueError("Child model shape rank must match parent shape rank.")
        block_spatial_shape = tuple(int(v) for v in first_child_shape[:-1])

        child_frames = []
        for child_models, child_shape, child_frame in unpacked_given_models:
            if len(child_models) != batch_size:
                raise ValueError(
                    "All given model batches must match the parent model batch size."
                )
            if len(child_shape) != len(parent_shape):
                raise ValueError("Child model shape rank must match parent shape rank.")
            if int(child_shape[-1]) != num_channels:
                raise ValueError("Child and parent channel counts must match.")
            if tuple(int(v) for v in child_shape[:-1]) != block_spatial_shape:
                raise ValueError(
                    "All child blocks must have identical spatial shapes for tiled routing."
                )
            if (
                child_frame is not None
                and child_frame.spatial_shape != block_spatial_shape
            ):
                raise ValueError("Child CPPN coordinate frame does not match model shape.")
            child_frames.append(child_frame)

        if parent_frame is None:
            if any(frame is not None for frame in child_frames):
                raise ValueError(
                    "CPPN conditional compression cannot mix local and shared "
                    "coordinate models."
                )
        elif any(frame is None for frame in child_frames):
            raise ValueError(
                "CPPN conditional compression cannot mix local and shared "
                "coordinate models."
            )

        blocks_per_axis = []
        for parent_size, block_size in zip(parent_spatial_shape, block_spatial_shape):
            if block_size <= 0 or parent_size <= 0:
                raise ValueError("Spatial dimensions must be positive.")
            if parent_size % block_size != 0:
                raise ValueError(
                    "Child block shapes must tile parent shape exactly on each axis."
                )
            blocks_per_axis.append(parent_size // block_size)

        expected_num_children = int(
            np.prod(np.asarray(blocks_per_axis, dtype=np.int64))
        )
        if expected_num_children != len(given_models):
            raise ValueError(
                "Number of given child models does not match expected tiling grid size. "
                f"Expected {expected_num_children}, got {len(given_models)}."
            )
        if parent_frame is not None:
            expected_origins = [
                tuple(
                    parent_frame.origin[axis] + block_coord[axis] * block_spatial_shape[axis]
                    for axis in range(len(parent_spatial_shape))
                )
                for block_coord in np.ndindex(*blocks_per_axis)
            ]
            for child_frame, expected_origin in zip(child_frames, expected_origins):
                assert child_frame is not None
                if child_frame.full_spatial_shape != parent_frame.full_spatial_shape:
                    raise ValueError(
                        "Shared-coordinate CPPN models must use the same full-image frame."
                    )
                if child_frame.origin != expected_origin:
                    raise ValueError(
                        "Shared-coordinate CPPN child frames must tile their parent "
                        "in row-major routing order."
                    )
        return (
            batch_size,
            parent_spatial_shape,
            num_channels,
            block_spatial_shape,
            parent_frame,
        )

    @staticmethod
    def _build_routing(
        parent_spatial_shape: tuple[int, ...],
        block_spatial_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        parent_frame: CPPNCoordinateFrame | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_dims = len(parent_spatial_shape)
        parent_sizes = torch.tensor(
            parent_spatial_shape, device=device, dtype=torch.long
        )
        block_sizes = torch.tensor(block_spatial_shape, device=device, dtype=torch.long)
        blocks_per_axis = parent_sizes // block_sizes
        if (
            parent_frame is not None
            and parent_frame.spatial_shape != parent_spatial_shape
        ):
            raise ValueError("Parent CPPN coordinate frame does not match model shape.")

        axes = [
            torch.arange(int(s), device=device, dtype=torch.long) for s in parent_sizes
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        flat_indices = torch.stack(mesh, dim=-1).reshape(-1, num_dims)  # (N, n)

        block_coords = torch.div(flat_indices, block_sizes, rounding_mode="floor")

        strides = torch.ones(num_dims, dtype=torch.long, device=device)
        for axis in range(num_dims - 2, -1, -1):
            strides[axis] = strides[axis + 1] * blocks_per_axis[axis + 1]
        block_ids = (block_coords * strides.view(1, -1)).sum(dim=-1)  # (N,)

        if parent_frame is None:
            local_indices = flat_indices - block_coords * block_sizes
            local_coords = torch.zeros(flat_indices.shape, device=device, dtype=dtype)
            for axis in range(num_dims):
                block_size = int(block_spatial_shape[axis])
                if block_size <= 1:
                    local_coords[:, axis] = 0.0
                else:
                    local_coords[:, axis] = (
                        2.0 * local_indices[:, axis].to(dtype)
                        / float(block_size - 1)
                        - 1.0
                    )
            teacher_coords = _coordinate_grid(
                parent_spatial_shape, device=device, dtype=dtype
            )
            student_coords = local_coords
        else:
            teacher_coords = coordinate_grid_from_frame(
                parent_frame, device=device, dtype=dtype
            )
            student_coords = teacher_coords
        return teacher_coords, student_coords, block_ids

    def _build_encoded_routing(
        self,
        parent_spatial_shape: tuple[int, ...],
        block_spatial_shape: tuple[int, ...],
        encoding_spec: _CoordinateEncodingSpec,
        parent_frame: CPPNCoordinateFrame | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_coords, student_coords, block_ids = self._build_routing(
            parent_spatial_shape=parent_spatial_shape,
            block_spatial_shape=block_spatial_shape,
            device=self.device,
            dtype=self.training_dtype,
            parent_frame=parent_frame,
        )
        encoded_teacher_coords = self._encode_coordinates(
            teacher_coords, encoding_spec
        )
        if parent_frame is None:
            encoded_student_coords = self._encode_coordinates(
                student_coords, encoding_spec
            )
        else:
            encoded_student_coords = encoded_teacher_coords
        return encoded_teacher_coords, encoded_student_coords, block_ids

    @staticmethod
    def _kl_nats_from_log_probs(
        teacher_log_probs: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        kl_per_symbol_nats = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        return kl_per_symbol_nats

    @staticmethod
    def _forward_student_components(
        components: nn.ModuleList,
        encoded_local_coords: torch.Tensor,
        block_ids: torch.Tensor,
        num_channels: int,
        vocab_size: int,
    ) -> torch.Tensor:
        num_points = int(encoded_local_coords.shape[0])
        logits = torch.empty(
            (num_points, num_channels, vocab_size),
            device=encoded_local_coords.device,
            dtype=encoded_local_coords.dtype,
        )
        unique_block_ids = torch.unique(block_ids, sorted=True)
        for block_id_tensor in unique_block_ids:
            block_id = int(block_id_tensor.item())
            if block_id < 0 or block_id >= len(components):
                raise ValueError(
                    f"Invalid routed block id={block_id} for {len(components)} components."
                )
            idx = torch.nonzero(block_ids == block_id, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            component_logits = components[block_id](encoded_local_coords[idx])
            if component_logits.ndim != 3:
                raise ValueError(
                    f"CPPN component output must be rank-3 (N, C, V), got rank={component_logits.ndim}."
                )
            if component_logits.shape[1] != num_channels:
                raise ValueError(
                    f"Component channel mismatch: expected {num_channels}, got {component_logits.shape[1]}."
                )
            if component_logits.shape[2] != vocab_size:
                raise ValueError(
                    f"Component vocabulary mismatch: expected {vocab_size}, got {component_logits.shape[2]}."
                )
            logits.index_copy_(0, idx, component_logits.to(logits.dtype))
        return logits

    def _eval_mean_distill_bits(
        self,
        components: nn.ModuleList,
        encoded_local_coords: torch.Tensor,
        block_ids: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        indices: torch.Tensor,
        num_channels: int,
        vocab_size: int,
    ) -> float:
        components.eval()
        with torch.no_grad():
            student_logits = self._forward_student_components(
                components=components,
                encoded_local_coords=encoded_local_coords[indices],
                block_ids=block_ids[indices],
                num_channels=num_channels,
                vocab_size=vocab_size,
            )
            kl_nats = self._kl_nats_from_log_probs(
                teacher_log_probs=teacher_log_probs[indices],
                student_logits=student_logits,
            )
            bits = kl_nats.mean() / math.log(2.0)
        return float(bits.item())

    def _eval_total_distill_bits(
        self,
        components: nn.ModuleList,
        encoded_local_coords: torch.Tensor,
        block_ids: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        num_channels: int,
        vocab_size: int,
    ) -> float:
        components.eval()
        with torch.no_grad():
            student_logits = self._forward_student_components(
                components=components,
                encoded_local_coords=encoded_local_coords,
                block_ids=block_ids,
                num_channels=num_channels,
                vocab_size=vocab_size,
            )
            kl_nats = self._kl_nats_from_log_probs(
                teacher_log_probs=teacher_log_probs,
                student_logits=student_logits,
            )
            bits = kl_nats.sum() / math.log(2.0)
        return float(bits.item())

    def _fit_single(
        self,
        teacher_model: nn.Module,
        given_components: list[nn.Module],
        parent_spatial_shape: tuple[int, ...],
        block_spatial_shape: tuple[int, ...],
        num_channels: int,
        sample_idx: int,
        parent_frame: CPPNCoordinateFrame | None,
    ) -> CPPNConditionalFitResult:
        device = self.device
        rng = self._sample_rng(sample_idx)

        encoding_spec = self._infer_encoding_for_modules(
            [teacher_model, *given_components],
            num_spatial_dims=len(parent_spatial_shape),
        )

        encoded_teacher_coords, encoded_local_coords, block_ids = (
            self._build_encoded_routing(
                parent_spatial_shape=parent_spatial_shape,
                block_spatial_shape=block_spatial_shape,
                encoding_spec=encoding_spec,
                parent_frame=parent_frame,
            )
        )

        if encoded_teacher_coords.shape[-1] != encoding_spec.input_dim:
            raise ValueError(
                "Encoded teacher-coordinate dimension does not match inferred CPPN input dimension."
            )
        if encoded_local_coords.shape[-1] != encoding_spec.input_dim:
            raise ValueError(
                "Encoded local-coordinate dimension does not match inferred CPPN input dimension."
            )

        teacher = copy.deepcopy(teacher_model).to(
            device=device, dtype=self.training_dtype
        )
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            teacher_logits = teacher(encoded_teacher_coords)
            if teacher_logits.ndim != 3:
                raise ValueError(
                    f"Teacher output must have shape (N, C, V), got rank={teacher_logits.ndim}."
                )
            if int(teacher_logits.shape[0]) != int(encoded_teacher_coords.shape[0]):
                raise ValueError(
                    "Teacher output length does not match coordinate count."
                )
            if int(teacher_logits.shape[1]) != int(num_channels):
                raise ValueError(
                    f"Teacher channel mismatch: expected {num_channels}, got {teacher_logits.shape[1]}."
                )
            vocab_size = int(teacher_logits.shape[2])
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        num_points = int(encoded_teacher_coords.shape[0])
        train_idx_cpu, val_idx_cpu = self._train_val_split(
            parent_spatial_shape,
            num_points,
            rng,
        )
        train_idx = train_idx_cpu.to(device=device)
        val_idx = val_idx_cpu.to(device=device)

        baseline_components = nn.ModuleList(
            [
                copy.deepcopy(component).to(device=device, dtype=self.training_dtype)
                for component in given_components
            ]
        )
        baseline_components.eval()

        baseline_val_bits = self._eval_mean_distill_bits(
            components=baseline_components,
            encoded_local_coords=encoded_local_coords,
            block_ids=block_ids,
            teacher_log_probs=teacher_log_probs,
            indices=val_idx,
            num_channels=num_channels,
            vocab_size=vocab_size,
        )

        logs = _empty_conditional_fit_logs()

        student_components = copy.deepcopy(baseline_components)
        trainable_params: list[nn.Parameter] = []
        trainable_component_flags: list[bool] = []
        for component in student_components:
            component_trainable = False
            for param in component.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
                    component_trainable = True
            trainable_component_flags.append(component_trainable)
        num_trainable_components = int(sum(trainable_component_flags))
        num_non_trainable_components = int(
            len(given_components) - num_trainable_components
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_score = float("inf")
        best_train_bits = float("inf")
        best_step = 0
        best_samples_seen = 0
        samples_seen = 0
        final_step = 0

        if num_trainable_components > 0:
            optimizer = torch.optim.AdamW(
                trainable_params, lr=self.lr, weight_decay=self.weight_decay
            )
            if self.lr_scheduler == "none":
                scheduler = None
            elif self.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, self.max_steps),
                    eta_min=self.lr * self.min_lr_ratio,
                )
            else:
                raise ValueError("lr_scheduler must be one of: none, cosine.")

            num_evals_without_improve = 0
            num_train = int(train_idx.numel())
            if (
                self.coordinate_batch_size is None
                or self.coordinate_batch_size >= num_train
            ):
                batch_size = num_train
            else:
                batch_size = int(self.coordinate_batch_size)
            batch_size = max(1, batch_size)

            for step in range(1, self.max_steps + 1):
                final_step = step
                student_components.train()

                if batch_size >= num_train:
                    batch_idx = train_idx
                else:
                    chosen_np = rng.choice(
                        train_idx_cpu.numpy(),
                        size=batch_size,
                        replace=num_train < batch_size,
                    )
                    batch_idx = torch.from_numpy(chosen_np.astype(np.int64)).to(device)

                optimizer.zero_grad(set_to_none=True)
                batch_student_logits = self._forward_student_components(
                    components=student_components,
                    encoded_local_coords=encoded_local_coords[batch_idx],
                    block_ids=block_ids[batch_idx],
                    num_channels=num_channels,
                    vocab_size=vocab_size,
                )
                batch_kl_nats = self._kl_nats_from_log_probs(
                    teacher_log_probs=teacher_log_probs[batch_idx],
                    student_logits=batch_student_logits,
                )
                loss = batch_kl_nats.mean()
                loss.backward()
                if self.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        student_components.parameters(), self.gradient_clip_norm
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                samples_seen += int(batch_idx.numel())

                if step % self.eval_every != 0 and step != 1 and step != self.max_steps:
                    continue

                train_bits = self._eval_mean_distill_bits(
                    components=student_components,
                    encoded_local_coords=encoded_local_coords,
                    block_ids=block_ids,
                    teacher_log_probs=teacher_log_probs,
                    indices=train_idx,
                    num_channels=num_channels,
                    vocab_size=vocab_size,
                )
                val_bits = self._eval_mean_distill_bits(
                    components=student_components,
                    encoded_local_coords=encoded_local_coords,
                    block_ids=block_ids,
                    teacher_log_probs=teacher_log_probs,
                    indices=val_idx,
                    num_channels=num_channels,
                    vocab_size=vocab_size,
                )
                lr_value = float(optimizer.param_groups[0]["lr"])

                _append_conditional_fit_log(
                    logs,
                    step,
                    train_bits,
                    val_bits,
                    lr_value,
                    samples_seen,
                )

                score = val_bits
                if score < best_score - self.min_delta_bits:
                    best_score = score
                    best_train_bits = train_bits
                    best_state = self._copy_state_dict(student_components)
                    best_step = step
                    best_samples_seen = samples_seen
                    num_evals_without_improve = 0
                else:
                    num_evals_without_improve += 1

                if (
                    step >= self.min_steps
                    and num_evals_without_improve >= self.patience_evals
                ):
                    break

            if best_state is not None:
                student_components.load_state_dict(best_state)
        else:
            final_step = 0
            best_step = 0
            best_samples_seen = 0
            best_train_bits = baseline_val_bits
            best_score = baseline_val_bits

        trained_val_bits = self._eval_mean_distill_bits(
            components=student_components,
            encoded_local_coords=encoded_local_coords,
            block_ids=block_ids,
            teacher_log_probs=teacher_log_probs,
            indices=val_idx,
            num_channels=num_channels,
            vocab_size=vocab_size,
        )
        abs_improvement = baseline_val_bits - trained_val_bits
        rel_improvement = abs_improvement / max(abs(baseline_val_bits), 1e-3)

        use_fallback_baseline = (
            num_trainable_components == 0
            or rel_improvement < self.min_relative_improvement_over_constant
            or abs_improvement < self.min_absolute_improvement_bits
        )

        if use_fallback_baseline:
            selected_components = baseline_components
            raw_model_complexity = 0.0
            model_complexity = 0.0
            selected_step = 0
            selected_samples_seen = 0
            selected_train_bits = baseline_val_bits
            selected_val_bits = baseline_val_bits
        else:
            selected_components = student_components
            selected_step = int(best_step)
            selected_samples_seen = int(best_samples_seen)
            selected_train_bits = float(best_train_bits)
            selected_val_bits = float(trained_val_bits)
            if self.model_complexity_mode == "samples_seen":
                raw_model_complexity = float(best_samples_seen)
            else:
                raw_model_complexity = self._complexity_from_logs_excess_train_bits_auc(
                    logs=logs,
                    selected_samples_seen=best_samples_seen,
                    selected_train_bits=selected_train_bits,
                    num_channels=num_channels,
                )
            model_complexity = float(
                raw_model_complexity * self.complexity_samples_scale
            )

        residual_complexity = self._eval_total_distill_bits(
            components=selected_components,
            encoded_local_coords=encoded_local_coords,
            block_ids=block_ids,
            teacher_log_probs=teacher_log_probs,
            num_channels=num_channels,
            vocab_size=vocab_size,
        )
        total_conditional_complexity = float(model_complexity + residual_complexity)

        target_device = torch.device("cpu") if self.store_models_on_cpu else device
        stored_components = [
            copy.deepcopy(m).to(target_device) for m in selected_components
        ]
        for module in stored_components:
            module.eval()

        summary = _build_conditional_fit_summary(
            best_step=best_step,
            steps_ran=final_step,
            selected_step=selected_step,
            samples_seen=samples_seen,
            selected_samples_seen=selected_samples_seen,
            num_train_points=train_idx.numel(),
            num_val_points=val_idx.numel(),
            baseline_val_distill_bits_per_symbol=baseline_val_bits,
            trained_val_distill_bits_per_symbol=trained_val_bits,
            selected_train_distill_bits_per_symbol=selected_train_bits,
            selected_val_distill_bits_per_symbol=selected_val_bits,
            absolute_improvement_over_baseline=abs_improvement,
            relative_improvement_over_baseline=rel_improvement,
            used_fallback_baseline=use_fallback_baseline,
            model_complexity_mode=self.model_complexity_mode,
            raw_model_complexity=raw_model_complexity,
            model_complexity=model_complexity,
            residual_complexity=residual_complexity,
            total_conditional_complexity=total_conditional_complexity,
            num_components=len(given_components),
            num_trainable_components=num_trainable_components,
            num_non_trainable_components=num_non_trainable_components,
        )

        return CPPNConditionalFitResult(
            model=stored_components,
            model_complexity=float(model_complexity),
            residual_complexity=float(residual_complexity),
            total_conditional_complexity=float(total_conditional_complexity),
            logs=logs,
            summary=summary,
        )

    def _fit_batched(
        self,
        model: CPPNModel,
        given_models: list[CPPNModel],
        batch_size: int,
        parent_spatial_shape: tuple[int, ...],
        block_spatial_shape: tuple[int, ...],
        num_channels: int,
        parent_frame: CPPNCoordinateFrame | None,
    ) -> list[CPPNConditionalFitResult]:
        teacher_models, _, _ = _unpack_cppn_model(model)
        given_model_batches = [gm[0] for gm in given_models]
        device = self.device

        sample_specs = []
        num_trainable_components_list: list[int] = []
        for sample_idx in range(batch_size):
            components = [given_batch[sample_idx] for given_batch in given_model_batches]
            sample_specs.append(
                self._infer_encoding_for_modules(
                    [teacher_models[sample_idx], *components],
                    num_spatial_dims=len(parent_spatial_shape),
                )
            )
            num_trainable_components_list.append(
                sum(
                    any(param.requires_grad for param in component.parameters())
                    for component in components
                )
            )

        encoding_spec = sample_specs[0]
        if any(spec != encoding_spec for spec in sample_specs):
            raise BatchedModuleStackUnavailable(
                "Batched conditional fitting requires a shared coordinate encoding."
            )

        encoded_teacher_coords, encoded_local_coords, block_ids = (
            self._build_encoded_routing(
                parent_spatial_shape=parent_spatial_shape,
                block_spatial_shape=block_spatial_shape,
                encoding_spec=encoding_spec,
                parent_frame=parent_frame,
            )
        )

        if encoded_teacher_coords.shape[-1] != encoding_spec.input_dim:
            raise ValueError(
                "Encoded teacher-coordinate dimension does not match inferred CPPN input dimension."
            )
        if encoded_local_coords.shape[-1] != encoding_spec.input_dim:
            raise ValueError(
                "Encoded local-coordinate dimension does not match inferred CPPN input dimension."
            )

        num_points = int(encoded_teacher_coords.shape[0])
        coordinate_rng = self._sample_rng(0)
        train_idx_cpu, val_idx_cpu = self._train_val_split(
            parent_spatial_shape,
            num_points,
            coordinate_rng,
        )
        train_idx_np = train_idx_cpu.numpy()
        train_idx_row = train_idx_cpu.to(device=device)
        val_idx_row = val_idx_cpu.to(device=device)
        train_idx = train_idx_row.unsqueeze(0).expand(batch_size, -1)
        val_idx = val_idx_row.unsqueeze(0).expand(batch_size, -1)
        num_train = int(train_idx_row.numel())
        num_val = int(val_idx_row.numel())
        all_sample_indices = torch.arange(batch_size, device=device, dtype=torch.long)

        teacher_groups = build_batched_module_groups(
            list(teacher_models), device=device, dtype=self.training_dtype
        )
        component_groups_by_block = [
            build_batched_module_groups(
                [given_batch[sample_idx] for sample_idx in range(batch_size)],
                device=device,
                dtype=self.training_dtype,
            )
            for given_batch in given_model_batches
        ]
        component_groups = flatten_batched_module_groups(component_groups_by_block)
        set_batched_module_group_mode(teacher_groups, False)
        set_batched_module_group_mode(component_groups, False)

        probe_indices = []
        for block_id in range(len(given_model_batches)):
            matches = torch.nonzero(block_ids == block_id, as_tuple=False).squeeze(-1)
            if matches.numel() == 0:
                raise ValueError(f"No coordinates route to child block {block_id}.")
            probe_indices.append(matches[0])
        probe_coord_idx = torch.stack(probe_indices, dim=0).unsqueeze(0).expand(
            batch_size, -1
        )
        try:
            probe_teacher_log_probs = self._forward_teacher_log_probs_batched(
                teacher_groups=teacher_groups,
                sample_indices=all_sample_indices,
                coord_indices=probe_coord_idx,
                encoded_teacher_coords=encoded_teacher_coords,
                num_channels=num_channels,
            )
            vocab_size = int(probe_teacher_log_probs.shape[-1])
            _ = self._forward_routed_components_batched(
                component_groups_by_block=component_groups_by_block,
                sample_indices=all_sample_indices,
                coord_indices=probe_coord_idx,
                encoded_local_coords=encoded_local_coords,
                block_ids=block_ids,
                num_channels=num_channels,
                vocab_size=vocab_size,
                use_initial=True,
            )
        except RuntimeError as exc:
            raise BatchedModuleStackUnavailable(
                "CPPN modules do not support vectorized conditional fitting."
            ) from exc

        baseline_val_bits = self._eval_batched_distill_bits_per_sample(
            component_groups_by_block=component_groups_by_block,
            teacher_groups=teacher_groups,
            sample_indices=all_sample_indices,
            coord_indices=val_idx,
            encoded_teacher_coords=encoded_teacher_coords,
            encoded_local_coords=encoded_local_coords,
            block_ids=block_ids,
            num_channels=num_channels,
            vocab_size=vocab_size,
            use_initial_components=True,
            mean=True,
        )

        logs = [_empty_conditional_fit_logs() for _ in range(batch_size)]
        num_trainable_components = torch.tensor(
            num_trainable_components_list, device=device, dtype=torch.long
        )
        trainable_sample_mask = num_trainable_components > 0

        best_score = torch.full((batch_size,), float("inf"), device=device)
        best_val_bits = torch.full((batch_size,), float("inf"), device=device)
        best_train_bits = torch.full((batch_size,), float("inf"), device=device)
        best_step = torch.zeros(batch_size, device=device, dtype=torch.long)
        best_samples_seen = torch.zeros(batch_size, device=device, dtype=torch.long)
        has_best_state = torch.zeros(batch_size, device=device, dtype=torch.bool)
        num_evals_without_improve = torch.zeros(
            batch_size, device=device, dtype=torch.long
        )
        samples_seen = torch.zeros(batch_size, device=device, dtype=torch.long)
        steps_ran = torch.zeros(batch_size, device=device, dtype=torch.long)
        optimizer_steps = torch.zeros(batch_size, device=device, dtype=torch.long)
        current_lr = torch.full(
            (batch_size,), float(self.lr), device=device, dtype=torch.float32
        )
        active_mask = trainable_sample_mask.clone()

        no_trainable_indices = torch.nonzero(
            ~trainable_sample_mask, as_tuple=False
        ).squeeze(-1)
        if no_trainable_indices.numel() > 0:
            best_score[no_trainable_indices] = baseline_val_bits.index_select(
                0, no_trainable_indices
            )
            best_val_bits[no_trainable_indices] = baseline_val_bits.index_select(
                0, no_trainable_indices
            )
            best_train_bits[no_trainable_indices] = baseline_val_bits.index_select(
                0, no_trainable_indices
            )

        if self.coordinate_batch_size is None or self.coordinate_batch_size >= num_train:
            per_sample_batch_size = num_train
        else:
            per_sample_batch_size = int(self.coordinate_batch_size)
        per_sample_batch_size = max(1, per_sample_batch_size)

        for step in range(1, self.max_steps + 1):
            active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break

            set_batched_module_group_mode(component_groups, True)
            if per_sample_batch_size >= num_train:
                batch_coord_idx = train_idx[:1].expand(
                    int(active_indices.numel()), -1
                )
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

            zero_batched_module_group_grads(component_groups)
            self._backward_batched_distill_loss(
                component_groups_by_block=component_groups_by_block,
                teacher_groups=teacher_groups,
                sample_indices=active_indices,
                coord_indices=batch_coord_idx,
                encoded_teacher_coords=encoded_teacher_coords,
                encoded_local_coords=encoded_local_coords,
                block_ids=block_ids,
                num_channels=num_channels,
                vocab_size=vocab_size,
            )
            if self.gradient_clip_norm is not None:
                clip_batched_module_group_grad_norm_(
                    component_groups,
                    active_indices,
                    float(self.gradient_clip_norm),
                )

            optimizer_steps[active_indices] += 1
            adamw_step_batched_module_groups_(
                groups=component_groups,
                active_indices=active_indices,
                lr_values=current_lr,
                step_counts=optimizer_steps,
                weight_decay=self.weight_decay,
            )
            current_lr[active_indices] = self._lr_after_scheduler_step(
                optimizer_steps.index_select(0, active_indices)
            ).to(device=device)
            zero_batched_module_group_grads(component_groups)

            samples_seen[active_indices] += int(batch_coord_idx.shape[1])
            steps_ran[active_indices] = int(step)

            if step % self.eval_every != 0 and step != 1 and step != self.max_steps:
                continue

            set_batched_module_group_mode(component_groups, False)
            train_bits = self._eval_batched_distill_bits_per_sample(
                component_groups_by_block=component_groups_by_block,
                teacher_groups=teacher_groups,
                sample_indices=active_indices,
                coord_indices=train_idx,
                encoded_teacher_coords=encoded_teacher_coords,
                encoded_local_coords=encoded_local_coords,
                block_ids=block_ids,
                num_channels=num_channels,
                vocab_size=vocab_size,
                use_initial_components=False,
                mean=True,
            )
            val_bits = self._eval_batched_distill_bits_per_sample(
                component_groups_by_block=component_groups_by_block,
                teacher_groups=teacher_groups,
                sample_indices=active_indices,
                coord_indices=val_idx,
                encoded_teacher_coords=encoded_teacher_coords,
                encoded_local_coords=encoded_local_coords,
                block_ids=block_ids,
                num_channels=num_channels,
                vocab_size=vocab_size,
                use_initial_components=False,
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
                _append_conditional_fit_log(
                    logs[sample_idx],
                    step,
                    train_bits_list[local_pos],
                    val_bits_list[local_pos],
                    lr_values_list[local_pos],
                    samples_seen_list[local_pos],
                )

            previous_best = best_score.index_select(0, active_indices)
            improved = val_bits < previous_best - self.min_delta_bits
            if improved.any():
                improved_indices = active_indices[improved]
                best_score[improved_indices] = val_bits[improved]
                best_val_bits[improved_indices] = val_bits[improved]
                best_train_bits[improved_indices] = train_bits[improved]
                best_step[improved_indices] = int(step)
                best_samples_seen[improved_indices] = samples_seen[improved_indices]
                has_best_state[improved_indices] = True
                num_evals_without_improve[improved_indices] = 0
                save_best_batched_module_group_state(
                    component_groups,
                    improved_indices,
                )

            if (~improved).any():
                not_improved_indices = active_indices[~improved]
                num_evals_without_improve[not_improved_indices] += 1

            should_stop = (
                step >= self.min_steps
            ) & (
                num_evals_without_improve.index_select(0, active_indices)
                >= self.patience_evals
            )
            if should_stop.any():
                stopped_indices = active_indices[should_stop]
                active_mask[stopped_indices] = False
                steps_ran[stopped_indices] = int(step)

        restore_indices = torch.nonzero(has_best_state, as_tuple=False).squeeze(-1)
        if restore_indices.numel() > 0:
            restore_best_batched_module_group_state(component_groups, restore_indices)

        trained_val_bits = baseline_val_bits.clone().to(torch.float32)
        trained_indices_for_eval = torch.nonzero(
            trainable_sample_mask, as_tuple=False
        ).squeeze(-1)
        if trained_indices_for_eval.numel() > 0:
            trained_val_bits[trained_indices_for_eval] = (
                self._eval_batched_distill_bits_per_sample(
                    component_groups_by_block=component_groups_by_block,
                    teacher_groups=teacher_groups,
                    sample_indices=trained_indices_for_eval,
                    coord_indices=val_idx,
                    encoded_teacher_coords=encoded_teacher_coords,
                    encoded_local_coords=encoded_local_coords,
                    block_ids=block_ids,
                    num_channels=num_channels,
                    vocab_size=vocab_size,
                    use_initial_components=False,
                    mean=True,
                )
            )

        abs_improvement = baseline_val_bits - trained_val_bits
        rel_improvement = abs_improvement / torch.clamp(
            baseline_val_bits.abs(), min=1e-3
        )

        use_fallback_baseline = (
            ~trainable_sample_mask
            | (rel_improvement < self.min_relative_improvement_over_constant)
            | (abs_improvement < self.min_absolute_improvement_bits)
        )

        raw_model_complexity = torch.zeros(
            batch_size, device=device, dtype=torch.float32
        )
        model_complexity = torch.zeros(batch_size, device=device, dtype=torch.float32)
        selected_train_bits = torch.where(
            use_fallback_baseline,
            baseline_val_bits,
            best_train_bits.to(torch.float32),
        )
        selected_val_bits = torch.where(
            use_fallback_baseline, baseline_val_bits, trained_val_bits
        )
        selected_samples_seen = torch.where(
            use_fallback_baseline,
            torch.zeros_like(best_samples_seen),
            best_samples_seen,
        )
        selected_step = torch.where(
            use_fallback_baseline, torch.zeros_like(best_step), best_step
        )

        use_fallback_baseline_list = use_fallback_baseline.detach().cpu().tolist()
        best_samples_seen_list = best_samples_seen.detach().cpu().tolist()
        best_train_bits_list = best_train_bits.detach().cpu().tolist()
        for sample_idx in range(batch_size):
            if bool(use_fallback_baseline_list[sample_idx]):
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

        residual_complexity = torch.empty(
            batch_size, device=device, dtype=torch.float32
        )
        all_coord_idx = torch.arange(
            num_points, device=device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        fallback_indices = torch.nonzero(
            use_fallback_baseline, as_tuple=False
        ).squeeze(-1)
        if fallback_indices.numel() > 0:
            residual_complexity[fallback_indices] = (
                self._eval_batched_distill_bits_per_sample(
                    component_groups_by_block=component_groups_by_block,
                    teacher_groups=teacher_groups,
                    sample_indices=fallback_indices,
                    coord_indices=all_coord_idx,
                    encoded_teacher_coords=encoded_teacher_coords,
                    encoded_local_coords=encoded_local_coords,
                    block_ids=block_ids,
                    num_channels=num_channels,
                    vocab_size=vocab_size,
                    use_initial_components=True,
                    mean=False,
                )
            )
        trained_indices = torch.nonzero(
            ~use_fallback_baseline, as_tuple=False
        ).squeeze(-1)
        if trained_indices.numel() > 0:
            residual_complexity[trained_indices] = (
                self._eval_batched_distill_bits_per_sample(
                    component_groups_by_block=component_groups_by_block,
                    teacher_groups=teacher_groups,
                    sample_indices=trained_indices,
                    coord_indices=all_coord_idx,
                    encoded_teacher_coords=encoded_teacher_coords,
                    encoded_local_coords=encoded_local_coords,
                    block_ids=block_ids,
                    num_channels=num_channels,
                    vocab_size=vocab_size,
                    use_initial_components=False,
                    mean=False,
                )
            )

        total_conditional_complexity = model_complexity + residual_complexity
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
        abs_improvement_list = abs_improvement.detach().cpu().tolist()
        rel_improvement_list = rel_improvement.detach().cpu().tolist()
        raw_model_complexity_list = raw_model_complexity.detach().cpu().tolist()
        model_complexity_list = model_complexity.detach().cpu().tolist()
        residual_complexity_list = residual_complexity.detach().cpu().tolist()
        total_conditional_complexity_list = (
            total_conditional_complexity.detach().cpu().tolist()
        )
        num_trainable_components_list = num_trainable_components.detach().cpu().tolist()

        results: list[CPPNConditionalFitResult] = []
        for sample_idx in range(batch_size):
            use_initial = bool(use_fallback_baseline_list[sample_idx])
            stored_components = []
            for block_groups in component_groups_by_block:
                group = find_batched_module_group_for_sample(block_groups, sample_idx)
                stored_components.append(
                    group.materialize(
                        sample_idx,
                        use_initial=use_initial,
                        target_device=target_device,
                    )
                )

            summary = _build_conditional_fit_summary(
                best_step=best_step_list[sample_idx],
                steps_ran=steps_ran_list[sample_idx],
                selected_step=selected_step_list[sample_idx],
                samples_seen=samples_seen_list[sample_idx],
                selected_samples_seen=selected_samples_seen_list[sample_idx],
                num_train_points=num_train,
                num_val_points=num_val,
                baseline_val_distill_bits_per_symbol=baseline_val_bits_list[
                    sample_idx
                ],
                trained_val_distill_bits_per_symbol=trained_val_bits_list[sample_idx],
                selected_train_distill_bits_per_symbol=selected_train_bits_list[
                    sample_idx
                ],
                selected_val_distill_bits_per_symbol=selected_val_bits_list[
                    sample_idx
                ],
                absolute_improvement_over_baseline=abs_improvement_list[sample_idx],
                relative_improvement_over_baseline=rel_improvement_list[sample_idx],
                used_fallback_baseline=use_fallback_baseline_list[sample_idx],
                model_complexity_mode=self.model_complexity_mode,
                raw_model_complexity=raw_model_complexity_list[sample_idx],
                model_complexity=model_complexity_list[sample_idx],
                residual_complexity=residual_complexity_list[sample_idx],
                total_conditional_complexity=total_conditional_complexity_list[
                    sample_idx
                ],
                num_components=len(given_models),
                num_trainable_components=num_trainable_components_list[sample_idx],
                num_non_trainable_components=(
                    len(given_models) - num_trainable_components_list[sample_idx]
                ),
            )

            results.append(
                CPPNConditionalFitResult(
                    model=stored_components,
                    model_complexity=float(model_complexity_list[sample_idx]),
                    residual_complexity=float(residual_complexity_list[sample_idx]),
                    total_conditional_complexity=float(
                        total_conditional_complexity_list[sample_idx]
                    ),
                    logs=logs[sample_idx],
                    summary=summary,
                )
            )

        return results

    def fit_with_logs(
        self, model: CPPNModel, given_models: list[CPPNModel]
    ) -> list[CPPNConditionalFitResult]:
        (
            batch_size,
            parent_spatial_shape,
            num_channels,
            block_spatial_shape,
            parent_frame,
        ) = self._validate_cppn_models(model=model, given_models=given_models)

        teacher_models, _, _ = _unpack_cppn_model(model)
        given_model_batches = [gm[0] for gm in given_models]

        if batch_size > 1:
            try:
                return self._fit_batched(
                    model=model,
                    given_models=given_models,
                    batch_size=batch_size,
                    parent_spatial_shape=parent_spatial_shape,
                    block_spatial_shape=block_spatial_shape,
                    num_channels=num_channels,
                    parent_frame=parent_frame,
                )
            except BatchedModuleStackUnavailable:
                pass

        results: list[CPPNConditionalFitResult] = []
        for i in range(batch_size):
            teacher = teacher_models[i]
            components = [given_batch[i] for given_batch in given_model_batches]
            result = self._fit_single(
                teacher_model=teacher,
                given_components=components,
                parent_spatial_shape=parent_spatial_shape,
                block_spatial_shape=block_spatial_shape,
                num_channels=num_channels,
                sample_idx=i,
                parent_frame=parent_frame,
            )
            results.append(result)
        return results

    def compress(self, model: CPPNModel, given_models: list[CPPNModel]) -> torch.Tensor:
        fit_results = self.fit_with_logs(model=model, given_models=given_models)
        costs = [result.total_conditional_complexity for result in fit_results]
        return torch.tensor(costs, dtype=torch.float32, device=self.device)


def _split_blocks(
    data: torch.Tensor,
    block_spatial_shape: tuple[int, ...],
) -> list[torch.Tensor]:
    if data.ndim < 3:
        raise ValueError("Data must be shaped (batch, *spatial_dims, channels).")
    spatial_shape = tuple(int(v) for v in data.shape[1:-1])
    if len(spatial_shape) != len(block_spatial_shape):
        raise ValueError("block_spatial_shape rank must match data spatial rank.")

    blocks_per_axis = []
    for parent_size, block_size in zip(spatial_shape, block_spatial_shape):
        if parent_size % block_size != 0:
            raise ValueError("Block shape must tile data spatial shape exactly.")
        blocks_per_axis.append(parent_size // block_size)

    blocks = []
    for block_coords in np.ndindex(*blocks_per_axis):
        slices = tuple(
            slice(coord * block_size, (coord + 1) * block_size)
            for coord, block_size in zip(block_coords, block_spatial_shape)
        )
        blocks.append(data[(slice(None), *slices, slice(None))])
    return blocks


def _run_spot_checks() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    class _ToyConstantCPPN(nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("logits", logits.view(1, 1, -1).to(torch.float32))

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            return self.logits.expand(coords.shape[0], -1, -1)

    class _ToyLinearCPPN(nn.Module):
        def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
            super().__init__()
            out_dim, in_dim = int(weight.shape[0]), int(weight.shape[1])
            self.input_layer = nn.Linear(in_dim, out_dim, bias=True)
            with torch.no_grad():
                self.input_layer.weight.copy_(weight.to(torch.float32))
                self.input_layer.bias.copy_(bias.to(torch.float32))

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            logits = self.input_layer(coords)
            return logits.view(coords.shape[0], 1, -1)

    class _ToyPiecewiseTeacher(nn.Module):
        def __init__(
            self,
            components: list[nn.Module],
            parent_spatial_shape: tuple[int, ...],
            block_spatial_shape: tuple[int, ...],
        ) -> None:
            super().__init__()
            self.components = nn.ModuleList([copy.deepcopy(c) for c in components])
            self.register_buffer(
                "parent_sizes",
                torch.tensor(parent_spatial_shape, dtype=torch.long),
            )
            self.register_buffer(
                "block_sizes",
                torch.tensor(block_spatial_shape, dtype=torch.long),
            )
            blocks_per_axis = self.parent_sizes // self.block_sizes
            strides = torch.ones_like(blocks_per_axis)
            for axis in range(strides.numel() - 2, -1, -1):
                strides[axis] = strides[axis + 1] * blocks_per_axis[axis + 1]
            self.register_buffer("strides", strides)
            self.num_channels = 1
            self.vocab_size = 2

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            num_points, num_dims = coords.shape[0], coords.shape[1]
            parent_sizes = self.parent_sizes.to(coords.device)
            block_sizes = self.block_sizes.to(coords.device)
            strides = self.strides.to(coords.device)

            integer_indices = []
            for axis in range(num_dims):
                size = int(parent_sizes[axis].item())
                if size <= 1:
                    idx = torch.zeros(
                        num_points, dtype=torch.long, device=coords.device
                    )
                else:
                    idx = torch.round(
                        0.5 * (coords[:, axis] + 1.0) * float(size - 1)
                    ).to(torch.long)
                    idx = torch.clamp(idx, min=0, max=size - 1)
                integer_indices.append(idx)
            u = torch.stack(integer_indices, dim=-1)  # (N, n)

            block_coords = torch.div(u, block_sizes.view(1, -1), rounding_mode="floor")
            local_indices = u - block_coords * block_sizes.view(1, -1)
            block_ids = (block_coords * strides.view(1, -1)).sum(dim=-1)

            local_coords = torch.zeros(
                (num_points, num_dims),
                device=coords.device,
                dtype=coords.dtype,
            )
            for axis in range(num_dims):
                b = int(block_sizes[axis].item())
                if b <= 1:
                    local_coords[:, axis] = 0.0
                else:
                    local_coords[:, axis] = (
                        2.0 * local_indices[:, axis].to(coords.dtype) / float(b - 1)
                        - 1.0
                    )

            logits = torch.empty(
                (num_points, self.num_channels, self.vocab_size),
                device=coords.device,
                dtype=coords.dtype,
            )
            for component_id, component in enumerate(self.components):
                idx = torch.nonzero(block_ids == component_id, as_tuple=False).squeeze(
                    -1
                )
                if idx.numel() == 0:
                    continue
                logits.index_copy_(0, idx, component(local_coords[idx]))
            return logits

    parent_spatial_shape = (8, 8)
    block_spatial_shape = (4, 4)
    parent_shape = (*parent_spatial_shape, 1)
    block_shape = (*block_spatial_shape, 1)

    cond = CPPNRoutedMixtureConditionalCompressor(
        validation_fraction=0.2,
        validation_mode="block",
        min_validation_points=4,
        max_steps=80,
        min_steps=5,
        eval_every=5,
        patience_evals=4,
        lr=3e-3,
        weight_decay=1e-4,
        min_relative_improvement_over_constant=0.01,
        min_absolute_improvement_bits=0.001,
        complexity_samples_scale=1.0,
        device="cpu",
        seed=17,
        store_models_on_cpu=True,
    )
    cond_force_fallback = CPPNRoutedMixtureConditionalCompressor(
        validation_fraction=0.2,
        validation_mode="block",
        min_validation_points=4,
        max_steps=40,
        min_steps=5,
        eval_every=5,
        patience_evals=4,
        lr=3e-3,
        weight_decay=1e-4,
        min_relative_improvement_over_constant=1e9,
        min_absolute_improvement_bits=1e9,
        complexity_samples_scale=1.0,
        device="cpu",
        seed=17,
        store_models_on_cpu=True,
    )

    print("CPPN conditional compressor spot checks")
    print("-" * 88)

    # Test 1: all constant components => no trainable adaptation.
    teacher_const = _ToyConstantCPPN(torch.tensor([2.0, -2.0], dtype=torch.float32))
    parent_model_const: CPPNModel = ([teacher_const], parent_shape)
    child_models_const: list[CPPNModel] = [
        (
            [_ToyConstantCPPN(torch.tensor([2.0, -2.0], dtype=torch.float32))],
            block_shape,
        )
        for _ in range(4)
    ]
    fit_const = cond.fit_with_logs(parent_model_const, child_models_const)[0]
    print(
        "Test 1 | "
        f"fallback={fit_const.summary['used_fallback_baseline']} "
        f"trainable={fit_const.summary['num_trainable_components']} "
        f"model_c={fit_const.summary['model_complexity']:.4f} "
        f"resid_c={fit_const.summary['residual_complexity']:.4f}"
    )
    assert fit_const.summary["num_trainable_components"] == 0
    assert fit_const.summary["used_fallback_baseline"]
    assert abs(fit_const.summary["model_complexity"]) < 1e-8

    # Test 2: routed ordering sanity (correct order should beat permuted order).
    components = [
        _ToyLinearCPPN(
            weight=torch.tensor([[4.0, 0.0], [-4.0, 0.0]]),
            bias=torch.tensor([0.0, 0.0]),
        ),
        _ToyLinearCPPN(
            weight=torch.tensor([[0.0, 4.0], [0.0, -4.0]]),
            bias=torch.tensor([0.0, 0.0]),
        ),
        _ToyLinearCPPN(
            weight=torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
            bias=torch.tensor([0.0, 0.0]),
        ),
        _ToyLinearCPPN(
            weight=torch.tensor([[3.0, -3.0], [-3.0, 3.0]]),
            bias=torch.tensor([0.0, 0.0]),
        ),
    ]
    teacher_piecewise = _ToyPiecewiseTeacher(
        components=components,
        parent_spatial_shape=parent_spatial_shape,
        block_spatial_shape=block_spatial_shape,
    )
    parent_model_piecewise: CPPNModel = ([teacher_piecewise], parent_shape)
    child_models_related: list[CPPNModel] = [
        ([copy.deepcopy(component)], block_shape) for component in components
    ]
    perm = [1, 0, 3, 2]
    child_models_permuted: list[CPPNModel] = [
        ([copy.deepcopy(components[i])], block_shape) for i in perm
    ]

    cost_related = float(
        cond_force_fallback.compress(parent_model_piecewise, child_models_related)[
            0
        ].item()
    )
    cost_permuted = float(
        cond_force_fallback.compress(parent_model_piecewise, child_models_permuted)[
            0
        ].item()
    )
    print(
        "Test 2 | "
        f"correct_order={cost_related:.3f} "
        f"permuted_order={cost_permuted:.3f}"
    )
    assert cost_related < cost_permuted

    # Test 3: tiling validation should fail for incorrect child count.
    try:
        _ = cond.compress(parent_model_piecewise, child_models_related[:-1])
        raise AssertionError(
            "Expected ValueError for incorrect number of child blocks."
        )
    except ValueError:
        pass
    print("Test 3 | invalid child count correctly raises ValueError")

    # Test 4: shared global coordinates are reused after frequency encoding.
    shared_parent_frame = CPPNCoordinateFrame(
        origin=(2, 2),
        spatial_shape=(4, 4),
        full_spatial_shape=(8, 8),
    )
    shared_child_frames = [
        CPPNCoordinateFrame(
            origin=origin,
            spatial_shape=(2, 2),
            full_spatial_shape=(8, 8),
        )
        for origin in [(2, 2), (2, 4), (4, 2), (4, 4)]
    ]
    encoded_dim = 2 * (1 + 2 * 2)
    shared_linear = _ToyLinearCPPN(
        weight=torch.tensor(
            [
                [3.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        bias=torch.tensor([0.0, 0.0]),
    )
    shared_children = []
    for child_frame in shared_child_frames:
        component = copy.deepcopy(shared_linear)
        for parameter in component.parameters():
            parameter.requires_grad_(False)
        shared_children.append(([component], (2, 2, 1), child_frame))
    shared_parent: CPPNModel = ([shared_linear], (4, 4, 1), shared_parent_frame)
    shared_children_models: list[CPPNModel] = shared_children

    encoding_spec = cond._infer_coordinate_encoding(
        num_spatial_dims=2, input_dim=encoded_dim
    )
    encoded_teacher_coords, encoded_student_coords, _ = cond._build_encoded_routing(
        parent_spatial_shape=(4, 4),
        block_spatial_shape=(2, 2),
        encoding_spec=encoding_spec,
        parent_frame=shared_parent_frame,
    )
    assert encoded_teacher_coords.data_ptr() == encoded_student_coords.data_ptr()
    shared_cost = float(cond.compress(shared_parent, shared_children_models)[0].item())
    print(f"Test 4 | shared-coordinate encoded routed cost={shared_cost:.6f}")
    assert abs(shared_cost) < 1e-5

    # Test 5: shared-coordinate metadata must be consistent across a join.
    try:
        _ = cond.compress(shared_parent, child_models_related)
        raise AssertionError("Expected ValueError for mixed coordinate metadata.")
    except ValueError:
        pass
    misaligned_children = list(shared_children_models)
    misaligned_children[0] = (
        misaligned_children[0][0],
        misaligned_children[0][1],
        CPPNCoordinateFrame(
            origin=(2, 3),
            spatial_shape=(2, 2),
            full_spatial_shape=(8, 8),
        ),
    )
    try:
        _ = cond.compress(shared_parent, misaligned_children)
        raise AssertionError("Expected ValueError for misaligned coordinate metadata.")
    except ValueError:
        pass
    print("Test 5 | inconsistent shared-coordinate metadata correctly raises ValueError")

    # Test 6: smoke test on real two-part CPPN outputs.
    from two_part_codes.cppn import CPPNMLPTwoPartCode

    two_part = CPPNMLPTwoPartCode(
        vocab_size=4,
        hidden_dim=16,
        num_hidden_layers=2,
        activation="tanh",
        use_positional_encoding=True,
        positional_encoding_bands=2,
        include_raw_coordinates=True,
        validation_fraction=0.2,
        validation_mode="block",
        min_validation_points=4,
        max_steps=30,
        min_steps=5,
        eval_every=5,
        patience_evals=3,
        lr=4e-3,
        weight_decay=1e-4,
        min_relative_improvement_over_constant=0.01,
        min_absolute_improvement_bits=0.01,
        device="cpu",
        seed=23,
        store_models_on_cpu=True,
    )
    random_data = torch.randint(0, 4, (1, *parent_spatial_shape, 1), dtype=torch.long)
    parent_model_real, _, _ = two_part.compress(random_data)
    child_models_real = [
        two_part.compress(block)[0]
        for block in _split_blocks(random_data, block_spatial_shape)
    ]
    fit_real = cond.fit_with_logs(parent_model_real, child_models_real)[0]
    print(
        "Test 6 | "
        f"total={fit_real.total_conditional_complexity:.3f} "
        f"model={fit_real.model_complexity:.3f} "
        f"resid={fit_real.residual_complexity:.3f}"
    )
    assert math.isfinite(fit_real.total_conditional_complexity)
    assert fit_real.total_conditional_complexity >= 0.0
    total_from_parts = (
        fit_real.summary["model_complexity"] + fit_real.summary["residual_complexity"]
    )
    assert (
        abs(total_from_parts - fit_real.summary["total_conditional_complexity"]) < 1e-5
    )

    # Test 7: hierarchical rewards preserve shared-coordinate crop metadata.
    from decompositions.hierarchical import HierarchicalDecomposition
    from rewards.decomposition_level_mean import DecompositionLevelMeanReward

    shared_two_part = CPPNMLPTwoPartCode(
        vocab_size=4,
        hidden_dim=4,
        num_hidden_layers=1,
        activation="tanh",
        use_positional_encoding=True,
        positional_encoding_bands=2,
        include_raw_coordinates=True,
        use_shared_coordinates=True,
        validation_fraction=0.0,
        max_steps=1,
        min_steps=0,
        eval_every=1,
        patience_evals=1,
        min_relative_improvement_over_constant=1e9,
        min_absolute_improvement_bits=1e9,
        device="cpu",
        seed=29,
    )
    shared_reward = DecompositionLevelMeanReward(
        two_part_code=shared_two_part,
        conditional_compressor=cond,
        decomposition=HierarchicalDecomposition(
            min_part_size=(2, 2),
            branching_factor=(2, 2),
            num_spatial_dims=2,
            num_channel_dims=1,
        ),
    )
    shared_reward_values = shared_reward(
        torch.randint(0, 4, (2, 4, 4, 1), dtype=torch.long)
    )
    print(f"Test 7 | hierarchical shared-coordinate rewards={shared_reward_values}")
    assert shared_reward_values.shape == (2,)
    assert bool(torch.isfinite(shared_reward_values).all())

    print("All spot checks passed.")


if __name__ == "__main__":
    _run_spot_checks()
