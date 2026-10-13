import copy
import math
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

CPPN_ACTIVATION_NAMES: tuple[str, ...] = (
    "identity",
    "sigmoid",
    "tanh",
    "sin",
    "gauss",
    "silu",
)


def parse_cppn_activation_names(activation: str) -> tuple[str, ...]:
    names = tuple(part.lower() for part in activation.split())
    if len(names) == 0:
        raise ValueError("activation must include at least one activation function.")

    unsupported = [name for name in names if name not in CPPN_ACTIVATION_NAMES]
    if len(unsupported) > 0:
        supported = ", ".join(CPPN_ACTIVATION_NAMES)
        raise ValueError(
            f"Unsupported activation(s)={unsupported}. Choose from: {supported}."
        )
    return names


class NEATCPPNActivation(nn.Module):
    def __init__(self, activation: str) -> None:
        super().__init__()
        self.names = parse_cppn_activation_names(activation)

    def extra_repr(self) -> str:
        return " ".join(self.names)

    @staticmethod
    def _apply_name(name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "identity":
            return x
        if name == "sigmoid":
            return torch.sigmoid(5.0 * x)
        if name == "tanh":
            return torch.tanh(2.5 * x)
        if name == "sin":
            return torch.sin(5.0 * x)
        if name == "gauss":
            return torch.exp(-5.0 * x.pow(2))
        if name == "silu":
            return x * torch.sigmoid(5.0 * x)
        raise ValueError(f"Unsupported activation={name}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_functions = len(self.names)
        if x.shape[-1] % num_functions != 0:
            raise ValueError(
                "The last activation dimension must be divisible by the number "
                f"of activation functions: dim={x.shape[-1]}, "
                f"num_functions={num_functions}."
            )
        if num_functions == 1:
            return self._apply_name(self.names[0], x)

        group_width = x.shape[-1] // num_functions
        grouped = x.reshape(*x.shape[:-1], num_functions, group_width)
        outputs = [
            self._apply_name(name, grouped[..., idx, :])
            for idx, name in enumerate(self.names)
        ]
        return torch.stack(outputs, dim=-2).reshape_as(x)


def build_activation(name: str) -> nn.Module:
    return NEATCPPNActivation(name)


def gather_batched_targets(
    targets: torch.Tensor,
    sample_indices: torch.Tensor,
    coord_indices: torch.Tensor,
) -> torch.Tensor:
    return targets[sample_indices.unsqueeze(1), coord_indices]


def iter_sample_coordinate_chunks(
    num_samples: int,
    num_coords: int,
    max_pairs: int | None,
) -> Iterator[tuple[int, int, int, int]]:
    if max_pairs is None:
        yield 0, num_samples, 0, num_coords
        return

    coord_chunk = max(1, min(num_coords, max_pairs))
    sample_chunk = max(1, min(num_samples, max_pairs // coord_chunk))
    for sample_start in range(0, num_samples, sample_chunk):
        sample_end = min(num_samples, sample_start + sample_chunk)
        for coord_start in range(0, num_coords, coord_chunk):
            coord_end = min(num_coords, coord_start + coord_chunk)
            yield sample_start, sample_end, coord_start, coord_end


class BatchedModuleStackUnavailable(RuntimeError):
    pass


def module_signature(module: nn.Module) -> tuple[Any, ...]:
    params = tuple(
        (
            name,
            tuple(int(v) for v in parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for name, parameter in module.named_parameters()
    )
    buffers = tuple(
        (name, tuple(int(v) for v in buffer.shape), str(buffer.dtype))
        for name, buffer in module.named_buffers()
    )
    return (type(module), repr(module), params, buffers)


class BatchedModuleGroup:
    def __init__(
        self,
        modules: list[nn.Module],
        sample_indices: list[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if len(modules) == 0:
            raise ValueError("Cannot build an empty batched module group.")
        if len(modules) != len(sample_indices):
            raise ValueError("modules and sample_indices must have the same length.")

        copied_modules = [
            copy.deepcopy(module).to(device=device, dtype=dtype) for module in modules
        ]
        try:
            params, buffers = torch.func.stack_module_state(copied_modules)
        except RuntimeError as exc:
            raise BatchedModuleStackUnavailable(
                "Modules in this group cannot be stacked for batched fitting."
            ) from exc

        self.base_module = copy.deepcopy(copied_modules[0]).to(
            device=device, dtype=dtype
        )
        self.params = params
        self.buffers = buffers
        self.initial_params = {
            name: value.detach().clone() for name, value in self.params.items()
        }
        self.initial_buffers = {
            name: value.detach().clone() for name, value in self.buffers.items()
        }
        self.best_params = {
            name: torch.empty_like(value.detach())
            for name, value in self.params.items()
        }
        self.best_buffers = {
            name: torch.empty_like(value.detach())
            for name, value in self.buffers.items()
        }
        self.sample_indices = torch.tensor(
            sample_indices, device=device, dtype=torch.long
        )
        max_sample_idx = max(int(sample_idx) for sample_idx in sample_indices)
        self.sample_to_row_tensor = torch.full(
            (max_sample_idx + 1,), -1, device=device, dtype=torch.long
        )
        self.sample_to_row_tensor[self.sample_indices] = torch.arange(
            len(sample_indices), device=device, dtype=torch.long
        )
        self.sample_to_row = {
            int(sample_idx): row for row, sample_idx in enumerate(sample_indices)
        }
        self.device = device
        self.has_trainable_parameters = any(
            value.requires_grad for value in self.params.values()
        )
        self._named_trainable_parameters = [
            (name, value) for name, value in self.params.items() if value.requires_grad
        ]
        self.adamw_state = init_batched_adamw_state(self._named_trainable_parameters)

    def train(self, mode: bool = True) -> None:
        self.base_module.train(mode)

    def named_trainable_parameters(self) -> list[tuple[str, torch.Tensor]]:
        return self._named_trainable_parameters

    def zero_grad(self) -> None:
        for parameter in self.params.values():
            parameter.grad = None

    def rows_for_sample_indices(
        self, sample_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_indices.numel() == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty, empty

        sample_indices = sample_indices.to(device=self.device, dtype=torch.long)
        in_range = (sample_indices >= 0) & (
            sample_indices < int(self.sample_to_row_tensor.numel())
        )
        safe_indices = torch.where(
            in_range, sample_indices, torch.zeros_like(sample_indices)
        )
        row_values = self.sample_to_row_tensor.index_select(0, safe_indices)
        matched_mask = in_range & (row_values >= 0)
        positions = torch.nonzero(matched_mask, as_tuple=False).squeeze(-1)
        rows = row_values.index_select(0, positions)
        matched_samples = sample_indices.index_select(0, positions)
        return rows, positions, matched_samples

    def row_for_sample(self, sample_idx: int) -> int:
        return self.sample_to_row[int(sample_idx)]

    def _select_state(
        self,
        rows: torch.Tensor,
        *,
        use_initial: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        source_params = self.initial_params if use_initial else self.params
        source_buffers = self.initial_buffers if use_initial else self.buffers
        selected_params = {
            name: value.index_select(0, rows) for name, value in source_params.items()
        }
        selected_buffers = {
            name: value.index_select(0, rows) for name, value in source_buffers.items()
        }
        return selected_params, selected_buffers

    def forward_rows(
        self,
        rows: torch.Tensor,
        coords: torch.Tensor,
        *,
        use_initial: bool,
    ) -> torch.Tensor:
        selected_params, selected_buffers = self._select_state(
            rows, use_initial=use_initial
        )

        def _call(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            single_coords: torch.Tensor,
        ) -> torch.Tensor:
            return torch.func.functional_call(
                self.base_module,
                (params, buffers),
                (single_coords,),
            )

        return torch.func.vmap(_call, randomness="different")(
            selected_params,
            selected_buffers,
            coords,
        )

    def forward(
        self,
        sample_indices: torch.Tensor,
        coords: torch.Tensor,
        *,
        use_initial: bool,
    ) -> torch.Tensor:
        rows, _, _ = self.rows_for_sample_indices(sample_indices)
        if rows.numel() != sample_indices.numel():
            raise ValueError("All requested samples must belong to the module group.")
        return self.forward_rows(rows, coords, use_initial=use_initial)

    def save_best(self, sample_indices: torch.Tensor) -> None:
        rows, _, _ = self.rows_for_sample_indices(sample_indices)
        if rows.numel() == 0:
            return
        with torch.no_grad():
            for name, value in self.params.items():
                self.best_params[name][rows] = value.detach().index_select(0, rows)
            for name, value in self.buffers.items():
                self.best_buffers[name][rows] = value.detach().index_select(0, rows)

    def restore_best(self, sample_indices: torch.Tensor) -> None:
        rows, _, _ = self.rows_for_sample_indices(sample_indices)
        if rows.numel() == 0:
            return
        with torch.no_grad():
            for name, value in self.params.items():
                value[rows] = self.best_params[name].index_select(0, rows)
            for name, value in self.buffers.items():
                value[rows] = self.best_buffers[name].index_select(0, rows)

    def materialize(
        self,
        sample_idx: int,
        *,
        use_initial: bool,
        target_device: torch.device,
    ) -> nn.Module:
        row = self.row_for_sample(sample_idx)
        source_params = self.initial_params if use_initial else self.params
        source_buffers = self.initial_buffers if use_initial else self.buffers

        module = copy.deepcopy(self.base_module).to(target_device)
        state = {}
        for name, value in source_params.items():
            state[name] = value[row].detach().to(target_device).clone()
        for name, value in source_buffers.items():
            state[name] = value[row].detach().to(target_device).clone()
        state_dict_keys = set(module.state_dict().keys())
        load_state = {
            name: value for name, value in state.items() if name in state_dict_keys
        }
        module.load_state_dict(load_state, strict=False)
        module.eval()
        return module


def build_batched_module_groups(
    modules: list[nn.Module],
    device: torch.device,
    dtype: torch.dtype,
) -> list[BatchedModuleGroup]:
    buckets: dict[tuple[Any, ...], tuple[list[int], list[nn.Module]]] = {}
    for sample_idx, module in enumerate(modules):
        signature = module_signature(module)
        if signature not in buckets:
            buckets[signature] = ([], [])
        buckets[signature][0].append(sample_idx)
        buckets[signature][1].append(module)
    return [
        BatchedModuleGroup(
            grouped_modules,
            sample_indices,
            device=device,
            dtype=dtype,
        )
        for sample_indices, grouped_modules in buckets.values()
    ]


def flatten_batched_module_groups(
    grouped: list[list[BatchedModuleGroup]],
) -> list[BatchedModuleGroup]:
    return [group for groups in grouped for group in groups]


def set_batched_module_group_mode(
    groups: list[BatchedModuleGroup],
    training: bool,
) -> None:
    for group in groups:
        group.train(training)


def zero_batched_module_group_grads(groups: list[BatchedModuleGroup]) -> None:
    for group in groups:
        group.zero_grad()


def find_batched_module_group_for_sample(
    groups: list[BatchedModuleGroup],
    sample_idx: int,
) -> BatchedModuleGroup:
    for group in groups:
        if int(sample_idx) in group.sample_to_row:
            return group
    raise ValueError(f"No component group contains sample {sample_idx}.")


def save_best_batched_module_group_state(
    groups: list[BatchedModuleGroup],
    sample_indices: torch.Tensor,
) -> None:
    for group in groups:
        group.save_best(sample_indices)


def restore_best_batched_module_group_state(
    groups: list[BatchedModuleGroup],
    sample_indices: torch.Tensor,
) -> None:
    for group in groups:
        group.restore_best(sample_indices)


def clip_batched_module_group_grad_norm_(
    groups: list[BatchedModuleGroup],
    active_indices: torch.Tensor,
    max_norm: float,
) -> None:
    if active_indices.numel() == 0:
        return

    total_sq = torch.zeros(
        active_indices.numel(), device=active_indices.device, dtype=torch.float32
    )

    for group in groups:
        if not group.has_trainable_parameters:
            continue
        rows, positions, _ = group.rows_for_sample_indices(active_indices)
        if rows.numel() == 0:
            continue
        for _, parameter in group.named_trainable_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.index_select(0, rows).detach().to(torch.float32)
            total_sq[positions] += grad.reshape(rows.numel(), -1).pow(2).sum(dim=1)

    clip_coef = (float(max_norm) / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for group in groups:
        if not group.has_trainable_parameters:
            continue
        rows, positions, _ = group.rows_for_sample_indices(active_indices)
        if rows.numel() == 0:
            continue
        for _, parameter in group.named_trainable_parameters():
            if parameter.grad is None:
                continue
            view_shape = [rows.numel()] + [1] * (parameter.grad.ndim - 1)
            clipped = parameter.grad.index_select(0, rows) * clip_coef.index_select(
                0, positions
            ).to(parameter.grad.dtype).view(view_shape)
            parameter.grad[rows] = clipped


def adamw_step_batched_module_groups_(
    groups: list[BatchedModuleGroup],
    active_indices: torch.Tensor,
    lr_values: torch.Tensor,
    step_counts: torch.Tensor,
    weight_decay: float,
) -> None:
    for group in groups:
        if not group.has_trainable_parameters:
            continue
        rows, _, matched_samples = group.rows_for_sample_indices(active_indices)
        if rows.numel() == 0:
            continue
        adamw_step_batched_(
            named_parameters=group.named_trainable_parameters(),
            state=group.adamw_state,
            active_indices=rows,
            lr_values=lr_values.index_select(0, matched_samples),
            step_counts=step_counts.index_select(0, matched_samples),
            weight_decay=weight_decay,
        )


class BatchedCoordinateMLP(nn.Module):
    def __init__(
        self,
        batch_size: int,
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
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout
        self.activation_names = parse_cppn_activation_names(activation)
        self.layer_dim = hidden_dim * len(self.activation_names)
        self.activation = build_activation(activation)

        self.input_weight = nn.Parameter(
            torch.empty(batch_size, self.layer_dim, input_dim)
        )
        self.input_bias = nn.Parameter(torch.empty(batch_size, self.layer_dim))
        self.hidden_weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(batch_size, self.layer_dim, self.layer_dim))
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.hidden_biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(batch_size, self.layer_dim))
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.output_weight = nn.Parameter(
            torch.empty(batch_size, num_channels * vocab_size, self.layer_dim)
        )
        self.output_bias = nn.Parameter(
            torch.empty(batch_size, num_channels * vocab_size)
        )
        self.reset_parameters()

    @staticmethod
    def _init_linear_weight_and_bias(
        weight: torch.Tensor,
        bias: torch.Tensor,
        fan_in: int,
    ) -> None:
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(weight, -bound, bound)
        nn.init.uniform_(bias, -bound, bound)

    def reset_parameters(self) -> None:
        self._init_linear_weight_and_bias(
            self.input_weight, self.input_bias, self.input_dim
        )
        for weight, bias in zip(self.hidden_weights, self.hidden_biases):
            self._init_linear_weight_and_bias(weight, bias, self.layer_dim)
        self._init_linear_weight_and_bias(
            self.output_weight, self.output_bias, self.layer_dim
        )

    def batched_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        params: list[tuple[str, nn.Parameter]] = [
            ("input_weight", self.input_weight),
            ("input_bias", self.input_bias),
        ]
        for i, (weight, bias) in enumerate(
            zip(self.hidden_weights, self.hidden_biases)
        ):
            params.append((f"hidden_weight_{i}", weight))
            params.append((f"hidden_bias_{i}", bias))
        params.extend(
            [
                ("output_weight", self.output_weight),
                ("output_bias", self.output_bias),
            ]
        )
        return params

    @staticmethod
    def _linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        selected_weight = weight.index_select(0, sample_indices)
        selected_bias = bias.index_select(0, sample_indices)
        return torch.bmm(x, selected_weight.transpose(1, 2)) + selected_bias.unsqueeze(
            1
        )

    def forward(
        self,
        sample_indices: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if coords.ndim == 2:
            coords = coords.unsqueeze(0).expand(sample_indices.numel(), -1, -1)
        if coords.ndim != 3:
            raise ValueError(
                "Batched CPPN coordinates must have shape (batch, points, features)."
            )
        if coords.shape[0] != sample_indices.numel():
            raise ValueError(
                "The coordinate batch dimension must match sample_indices length."
            )

        x = self._linear(coords, self.input_weight, self.input_bias, sample_indices)
        x = F.dropout(
            self.activation(x),
            p=self.dropout_p,
            training=self.training and self.dropout_p > 0.0,
        )
        for weight, bias in zip(self.hidden_weights, self.hidden_biases):
            x = self._linear(x, weight, bias, sample_indices)
            x = F.dropout(
                self.activation(x),
                p=self.dropout_p,
                training=self.training and self.dropout_p > 0.0,
            )
        logits = self._linear(x, self.output_weight, self.output_bias, sample_indices)
        return logits.view(
            sample_indices.numel(), coords.shape[1], self.num_channels, self.vocab_size
        )


def clip_batched_grad_norm_(
    named_parameters: list[tuple[str, nn.Parameter]],
    active_indices: torch.Tensor,
    max_norm: float,
) -> None:
    if active_indices.numel() == 0:
        return

    total_sq = torch.zeros(
        active_indices.numel(), device=active_indices.device, dtype=torch.float32
    )
    for _, parameter in named_parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.index_select(0, active_indices).detach()
        total_sq += (
            grad.to(torch.float32).reshape(active_indices.numel(), -1).pow(2).sum(dim=1)
        )

    clip_coef = (float(max_norm) / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for _, parameter in named_parameters:
        if parameter.grad is None:
            continue
        view_shape = [active_indices.numel()] + [1] * (parameter.grad.ndim - 1)
        clipped = parameter.grad.index_select(0, active_indices) * clip_coef.to(
            parameter.grad.dtype
        ).view(view_shape)
        parameter.grad[active_indices] = clipped


def init_batched_adamw_state(
    named_parameters: list[tuple[str, nn.Parameter]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        for name, parameter in named_parameters
    }


def adamw_step_batched_(
    named_parameters: list[tuple[str, nn.Parameter]],
    state: dict[str, dict[str, torch.Tensor]],
    active_indices: torch.Tensor,
    lr_values: torch.Tensor,
    step_counts: torch.Tensor,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    if active_indices.numel() == 0:
        return

    step_float = step_counts.to(device=active_indices.device, dtype=torch.float32)
    beta1_tensor = torch.tensor(beta1, device=active_indices.device)
    beta2_tensor = torch.tensor(beta2, device=active_indices.device)
    bias_correction1 = 1.0 - torch.pow(beta1_tensor, step_float)
    bias_correction2_sqrt = torch.sqrt(1.0 - torch.pow(beta2_tensor, step_float))

    with torch.no_grad():
        for name, parameter in named_parameters:
            if parameter.grad is None:
                continue

            param_slice = parameter.detach().index_select(0, active_indices)
            grad_slice = parameter.grad.detach().index_select(0, active_indices)
            exp_avg = state[name]["exp_avg"].index_select(0, active_indices)
            exp_avg_sq = state[name]["exp_avg_sq"].index_select(0, active_indices)

            exp_avg.mul_(beta1).add_(grad_slice, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_slice, grad_slice, value=1.0 - beta2)

            view_shape = [active_indices.numel()] + [1] * (parameter.ndim - 1)
            lr = lr_values.to(parameter.dtype).view(view_shape)
            bc1 = bias_correction1.to(parameter.dtype).view(view_shape)
            bc2_sqrt = bias_correction2_sqrt.to(parameter.dtype).view(view_shape)

            if weight_decay != 0.0:
                param_slice = param_slice * (1.0 - lr * weight_decay)
            denom = exp_avg_sq.sqrt().div(bc2_sqrt).add_(eps)
            param_slice = param_slice - (lr / bc1) * exp_avg / denom

            parameter[active_indices] = param_slice
            state[name]["exp_avg"][active_indices] = exp_avg
            state[name]["exp_avg_sq"][active_indices] = exp_avg_sq
