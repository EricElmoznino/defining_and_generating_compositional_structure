import glob
import math
import multiprocessing
import os
import random
import re
import tempfile
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import neat
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb

from lib.cppn_genome_metrics import summarize_cppn_genomes
from lib.helpers import positional_encode
from lib.image_logging import (
    ImageFileType,
    cleanup_temp_files,
    discrete_object_to_wandb_image,
)
from rewards.base import Reward
from rewards.decomposition_level_mean import DecompositionLevelMeanReward

OutputEncoding = Literal["quantized_scalar", "categorical_logits"]
HIGH_RESOLUTION_LOGGED_IMAGE_SHAPE: tuple[int, int] = (1024, 1024)
CPPN_STATISTICS_PREFIX = "CPPN Statistics/"


@dataclass
class _CPPNRenderContext:
    neat_config: neat.Config
    coordinate_inputs: list[list[float]]
    spatial_shape: tuple[int, ...]
    n_channels: int
    vocab_size: int
    output_encoding: OutputEncoding
    n_cppn_outputs: int


_WORKER_RENDER_CONTEXT: _CPPNRenderContext | None = None


class _TanhOutputGenome(neat.DefaultGenome):
    """Default NEAT genome with output node activations fixed to tanh."""

    def configure_new(self, config: Any) -> None:
        super().configure_new(config)
        self._force_output_activations(config)

    def configure_crossover(
        self,
        genome1: neat.DefaultGenome,
        genome2: neat.DefaultGenome,
        config: Any,
    ) -> None:
        super().configure_crossover(genome1, genome2, config)
        self._force_output_activations(config)

    def mutate(self, config: Any) -> None:
        super().mutate(config)
        self._force_output_activations(config)

    def _force_output_activations(self, config: Any) -> None:
        for node_key in config.output_keys:
            self.nodes[node_key].activation = "tanh"


@dataclass
class CPPNEvolutionResult:
    """Final result of CPPN evolution."""

    winner_genome: neat.DefaultGenome
    winner_reward: float
    winner_object: np.ndarray


class EvolutionaryOptimizationCPPN:
    """Evolve discrete objects with NEAT over CPPN genotypes."""

    def __init__(
        self,
        reward: Reward,
        vocab_size: int,
        spatial_shape: tuple[int, ...],
        n_channels: int,
        pop_size: int = 150,
        output_encoding: OutputEncoding = "quantized_scalar",
        use_positional_encoding: bool = False,
        positional_encoding_bands: int = 4,
        include_raw_coordinates: bool = True,
        n_workers: int = 1,
        reward_batch_size: int = 64,
        reward_device: str = "auto",
        n_logged_examples: int = 20,
        log_examples_every_n_generations: int = 25,
        neat_verbose: bool = True,
        neat_config_overrides: dict[str, dict[str, Any]] | None = None,
        checkpoint_root: str | None = None,
        checkpoint_interval_generations: int | None = 1,
        checkpoint_interval_seconds: float | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        """Initialize CPPN+NEAT optimization.

        Args:
            reward (Reward): Black-box reward function on batched discrete objects.
            vocab_size (int): Number of discrete symbols per channel.
            spatial_shape (tuple[int]): Spatial object shape (without channels).
            n_channels (int): Number of channels per location.
            pop_size (int, optional): NEAT population size.
            output_encoding (OutputEncoding, optional): Mapping from continuous CPPN output to discrete symbols.
            use_positional_encoding (bool, optional): Whether to apply NeRF-style
                sinusoidal coordinate encoding to spatial CPPN inputs.
            positional_encoding_bands (int, optional): Number of frequency bands
                used when positional encoding is enabled.
            include_raw_coordinates (bool, optional): If True, concatenate raw
                normalized coordinates with sinusoidal features.
            n_workers (int, optional): CPU workers for CPPN rendering (`-1` uses all CPUs).
            reward_batch_size (int, optional): Batch size passed to reward.
            reward_device (str, optional): Device for reward evaluation (`auto`, `cpu`, `cuda`, ...).
            n_logged_examples (int, optional): Number of top samples to log when logging examples.
            log_examples_every_n_generations (int, optional): Log example table every N generations.
            neat_verbose (bool, optional): Whether to add NEAT stdout reporter.
            neat_config_overrides (dict[str, dict[str, Any]] | None, optional): Section-wise NEAT config overrides.
            checkpoint_root (str | None, optional): Root directory for NEAT checkpoints.
            checkpoint_interval_generations (int | None, optional): Save every N
                generations when checkpointing is enabled.
            checkpoint_interval_seconds (float | None, optional): Also save after
                this many seconds when checkpointing is enabled.
            resume_run_id (str | None, optional): Checkpoint namespace to restore
                from, usually a W&B run id.
        """
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1.")
        if n_channels <= 0:
            raise ValueError("n_channels must be > 0.")
        if pop_size <= 0:
            raise ValueError("pop_size must be > 0.")
        if reward_batch_size <= 0:
            raise ValueError("reward_batch_size must be > 0.")
        if n_workers != -1 and n_workers <= 0:
            raise ValueError("n_workers must be -1 (all CPUs) or > 0.")
        if len(spatial_shape) == 0:
            raise ValueError("spatial_shape must have at least one dimension.")
        if any(v <= 0 for v in spatial_shape):
            raise ValueError("All spatial dimensions must be > 0.")
        if (
            checkpoint_interval_generations is not None
            and checkpoint_interval_generations <= 0
        ):
            raise ValueError("checkpoint_interval_generations must be None or > 0.")
        if checkpoint_interval_seconds is not None and checkpoint_interval_seconds <= 0:
            raise ValueError("checkpoint_interval_seconds must be None or > 0.")

        if n_workers == -1:
            n_workers = multiprocessing.cpu_count()

        self.reward = reward
        self.vocab_size = vocab_size
        self.spatial_shape = spatial_shape
        self.n_channels = n_channels
        self.pop_size = pop_size
        self.output_encoding = output_encoding
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_bands = positional_encoding_bands
        self.include_raw_coordinates = include_raw_coordinates
        self.n_workers = n_workers
        self.reward_batch_size = reward_batch_size
        self.n_logged_examples = n_logged_examples
        self.log_examples_every_n_generations = log_examples_every_n_generations
        self.neat_verbose = neat_verbose
        self.neat_config_overrides = neat_config_overrides or {}
        self.checkpoint_root = checkpoint_root
        self.checkpoint_interval_generations = checkpoint_interval_generations
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.resume_run_id = self._none_if_empty(resume_run_id)

        self.reward_device = self._resolve_device(reward_device)
        self.reward.to(self.reward_device)

        self.n_cppn_outputs = self._num_cppn_outputs()
        self.coordinates = self._build_coordinate_grid()
        self.n_cppn_inputs = int(self.coordinates.shape[1])
        self.coordinate_inputs: list[list[float]] = self.coordinates.tolist()

        if len(self.spatial_shape) > 2:
            print(
                "Warning: spatial_shape has more than two dimensions. "
                "CPPN spatial inputs are only (x, y), so extra dimensions share the same 2D coordinate field."
            )

        if isinstance(reward, DecompositionLevelMeanReward):
            self.using_decomposition_level_mean_reward = True
        else:
            self.using_decomposition_level_mean_reward = False

    def optimize(
        self,
        n_generations: int,
        logger: wandb.Run | None = None,
        seed: int = 0,
    ) -> CPPNEvolutionResult:
        """Run NEAT optimization and return the winner genome and decoded object."""
        if n_generations <= 0:
            raise ValueError("n_generations must be > 0.")

        self._set_seed(seed)
        neat_config = self._build_neat_config(seed=seed)

        checkpoint_run_id = self._resolve_checkpoint_run_id(logger)
        checkpoint_dir = self._checkpoint_dir(checkpoint_run_id)
        latest_checkpoint = (
            self._latest_checkpoint_path(checkpoint_dir)
            if checkpoint_dir is not None
            else None
        )
        if latest_checkpoint is None:
            population = neat.Population(neat_config, seed=seed)
        else:
            population = neat.Checkpointer.restore_checkpoint(
                latest_checkpoint,
                new_config=neat_config,
            )
            print(
                "Restored NEAT checkpoint "
                f"{latest_checkpoint} at generation {population.generation}."
            )

        if self.neat_verbose:
            population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            population.add_reporter(
                neat.Checkpointer(
                    generation_interval=self.checkpoint_interval_generations,
                    time_interval_seconds=self.checkpoint_interval_seconds,
                    filename_prefix=self._checkpoint_prefix(checkpoint_dir),
                )
            )

        render_pool: Pool | None = None
        if self.n_workers > 1:
            render_pool = multiprocessing.get_context("spawn").Pool(
                self.n_workers,
                initializer=_init_render_worker,
                initargs=(
                    _CPPNRenderContext(
                        neat_config=neat_config,
                        coordinate_inputs=self.coordinate_inputs,
                        spatial_shape=self.spatial_shape,
                        n_channels=self.n_channels,
                        vocab_size=self.vocab_size,
                        output_encoding=self.output_encoding,
                        n_cppn_outputs=self.n_cppn_outputs,
                    ),
                ),
            )

        generation_index = population.generation

        def eval_genomes(genomes, config):
            nonlocal generation_index

            objects = self._render_population(
                genomes=genomes,
                neat_config=config,
                render_pool=render_pool,
            )
            rewards = self._evaluate_rewards(objects)

            for (_, genome), reward_value in zip(genomes, rewards):
                genome.fitness = float(reward_value)

            if logger is not None:
                self.log_generation(
                    logger=logger,
                    generation=generation_index + 1,
                    rewards=rewards,
                    genomes=[genome for _, genome in genomes],
                    species_ids=[
                        population.species.get_species_id(genome_id)
                        for genome_id, _ in genomes
                    ],
                    objects=objects,
                    neat_config=config,
                )

            generation_index += 1

        try:
            winner = population.run(eval_genomes, n_generations)
            winner_object = _render_cppn_genome(
                genome=winner,
                neat_config=neat_config,
                coordinate_inputs=self.coordinate_inputs,
                spatial_shape=self.spatial_shape,
                n_channels=self.n_channels,
                vocab_size=self.vocab_size,
                output_encoding=self.output_encoding,
                n_cppn_outputs=self.n_cppn_outputs,
            )
            winner_reward = (
                float(winner.fitness)
                if winner.fitness is not None
                else float(self._evaluate_rewards(winner_object[None])[0])
            )
        finally:
            if render_pool is not None:
                render_pool.close()
                render_pool.join()
            if logger is not None:
                wandb.finish()

        return CPPNEvolutionResult(
            winner_genome=winner,
            winner_reward=winner_reward,
            winner_object=winner_object,
        )

    def log_generation(
        self,
        logger: wandb.Run,
        generation: int,
        rewards: np.ndarray,
        genomes: list[neat.DefaultGenome],
        species_ids: Sequence[int],
        objects: np.ndarray,
        neat_config: neat.Config,
    ) -> None:
        """Log one generation of metrics and optional top examples to W&B."""
        metrics: dict[str, Any] = {
            "generation": generation,
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "std_reward": float(np.std(rewards)),
            "reward_histogram": self.reward_histogram(rewards),
        }
        metrics.update(
            {
                f"{CPPN_STATISTICS_PREFIX}{name}": value
                for name, value in summarize_cppn_genomes(
                    genomes=genomes,
                    input_keys=neat_config.genome_config.input_keys,
                    output_keys=neat_config.genome_config.output_keys,
                ).items()
            }
        )
        temp_image_paths: list[str] = []

        if generation % self.log_examples_every_n_generations == 1:
            best_indices = self._diverse_top_reward_indices(
                rewards=rewards,
                species_ids=species_ids,
                n=self.n_logged_examples,
            )
            best_objects = objects[best_indices]
            best_rewards = rewards[best_indices]
            best_genomes = [genomes[int(i)] for i in best_indices]
            examples, example_image_paths = self.log_examples(
                objects=best_objects,
                rewards=best_rewards,
                genomes=best_genomes,
                neat_config=neat_config,
            )
            temp_image_paths.extend(example_image_paths)
            metrics["examples"] = examples
            metrics.update(self.reward_specific_metrics(best_objects))

        try:
            logger.log(metrics, step=generation)
        finally:
            cleanup_temp_files(temp_image_paths)

    def log_examples(
        self,
        objects: np.ndarray,
        rewards: np.ndarray,
        genomes: list[neat.DefaultGenome],
        neat_config: neat.Config,
    ) -> tuple[wandb.Table, list[str]]:
        """Build a table with top-reward objects."""
        if len(objects) != len(genomes):
            raise ValueError("objects and genomes must have the same length.")

        data: dict[str, list[Any]] = {
            "reward": rewards.tolist(),
            "shape": [str(tuple(obj.shape)) for obj in objects],
        }
        temp_image_paths: list[str] = []

        if self._is_image_logging_compatible():
            images: list[wandb.Image] = []
            high_resolution_images: list[wandb.Image] = []
            high_resolution_coordinates = self._build_coordinate_grid_for_shape(
                HIGH_RESOLUTION_LOGGED_IMAGE_SHAPE
            )
            for obj, genome in zip(objects, genomes):
                image, image_path = self._object_to_wandb_image(obj, file_type="png")
                temp_image_paths.append(image_path)
                images.append(image)

                high_resolution_obj = self._render_genome_with_coordinates(
                    genome=genome,
                    neat_config=neat_config,
                    coordinates=high_resolution_coordinates,
                    spatial_shape=HIGH_RESOLUTION_LOGGED_IMAGE_SHAPE,
                )
                high_resolution_image, high_resolution_image_path = (
                    self._object_to_wandb_image(high_resolution_obj, file_type="jpg")
                )
                temp_image_paths.append(high_resolution_image_path)
                high_resolution_images.append(high_resolution_image)

            data["image"] = images
            data["image_highres"] = high_resolution_images

        return (
            wandb.Table(data=list(zip(*data.values())), columns=list(data.keys())),
            temp_image_paths,
        )

    @staticmethod
    def _diverse_top_reward_indices(
        rewards: np.ndarray,
        species_ids: Sequence[int],
        n: int,
    ) -> list[int]:
        if len(rewards) != len(species_ids):
            raise ValueError("rewards and species_ids must have the same length.")
        if n <= 0:
            return []

        ranked_indices = [int(i) for i in np.argsort(-rewards)]
        selected: list[int] = []
        selected_indices: set[int] = set()
        selected_species: set[int] = set()

        for index in ranked_indices:
            species_id = int(species_ids[index])
            if species_id in selected_species:
                continue
            selected.append(index)
            selected_indices.add(index)
            selected_species.add(species_id)
            if len(selected) == n:
                return selected

        for index in ranked_indices:
            if index in selected_indices:
                continue
            selected.append(index)
            if len(selected) == n:
                break

        return selected

    @staticmethod
    def reward_histogram(rewards: np.ndarray) -> wandb.Histogram:
        """Create a log-scale histogram for positive rewards."""
        rewards = rewards.clip(min=1e-8)
        _, bins = np.histogram(rewards, bins=30)
        bins = bins.clip(min=1e-8)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        loghist = np.histogram(rewards, bins=logbins)
        return wandb.Histogram(np_histogram=loghist)

    def reward_specific_metrics(self, objects: np.ndarray) -> dict[str, Any]:
        if self.using_decomposition_level_mean_reward:
            level_reward_func: DecompositionLevelMeanReward = self.reward
            _, reward_compositions = level_reward_func(
                torch.tensor(objects), return_compositions=True
            )
            level_rewards = level_reward_func.compositions_to_level_rewards(
                reward_compositions, obj_shape=objects.shape[1:]
            )
            num_levels = level_rewards.shape[1]
            level_rewards = pd.DataFrame(
                [
                    {"level": level, "reward": level_rewards[i, level].item()}
                    for i in range(level_rewards.shape[0])
                    for level in range(num_levels)
                ]
            )
            fig, ax = plt.subplots(figsize=(num_levels, 4))
            sns.boxplot(
                data=level_rewards,
                x="level",
                y="reward",
                ax=ax,
            )
            ax.set_title("Level rewards distribution")
            fig.tight_layout()
            return {"decomposition_rewards": wandb.Image(fig)}

        return {}

    def _render_population(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
        render_pool: Pool | None,
    ) -> np.ndarray:
        if render_pool is None:
            rendered = [
                _render_cppn_genome(
                    genome=genome,
                    neat_config=neat_config,
                    coordinate_inputs=self.coordinate_inputs,
                    spatial_shape=self.spatial_shape,
                    n_channels=self.n_channels,
                    vocab_size=self.vocab_size,
                    output_encoding=self.output_encoding,
                    n_cppn_outputs=self.n_cppn_outputs,
                )
                for _, genome in genomes
            ]
        else:
            rendered = render_pool.map(
                _render_cppn_genome_worker, [genome for _, genome in genomes]
            )

        return np.stack(rendered, axis=0).astype(np.int64)

    def _render_genome_with_coordinates(
        self,
        genome: neat.DefaultGenome,
        neat_config: neat.Config,
        coordinates: np.ndarray,
        spatial_shape: tuple[int, ...],
    ) -> np.ndarray:
        return _render_cppn_genome(
            genome=genome,
            neat_config=neat_config,
            coordinate_inputs=coordinates,
            spatial_shape=spatial_shape,
            n_channels=self.n_channels,
            vocab_size=self.vocab_size,
            output_encoding=self.output_encoding,
            n_cppn_outputs=self.n_cppn_outputs,
        )

    def _evaluate_rewards(self, objects: np.ndarray) -> np.ndarray:
        rewards: list[np.ndarray] = []
        for start in range(0, objects.shape[0], self.reward_batch_size):
            end = min(start + self.reward_batch_size, objects.shape[0])
            batch = torch.from_numpy(objects[start:end]).to(self.reward_device)
            batch_rewards = self.reward(batch)
            rewards.append(batch_rewards.detach().float().cpu().numpy())
        all_rewards = np.concatenate(rewards, axis=0)
        return all_rewards

    def _build_coordinate_grid(self) -> np.ndarray:
        return self._build_coordinate_grid_for_shape(self.spatial_shape)

    def _build_coordinate_grid_for_shape(
        self, spatial_shape: tuple[int, ...]
    ) -> np.ndarray:
        if len(spatial_shape) == 1:
            x_axis = np.linspace(-1.0, 1.0, spatial_shape[0], dtype=np.float32)
            y_axis = np.zeros_like(x_axis, dtype=np.float32)
            raw_coordinates = np.stack([x_axis, y_axis], axis=1)
            return self._encode_coordinates(raw_coordinates)

        axes = [
            np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in spatial_shape
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        x_coords = mesh[-1].reshape(-1)
        y_coords = mesh[-2].reshape(-1)
        raw_coordinates = np.stack([x_coords, y_coords], axis=1)
        return self._encode_coordinates(raw_coordinates)

    def _encode_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        if not self.use_positional_encoding:
            return coordinates
        encoded = positional_encode(
            torch.from_numpy(coordinates),
            num_bands=self.positional_encoding_bands,
            include_raw=self.include_raw_coordinates,
        )
        return encoded.numpy().astype(np.float32, copy=False)

    def _num_cppn_outputs(self) -> int:
        if self.output_encoding == "quantized_scalar":
            return self.n_channels
        if self.output_encoding == "categorical_logits":
            return self.n_channels * self.vocab_size
        raise ValueError(f"Unsupported output_encoding: {self.output_encoding!r}")

    def _build_neat_config(self, seed: int) -> neat.Config:
        sections = self._default_neat_sections(seed=seed)
        self._deep_update(sections, self.neat_config_overrides)

        sections["NEAT"]["pop_size"] = self.pop_size
        sections["DefaultGenome"]["num_inputs"] = self.n_cppn_inputs
        sections["DefaultGenome"]["num_outputs"] = self.n_cppn_outputs

        genome_section_name = _TanhOutputGenome.__name__
        sections[genome_section_name] = sections.pop("DefaultGenome")

        section_order = [
            "NEAT",
            genome_section_name,
            "DefaultSpeciesSet",
            "DefaultStagnation",
            "DefaultReproduction",
        ]
        text = self._sections_to_text(sections=sections, section_order=section_order)

        with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
            f.write(text)
            config_path = f.name

        try:
            neat_config = neat.Config(
                _TanhOutputGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
            neat_config.genome_config.add_activation("silu", silu_activation)
            return neat_config
        finally:
            os.remove(config_path)

    def _default_neat_sections(self, seed: int) -> dict[str, dict[str, Any]]:
        return {
            "NEAT": {
                "fitness_criterion": "max",
                "fitness_threshold": 1e38,
                "pop_size": self.pop_size,
                "reset_on_extinction": False,
                "no_fitness_termination": False,
                "seed": seed,
            },
            "DefaultGenome": {
                "num_inputs": self.n_cppn_inputs,
                "num_hidden": 0,
                "num_outputs": self.n_cppn_outputs,
                "initial_connection": "partial_direct 0.5",
                "feed_forward": True,
                "compatibility_disjoint_coefficient": 1.0,
                "compatibility_weight_coefficient": 0.8,
                "conn_add_prob": 0.2,
                "conn_delete_prob": 0.2,
                "node_add_prob": 0.2,
                "node_delete_prob": 0.2,
                "activation_default": "sigmoid",
                "activation_options": "identity sigmoid tanh sin gauss silu",
                "activation_mutate_rate": 0.05,
                "aggregation_default": "sum",
                "aggregation_options": "sum",
                "aggregation_mutate_rate": 0.0,
                "bias_init_mean": 0.0,
                "bias_init_stdev": 1.0,
                "bias_replace_rate": 0.1,
                "bias_mutate_rate": 0.7,
                "bias_mutate_power": 0.5,
                "bias_max_value": 30.0,
                "bias_min_value": -30.0,
                "response_init_mean": 1.0,
                "response_init_stdev": 0.1,
                "response_replace_rate": 0.1,
                "response_mutate_rate": 0.1,
                "response_mutate_power": 0.1,
                "response_max_value": 30.0,
                "response_min_value": -30.0,
                "weight_max_value": 30.0,
                "weight_min_value": -30.0,
                "weight_init_mean": 0.0,
                "weight_init_stdev": 1.0,
                "weight_mutate_rate": 0.8,
                "weight_replace_rate": 0.1,
                "weight_mutate_power": 0.5,
                "enabled_default": True,
                "enabled_mutate_rate": 0.01,
                "single_structural_mutation": False,
                "structural_mutation_surer": "default",
                "bias_init_type": "gaussian",
                "response_init_type": "gaussian",
                "weight_init_type": "gaussian",
                "enabled_rate_to_true_add": 0.0,
                "enabled_rate_to_false_add": 0.0,
            },
            "DefaultSpeciesSet": {
                "compatibility_threshold": 3.0,
            },
            "DefaultStagnation": {
                "species_fitness_func": "max",
                "max_stagnation": 20,
                "species_elitism": 2,
            },
            "DefaultReproduction": {
                "elitism": 1,
                "survival_threshold": 0.2,
                "min_species_size": 2,
            },
        }

    @staticmethod
    def _sections_to_text(
        sections: dict[str, dict[str, Any]],
        section_order: list[str],
    ) -> str:
        lines: list[str] = []
        emitted = set()

        for section_name in section_order + [
            s for s in sections if s not in section_order
        ]:
            if section_name in emitted:
                continue
            emitted.add(section_name)
            section_values = sections[section_name]
            lines.append(f"[{section_name}]")
            for key, value in section_values.items():
                lines.append(
                    f"{key} = {EvolutionaryOptimizationCPPN._format_neat_value(value)}"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_neat_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
        for key, value in update.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, Mapping)
            ):
                EvolutionaryOptimizationCPPN._deep_update(base[key], value)
            else:
                base[key] = value

    def _resolve_checkpoint_run_id(self, logger: wandb.Run | None) -> str | None:
        if self.resume_run_id is not None:
            return self.resume_run_id

        if logger is not None:
            logger_id = self._none_if_empty(getattr(logger, "id", None))
            if logger_id is not None:
                return logger_id

        env_run_id = self._none_if_empty(os.environ.get("WANDB_RUN_ID"))
        if env_run_id is not None:
            return env_run_id

        return self._slurm_run_id()

    def _checkpoint_dir(self, run_id: str | None) -> str | None:
        if self.checkpoint_root is None or run_id is None:
            return None
        return os.path.join(self.checkpoint_root, run_id)

    @staticmethod
    def _checkpoint_prefix(checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, "neat-checkpoint-")

    def _latest_checkpoint_path(self, checkpoint_dir: str) -> str | None:
        prefix = self._checkpoint_prefix(checkpoint_dir)
        latest: tuple[int, str] | None = None

        for path in glob.glob(f"{prefix}*"):
            generation = self._checkpoint_generation(path=path, prefix=prefix)
            if generation is None:
                continue
            if latest is None or generation > latest[0]:
                latest = (generation, path)

        return None if latest is None else latest[1]

    @staticmethod
    def _checkpoint_generation(path: str, prefix: str) -> int | None:
        if not path.startswith(prefix):
            return None
        suffix = path[len(prefix) :]
        if re.fullmatch(r"\d+", suffix) is None:
            return None
        return int(suffix)

    @staticmethod
    def _slurm_run_id() -> str | None:
        array_job_id = EvolutionaryOptimizationCPPN._none_if_empty(
            os.environ.get("SLURM_ARRAY_JOB_ID")
        )
        array_task_id = EvolutionaryOptimizationCPPN._none_if_empty(
            os.environ.get("SLURM_ARRAY_TASK_ID")
        )
        if array_job_id is not None and array_task_id is not None:
            return f"slurm-{array_job_id}-{array_task_id}"

        job_id = EvolutionaryOptimizationCPPN._none_if_empty(
            os.environ.get("SLURM_JOB_ID")
        )
        if job_id is not None:
            return f"slurm-{job_id}"

        return None

    @staticmethod
    def _none_if_empty(value: Any) -> str | None:
        if value is None:
            return None
        string_value = str(value).strip()
        return string_value or None

    def _is_image_logging_compatible(self) -> bool:
        return len(self.spatial_shape) == 2 and self.n_channels in (1, 3)

    def _object_to_wandb_image(
        self, obj: np.ndarray, file_type: ImageFileType
    ) -> tuple[wandb.Image, str]:
        return discrete_object_to_wandb_image(
            obj=obj,
            vocab_size=self.vocab_size,
            n_channels=self.n_channels,
            file_type=file_type,
        )

    @staticmethod
    def _resolve_device(reward_device: str) -> torch.device:
        if reward_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(reward_device)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def silu_activation(z: float) -> float:
    """SiLU with NEAT-style scaling/clamping around [-1, 1] input range.

    Following the scaling convention used by built-in NEAT activations, we:
    1) Scale input by 5.
    2) Clamp the scaled value to [-60, 60] for numerical stability.
    3) Apply SiLU in a way that preserves the original input scale.

    This yields: f(z) = z * sigmoid(5z).
    """
    scaled = max(-60.0, min(60.0, 5.0 * z))
    return z * (1.0 / (1.0 + math.exp(-scaled)))


def _decode_cppn_outputs(
    outputs: np.ndarray,
    output_encoding: OutputEncoding,
    n_channels: int,
    vocab_size: int,
) -> np.ndarray:
    """Map continuous CPPN outputs to discrete vocabulary tokens."""
    if output_encoding == "quantized_scalar":
        channel_values = outputs[:, :n_channels]
        channel_values = np.clip(channel_values, -1.0, 1.0)
        channel_values = (channel_values + 1.0) * 0.5 * (vocab_size - 1)
        return np.rint(channel_values).astype(np.int64)

    if output_encoding == "categorical_logits":
        logits = outputs.reshape(outputs.shape[0], n_channels, vocab_size)
        return logits.argmax(axis=-1).astype(np.int64)

    raise ValueError(f"Unsupported output_encoding: {output_encoding!r}")


def _render_cppn_genome(
    genome: neat.DefaultGenome,
    neat_config: neat.Config,
    coordinate_inputs: Sequence[Sequence[float]] | np.ndarray,
    spatial_shape: tuple[int, ...],
    n_channels: int,
    vocab_size: int,
    output_encoding: OutputEncoding,
    n_cppn_outputs: int,
) -> np.ndarray:
    """Render one genome into one discrete object."""
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    outputs = np.empty((len(coordinate_inputs), n_cppn_outputs), dtype=np.float32)

    for i, coord in enumerate(coordinate_inputs):
        cppn_output = net.activate(coord)
        if len(cppn_output) != n_cppn_outputs:
            raise RuntimeError(
                "Unexpected CPPN output dimensionality. "
                f"Expected {n_cppn_outputs}, got {len(cppn_output)}."
            )
        outputs[i] = np.asarray(cppn_output, dtype=np.float32)

    tokens = _decode_cppn_outputs(
        outputs=outputs,
        output_encoding=output_encoding,
        n_channels=n_channels,
        vocab_size=vocab_size,
    )
    return tokens.reshape(*spatial_shape, n_channels)


def _init_render_worker(render_context: _CPPNRenderContext) -> None:
    """Store static render data once per spawned worker."""
    global _WORKER_RENDER_CONTEXT
    _WORKER_RENDER_CONTEXT = render_context


def _render_cppn_genome_worker(genome: neat.DefaultGenome) -> np.ndarray:
    """Multiprocessing wrapper to render one genome."""
    if _WORKER_RENDER_CONTEXT is None:
        raise RuntimeError("CPPN render worker was not initialized.")
    return _render_cppn_genome(
        genome=genome,
        neat_config=_WORKER_RENDER_CONTEXT.neat_config,
        coordinate_inputs=_WORKER_RENDER_CONTEXT.coordinate_inputs,
        spatial_shape=_WORKER_RENDER_CONTEXT.spatial_shape,
        n_channels=_WORKER_RENDER_CONTEXT.n_channels,
        vocab_size=_WORKER_RENDER_CONTEXT.vocab_size,
        output_encoding=_WORKER_RENDER_CONTEXT.output_encoding,
        n_cppn_outputs=_WORKER_RENDER_CONTEXT.n_cppn_outputs,
    )
