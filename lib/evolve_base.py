import multiprocessing
import warnings
from typing import Any, Callable

warnings.filterwarnings("ignore", message="Event loop was unresponsive in Nanny")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import wandb
from dask.distributed import Client, LocalCluster
from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem, LoopedElementwiseEvaluation
from pymoo.core.result import Result
from pymoo.optimize import minimize

from rewards.base import Reward
from rewards.decomposition_level_mean import DecompositionLevelMeanReward


class EvolutionaryOptimization:
    def __init__(
        self,
        reward: Reward,
        vocab_size: int,
        string_length: int,
        n_workers: int = -1,
        n_logged_examples: int = 20,
        log_examples_every_n_generations: int = 100,
        n_expensive_metric_samples: int = 100,
        log_expensive_metrics_every_n_generations: int = 50,
    ):
        assert (
            n_workers == -1 or n_workers > 0
        ), "n_workers must be -1 (all CPUs) or > 0"
        if n_workers == -1:
            n_workers = multiprocessing.cpu_count()

        self.vocab_size = vocab_size
        self.string_length = string_length
        self.reward = reward
        self.n_workers = n_workers
        self.n_logged_examples = n_logged_examples
        self.log_examples_every_n_generations = log_examples_every_n_generations
        self.n_expensive_metric_samples = n_expensive_metric_samples
        self.log_expensive_metrics_every_n_generations = (
            log_expensive_metrics_every_n_generations
        )

        if isinstance(reward, DecompositionLevelMeanReward):
            self.using_decomposition_level_mean_reward = True
        else:
            self.using_decomposition_level_mean_reward = False

    def optimize(
        self,
        algorithm: Algorithm,
        n_generations: int,
        logger: wandb.Run | None = None,
        seed: int = 0,
    ) -> Result:
        if logger is not None:
            callback = self.make_logger_callback(logger)
        else:
            callback = Callback()

        if self.n_workers > 1:
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=1,
                processes=True,
            )
            client = Client(cluster)
            print(f"Dask Dashboard running at: {client.dashboard_link}")

            def dask_runner(f, args):
                futures = client.map(f, args, pure=False)
                results = client.gather(futures)
                return results

            elementwise_runner = dask_runner
        else:
            elementwise_runner = LoopedElementwiseEvaluation()
        problem = self.make_problem(
            elementwise_runner=elementwise_runner,
        )

        try:
            result = minimize(
                problem=problem,
                algorithm=algorithm,
                termination=("n_gen", n_generations),
                callback=callback,
                seed=seed,
                save_history=False,
                verbose=True,
            )
        finally:
            if logger is not None:
                wandb.finish()
            if self.n_workers > 1:
                client.close()
                cluster.close()

        return result

    def reward_histogram(self, rewards: np.ndarray) -> wandb.Histogram:
        rewards = rewards.clip(min=1e-8)  # Avoid issues with log scale
        _, bins = np.histogram(rewards, bins=30)
        bins = bins.clip(min=1e-8)  # Avoid issues with log scale
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        loghist = np.histogram(rewards, bins=logbins)
        histogram = wandb.Histogram(np_histogram=loghist)
        return histogram

    def reward_specific_metrics(self, solutions: np.ndarray) -> dict[str, Any]:
        if self.using_decomposition_level_mean_reward:
            level_reward_func: DecompositionLevelMeanReward = self.reward
            _, reward_compositions = level_reward_func(
                torch.tensor(solutions), return_compositions=True
            )
            level_rewards = level_reward_func.compositions_to_level_rewards(
                reward_compositions, obj_shape=(self.string_length,)
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

    def make_problem(
        self,
        elementwise_runner: Callable[
            [Callable[[np.ndarray], float], np.ndarray], list[float]
        ],
    ) -> ElementwiseProblem:
        # NOTE: Can override in subclass if you want to return a subclass of EvolutionaryOptimizationProblem
        return EvolutionaryOptimizationProblem(
            reward=self.reward,
            string_length=self.string_length,
            vocab_size=self.vocab_size,
            elementwise_runner=elementwise_runner,
        )

    def make_logger_callback(self, logger: wandb.Run) -> Callback:
        outer = self

        class _(Callback):
            def __init__(self):
                super().__init__()

            def notify(self, algorithm: Algorithm) -> None:
                rewards = -algorithm.pop.get("F")[:, 0]
                metrics = {
                    "mean_reward": np.mean(rewards),
                    "max_reward": np.max(rewards),
                    "min_reward": np.min(rewards),
                    "std_reward": np.std(rewards),
                    "reward_histogram": outer.reward_histogram(rewards),
                }
                if (
                    algorithm.n_gen % outer.log_expensive_metrics_every_n_generations
                    == 1
                    or algorithm.n_gen == algorithm.termination.n_max_gen
                ):
                    solutions = algorithm.pop.get("X")
                    best_indices = np.argsort(-rewards)[
                        : outer.n_expensive_metric_samples
                    ]
                    best_solutions = solutions[best_indices]
                    metrics = {
                        **metrics,
                        **outer.reward_specific_metrics(best_solutions),
                    }
                metrics = {**metrics, **outer.log_additional_metrics(algorithm)}
                if (
                    algorithm.n_gen % outer.log_examples_every_n_generations == 1
                    or algorithm.n_gen == algorithm.termination.n_max_gen
                ):
                    examples = outer.log_examples(algorithm)
                    if examples is not None:
                        metrics["examples"] = examples
                logger.log(metrics)

        return _()

    def log_additional_metrics(self, algorithm: Algorithm) -> dict[str, Any]:
        # NOTE: Can override in subclass
        return {}

    def log_examples(self, algorithm: Algorithm) -> wandb.Table | None:
        # NOTE: Can override in subclass
        return None


class EvolutionaryOptimizationProblem(ElementwiseProblem):
    def __init__(
        self,
        reward: Reward,
        string_length: int,
        vocab_size: int,
        elementwise_runner: Callable,
    ):
        super().__init__(
            n_var=string_length,
            xl=0,
            xu=vocab_size - 1,
            vtype=int,
            n_obj=1,
            n_constr=0,
            elementwise_runner=elementwise_runner,
        )
        self.reward = reward

    def _evaluate(self, x: np.ndarray, out: dict[str, float], *args, **kwargs) -> None:
        out["F"] = -self.reward_function_wrapper(x)

    def reward_function_wrapper(self, x: np.ndarray) -> float:
        # NOTE: Can override in subclass
        x = torch.from_numpy(x)[None]
        r = self.reward(x)[0].item()
        return r

    def __getstate__(self):
        """
        Custom serialization to exclude 'elementwise_runner'.
        The runner contains the Dask Client, which cannot be pickled.
        Workers do not need the runner, so we can safely remove it.
        """
        state = self.__dict__.copy()
        if "elementwise_runner" in state:
            del state["elementwise_runner"]
        return state

    def __setstate__(self, state):
        """
        Restore state and set a default (None) runner for workers.
        """
        self.__dict__.update(state)
        # On the worker, we don't need the dask runner.
        # Setting to None falls back to standard execution if needed.
        self.elementwise_runner = None
