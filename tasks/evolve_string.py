from typing import Any

import numpy as np
import wandb
from pymoo.core.algorithm import Algorithm

from lib import helpers
from lib.evolve_base import EvolutionaryOptimization
from rewards.base import Reward
from two_part_codes.deterministic_grammar import (
    DeterministicGrammarRePairTwoPartCode,
    grammar_statistics,
)


class EvolutionaryOptimizationString(EvolutionaryOptimization):
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
        super().__init__(
            reward=reward,
            vocab_size=vocab_size,
            string_length=string_length,
            n_workers=n_workers,
            n_logged_examples=n_logged_examples,
            log_examples_every_n_generations=log_examples_every_n_generations,
            n_expensive_metric_samples=n_expensive_metric_samples,
            log_expensive_metrics_every_n_generations=log_expensive_metrics_every_n_generations,
        )

        self.repair_grammar_fitter = DeterministicGrammarRePairTwoPartCode(
            mdl=False,
            prune_to_random_grammar=False,
            start_at_zero=True,
        )

    def log_additional_metrics(self, algorithm: Algorithm) -> dict[str, Any]:
        rewards = -algorithm.pop.get("F")[:, 0]
        solutions = algorithm.pop.get("X")
        best_indices = np.argsort(-rewards)[: self.n_expensive_metric_samples]
        best_solutions = solutions[best_indices]
        metrics = {}

        # Grammar statistics
        if (
            algorithm.n_gen % self.log_expensive_metrics_every_n_generations == 1
            or algorithm.n_gen == algorithm.termination.n_max_gen
        ):
            grammars = self.repair_grammar_fitter.compress(best_solutions)[0]
            statistics = grammar_statistics(grammars)
            for k, v in statistics.items():
                metrics[f"grammar_statistics/{k}"] = v.mean().item()

        return metrics

    def log_examples(self, algorithm: Algorithm) -> wandb.Table:
        rewards = -algorithm.pop.get("F")[:, 0]
        best_indices = np.argsort(-rewards)[: self.n_logged_examples]
        best_solutions = algorithm.pop.get("X")[best_indices]
        best_rewards = rewards[best_indices]
        data = {"reward": best_rewards.tolist()}

        # Verbatim examples
        verbatim = []
        for i in range(len(best_solutions)):
            verbatim.append(" ".join([str(int(c)) for c in best_solutions[i]]))
        data["string"] = verbatim

        # Audio examples
        sample_rate = 16000
        waveforms = helpers.songs_from_strings(
            best_solutions, self.vocab_size, sample_rate=sample_rate
        )
        audio_items = []
        for waveform in waveforms:
            audio_items.append(wandb.Audio(waveform, sample_rate=sample_rate))
        data["audio"] = audio_items

        # Visual examples
        data["dot_plot"] = helpers.dot_plots_from_strings(best_solutions)
        data["hilbert_curve"] = helpers.hilbert_curves_from_strings(best_solutions)
        data["path"] = helpers.paths_from_strings(
            best_solutions, vocab_size=self.vocab_size
        )
        data["circle_path"] = helpers.circle_paths_from_strings(
            best_solutions, vocab_size=self.vocab_size
        )

        # RePair grammars of best examples
        grammars = self.repair_grammar_fitter.compress(best_solutions)[0]
        grammars = [
            "\n".join(
                f"R{i} -> "
                + " ".join([str(-s - 1) if s < 0 else f"R{s}" for s in rule])
                for i, rule in enumerate(g)
            )
            for g in grammars
        ]
        data["grammar"] = grammars

        return wandb.Table(data=list(zip(*data.values())), columns=list(data.keys()))
