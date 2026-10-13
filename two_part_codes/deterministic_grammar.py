from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch

from two_part_codes.base import TwoPartCode
from lib.re_pair_lib import mr_repair_encode
from lib.type_aliases import DeterministicGrammarModel


class DeterministicGrammarTwoPartCode(TwoPartCode[DeterministicGrammarModel], ABC):
    """Two-part compression based on deterministic context-free grammars."""

    def __init__(
        self,
        prune_grammar: bool = False,
        prune_to_random_grammar: bool = True,
        c: float = 0.0,
        start_at_zero: bool = False,  # If True, string symbols that are input to this class start at 0, otherwise they start at 1
    ) -> None:
        super().__init__()
        self.prune_grammar = prune_grammar
        self.prune_to_random_grammar = prune_to_random_grammar
        self.c = c
        self.start_at_zero = start_at_zero

    def complexities(
        self, grammar: list[list[int]], encoding: list[int]
    ) -> tuple[float, float]:
        # Flatten streams
        grammar_stream = [sym for rule in grammar for sym in rule]
        lhs_stream = list(range(len(grammar)))  # one LHS occurrence per rule
        all_stream = grammar_stream + encoding + lhs_stream
        if len(all_stream) == 0:
            return 0.0, 0.0

        # Build frequency distribution over the combined stream
        counts = Counter(all_stream)
        total = float(sum(counts.values()))

        # Precompute entropy code length per symbol: -log2(p)
        code_len = {sym: -np.log2(cnt / total) for sym, cnt in counts.items()}

        # Model complexity: grammar RHS + one LHS occurrence per rule
        model_complexity = sum(code_len[sym] for sym in grammar_stream) + sum(
            code_len[i] for i in lhs_stream
        )

        # Unstructured complexity: encoding RHS using the same distribution
        unstructured_complexity = sum(code_len[sym] for sym in encoding)

        return model_complexity, unstructured_complexity

    def compress_with_encoding(
        self, data: np.ndarray | torch.Tensor
    ) -> tuple[
        DeterministicGrammarModel,
        list[list[int]],
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
    ]:
        """Compress the data and return each string's final grammar encoding.

        The model is a string representation of the grammar. The model complexity is the length of the grammar string.

        Args:
            data (np.ndarray | torch.Tensor): The data to compress, with a leading batch dimension.

        Returns:
            tuple[list[list[int | str]], list[list[int]], np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]: A tuple containing a batch of grammars, each string's encoding, their complexities, and the complexities of the unstructured random components.
        """
        if isinstance(data, torch.Tensor):
            return_torch = True
            device = data.device
            data = data.cpu().numpy()
        else:
            return_torch = False

        if self.start_at_zero:
            data = data + 1  # Ensure symbols are positive integers starting at 1
        assert (data > 0).all(), "Symbols must be positive integers."
        data = [
            [-sym for sym in seq] for seq in data
        ]  # Terminals are negative integers

        def prune_grammar(
            grammar: list[list[int]],
            encoding: list[int],
            pruned_rule: int,
        ) -> tuple[list[list[int]], list[int]]:
            pruned_rule_definition = [
                sym if sym < pruned_rule else sym - 1 for sym in grammar[pruned_rule]
            ]

            def prune_rule(rule: list[int]) -> list[int]:
                new_rule = []
                for sym in rule:
                    if sym < pruned_rule:
                        new_rule.append(sym)
                    elif sym > pruned_rule:
                        new_rule.append(sym - 1)
                    else:
                        new_rule.extend(pruned_rule_definition)
                return new_rule

            new_grammar = [
                prune_rule(grammar[i]) for i in range(len(grammar)) if i != pruned_rule
            ]
            new_encoding = prune_rule(encoding)
            return new_grammar, new_encoding

        def compress_string(
            s: list[int],
        ) -> tuple[list[list[int]], list[int], float, float]:
            # Extract the grammar, the encoding of the string from the grammar, and their complexities
            grammar, encoding = self.parse_to_grammar(s)
            model_complexity, unstructured_complexity = self.complexities(
                grammar, encoding
            )
            simplest_complexity = model_complexity + unstructured_complexity

            # Try pruning one rule at a time recursively so long as compression doesn't suffer
            if self.prune_grammar:
                while len(grammar) > 0:
                    pruned = False
                    for i in range(len(grammar)):
                        pruned_grammar, pruned_encoding = prune_grammar(
                            grammar, encoding, i
                        )
                        pruned_model_complexity, pruned_unstructured_complexity = (
                            self.complexities(pruned_grammar, pruned_encoding)
                        )
                        pruned_complexity = (
                            pruned_model_complexity + pruned_unstructured_complexity
                        )
                        if pruned_complexity <= simplest_complexity + self.c:
                            grammar, encoding = pruned_grammar, pruned_encoding
                            model_complexity, unstructured_complexity = (
                                pruned_model_complexity,
                                pruned_unstructured_complexity,
                            )
                            if pruned_complexity < simplest_complexity:
                                simplest_complexity = pruned_complexity
                            pruned = True
                            break
                    if not pruned:
                        break

            # In case pruning the entire grammar and hard-coding the string is simplest
            if self.prune_to_random_grammar:
                empty_model_complexity, empty_unstructured_complexity = (
                    self.complexities([], s)
                )
                if (
                    empty_model_complexity + empty_unstructured_complexity
                    <= simplest_complexity + self.c
                ):
                    grammar, encoding = [], s
                    model_complexity, unstructured_complexity = (
                        empty_model_complexity,
                        empty_unstructured_complexity,
                    )

            return grammar, encoding, model_complexity, unstructured_complexity

        if len(data) == 0:
            grammars, encodings, model_complexities, unstructured_complexities = (
                [],
                [],
                [],
                [],
            )
        else:
            grammars, encodings, model_complexities, unstructured_complexities = zip(
                *[compress_string(s) for s in data]
            )
        grammars = list(grammars)
        encodings = list(encodings)

        if return_torch:
            model_complexities = torch.tensor(
                model_complexities,
                dtype=torch.float32,
                device=device,
            )
            unstructured_complexities = torch.tensor(
                unstructured_complexities,
                dtype=torch.float32,
                device=device,
            )
        else:
            model_complexities = np.array(
                model_complexities,
                dtype=np.float32,
            )
            unstructured_complexities = np.array(
                unstructured_complexities,
                dtype=np.float32,
            )

        return grammars, encodings, model_complexities, unstructured_complexities

    def compress(
        self, data: np.ndarray | torch.Tensor
    ) -> tuple[
        DeterministicGrammarModel, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor
    ]:
        """Compress the data into a structured model and unstructured random component.

        The model is a string representation of the grammar. The model complexity is the length of the grammar string.

        Args:
            data (np.ndarray | torch.Tensor): The data to compress, with a leading batch dimension.

        Returns:
            tuple[list[list[int | str]], np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]: A tuple containing a batch of grammars, their complexities, and the complexities of the unstructured random components.
        """
        grammars, _, model_complexities, unstructured_complexities = (
            self.compress_with_encoding(data)
        )
        return grammars, model_complexities, unstructured_complexities

    @abstractmethod
    def parse_to_grammar(self, s: list[int]) -> tuple[list[list[int]], list[int]]:
        """Parse the input string into a deterministic context-free grammar.

        Args:
            s (list[int]): The input string to parse (sequence of negative integers).

        Returns:
            tuple[list[list[int]], list[int]]: A tuple containing the grammar rules as a list of production rules from non-terminals and the encoding of the string as a sequence of non-terminal and terminal symbols.
        """
        pass


class DeterministicGrammarRePairTwoPartCode(DeterministicGrammarTwoPartCode):

    def __init__(
        self,
        prune_grammar: bool = False,
        prune_to_random_grammar: bool = True,
        c: float = 0.0,
        start_at_zero: bool = False,
        mdl: bool = True,
        mdl_mode: str = "global_static",
        mdl_lazy_top_k: int = 0,
        mdl_sa_top_k: int = 24,
        mdl_sa_every: int = 4,
        mdl_sa_min_len: int = 2,
        mdl_sa_max_len: int = 256,
    ):
        """RePair-based deterministic grammar code with MDL-guided rule induction.

        Args:
            prune_grammar (bool): If True, recursively remove rules that do not
                increase total two-part code length.
            prune_to_random_grammar (bool): If True, allow pruning to the empty
                grammar if that is no worse than the induced grammar.
            c (float): Tolerance slack for pruning comparisons.
            start_at_zero (bool): If True, terminal symbols in the input sequence
                are assumed to start at 0 and are shifted internally.
            mdl (bool): If True, use MDL gain scoring to accept rules.
            mdl_mode (str): MDL approximation mode used in candidate scoring.
            mdl_lazy_top_k (int): Number of frequent pairs to examine per
                iteration in pair-based search. Larger values can improve rule
                quality but increase per-iteration cost.
            mdl_sa_top_k (int): Number of suffix-array-derived global repeat
                candidates evaluated when global mining is triggered. Larger
                values improve robustness to non-local repeats but add roughly
                O(mdl_sa_top_k * n) matching work per trigger.
            mdl_sa_every (int): Trigger global suffix-array mining every this
                many accepted rules (and always when pair-only search stalls).
                Smaller values find global repeats earlier but increase runtime
                due to more O(n log n) suffix-array builds.
            mdl_sa_min_len (int): Minimum repeat length proposed by global
                suffix-array mining. Larger values reduce noise and speed up
                matching, but may miss short useful rules.
            mdl_sa_max_len (int): Maximum repeat length evaluated by global
                suffix-array mining. Larger values can capture long motifs but
                increase candidate-matching cost.
        """
        if not mdl:
            assert not (
                prune_grammar or prune_to_random_grammar
            ), "Pruning grammars fit without MDL enabled is too likely to trivially remove the entire grammar"
        super().__init__(
            prune_grammar=prune_grammar,
            prune_to_random_grammar=prune_to_random_grammar,
            c=c,
            start_at_zero=start_at_zero,
        )
        self.mdl = mdl
        self.mdl_mode = mdl_mode
        self.mdl_lazy_top_k = mdl_lazy_top_k
        self.mdl_sa_top_k = mdl_sa_top_k
        self.mdl_sa_every = mdl_sa_every
        self.mdl_sa_min_len = mdl_sa_min_len
        self.mdl_sa_max_len = mdl_sa_max_len

    def parse_to_grammar(self, s: list[int]) -> tuple[list[list[int]], list[int]]:
        grammar, encoding, vocab = mr_repair_encode(
            input_text=s,
            mdl=self.mdl,
            mdl_mode=self.mdl_mode,
            mdl_lazy_top_k=self.mdl_lazy_top_k,
            mdl_sa_top_k=self.mdl_sa_top_k,
            mdl_sa_every=self.mdl_sa_every,
            mdl_sa_min_len=self.mdl_sa_min_len,
            mdl_sa_max_len=self.mdl_sa_max_len,
        )
        num_terminals = len(vocab["term2id"])
        grammar = [grammar[i + num_terminals] for i in range(len(grammar))]
        grammar = [
            [
                sym - num_terminals if sym >= num_terminals else vocab["id2term"][sym]
                for sym in rule
            ]
            for rule in grammar
        ]
        encoding = [
            sym - num_terminals if sym >= num_terminals else vocab["id2term"][sym]
            for sym in encoding
        ]
        return grammar, encoding


def grammar_statistics(grammars: DeterministicGrammarModel) -> dict[str, np.ndarray]:
    def hierarchy(g, i):
        if all([s < 0 for s in g[i]]):
            return 0
        else:
            max_h = 0
            for s in g[i]:
                if s >= 0:
                    h = 1 + hierarchy(g, s)
                    if h > max_h:
                        max_h = h
            return max_h

    return {
        "num_rules": np.array([len(grammar) for grammar in grammars]),
        "num_symbols": np.array([sum(len(r) for r in grammar) for grammar in grammars]),
        "num_nonterminals": np.array(
            [
                sum(len([s for s in r if s >= 0]) for r in grammar)
                for grammar in grammars
            ]
        ),
        "mean_rule_length": np.array(
            [
                (1 / max(len(grammar), 1)) * sum(len(r) for r in grammar)
                for grammar in grammars
            ]
        ),
        "mean_depth": np.array(
            [
                (1 / max(len(grammar), 1))
                * sum(hierarchy(grammar, i) for i in range(len(grammar)))
                for grammar in grammars
            ]
        ),
    }
