import itertools
import math
import random
import time
from typing import Literal

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


from conditional_compressors.base import ConditionalCompressor
from lib.type_aliases import DeterministicGrammarModel
from lib.helpers import dictionary_conditional_compressed_size_bits


class DeterministicGrammarPatternMatchingConditionalCompressor(
    ConditionalCompressor[DeterministicGrammarModel]
):
    """Conditional compressor using deterministic grammar pattern matching.

    Key features:
    - Tensorized representation with shared terminal vocabulary across target + givens.
    - Fast structure-aware alignment via:
        * WL-style message passing to produce relabel-invariant rule embeddings
        * A single linear assignment (Hungarian algorithm) on a dense cost matrix
    - Exhaustive search for tiny grammars (N <= max_nonterminals_for_exhaustive_search)
    """

    def __init__(
        self,
        max_nonterminals_for_exhaustive_search: int = 6,
        max_shift: int = 10,
        wl_embed_dim: int = 64,
        wl_iters: int = 3,
        wl_pos_scale: float = 0.1,
        wl_smooth: bool = True,
        compression_strategy: Literal["diff", "dictionary"] = "diff",
        device: torch.device | str | None = None,
    ):
        """Initialize the compressor with optimization parameters.

        Args:
            max_nonterminals_for_exhaustive_search: N threshold to use exhaustive permutation search
            max_shift: Max +/- positions to consider in shift-aware cost
            wl_embed_dim: Dimension of WL rule embeddings
            wl_iters: Number of WL/message-passing iterations (2-4 is typically enough)
            wl_pos_scale: Sinusoidal PE scale (smaller => more shift-robust)
            wl_smooth: Apply tiny 1D smoothing along positions
            compression_strategy: "diff" for rule-diff-based compression; "dictionary" for dictionary-based compression
            device: torch device to use; defaults to CUDA if available
        """
        super().__init__()

        self.max_nonterminals_for_exhaustive_search = (
            max_nonterminals_for_exhaustive_search
        )
        self.max_shift = max_shift
        self.wl_embed_dim = wl_embed_dim
        self.wl_iters = wl_iters
        self.wl_pos_scale = wl_pos_scale
        self.wl_smooth = wl_smooth
        self.compression_strategy = compression_strategy
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

    # ---------------------------
    # Public API
    # ---------------------------
    def compress(
        self,
        model: DeterministicGrammarModel,
        given_models: list[DeterministicGrammarModel],
    ) -> torch.Tensor:
        """Compute conditional compression length of one or a batch of target grammars.

        Terminals are integers < 0; Nonterminals are integers >= 0.
        """
        batch_size = len(model)
        if batch_size == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        # --- 1) Preprocess: shapes + unified terminal vocab across target and givens ---
        (
            max_rules,
            max_rule_len,
            terminal_to_id,
            vocab_size,
            pad_id,
            target_num_nonterminals,
            target_num_terminals,
        ) = _analyze_and_build_vocab(model, given_models)

        # Tensorize
        target_tensor, target_rules_mask = _tensorize(
            model, max_rules, max_rule_len, terminal_to_id, pad_id, self.device
        )
        given_tensors = []
        for gm in given_models:
            gt, _ = _tensorize(
                gm, max_rules, max_rule_len, terminal_to_id, pad_id, self.device
            )
            given_tensors.append(gt)

        # bits per symbol per target item (log2 |Sigma| with grammar-specific terminals)
        bits_per_symbol = torch.log2(
            (target_num_nonterminals + target_num_terminals).clamp_min(1)
        ).to(self.device)

        # --- 2) Raw cost if encoded from scratch ---
        if self.compression_strategy == "diff":
            num_symbols_per_grammar = (
                (target_tensor != pad_id).sum(dim=[1, 2]).to(torch.float32)
            )
            raw_cost = num_symbols_per_grammar * bits_per_symbol
        elif self.compression_strategy == "dictionary":
            raw_cost = self._dictionary_compress(model, [])
        else:
            raise ValueError(
                f"Unknown compression_strategy: {self.compression_strategy}"
            )

        if not given_models:
            return raw_cost

        # --- 3) Conditional costs (rule-wise min across sources) ---
        raw_rule_costs = self._rule_diff_cost(
            target_tensor,
            torch.full_like(target_tensor, pad_id),
            bits_per_symbol,
            target_rules_mask,
            pad_id,
        )  # (B, N)

        all_rule_cost_options = [raw_rule_costs.unsqueeze(1)]  # (B, 1, N)
        all_best_perms = []
        for given_t in given_tensors:
            best_perms, given_rule_costs = self._find_best_alignment(
                target_t=target_tensor,
                given_t=given_t,
                pad_id=pad_id,
                vocab_size=vocab_size,
                bits_per_symbol=bits_per_symbol,
                target_rules_mask=target_rules_mask,  # only sum over real target rules
            )
            all_rule_cost_options.append(given_rule_costs.unsqueeze(1))
            all_best_perms.append(best_perms.cpu().numpy().tolist())

        if self.compression_strategy == "diff":
            # Reduce across sources
            stacked_costs = torch.cat(all_rule_cost_options, dim=1)  # (B, S, N)
            min_rule_costs, _ = torch.min(stacked_costs, dim=1)  # (B, N)
            conditional_cost = min_rule_costs.sum(dim=-1)  # (B,)
        elif self.compression_strategy == "dictionary":
            # Permute the given grammars according to best_perms then do dictionary compression
            permuted_given_models = [
                [
                    _permute_grammar(g, p, is_new_to_old=True)
                    for g, p in zip(g_batch, p_batch)
                ]
                for g_batch, p_batch in zip(given_models, all_best_perms)
            ]
            conditional_cost = self._dictionary_compress(model, permuted_given_models)
        else:
            raise ValueError(
                f"Unknown compression_strategy: {self.compression_strategy}"
            )

        final_cost = torch.minimum(raw_cost, conditional_cost)
        return final_cost

    def _dictionary_compress(
        self,
        target_grammar: DeterministicGrammarModel,
        given_grammars: list[DeterministicGrammarModel],
    ) -> torch.Tensor:
        """Compute conditional compression length via dictionary-based method."""
        flattened_target_grammar = [
            [sym for rule in grammar for sym in rule] for grammar in target_grammar
        ]
        given_parts = []
        for i in range(len(target_grammar)):
            given_parts_i = []
            for given_grammar in given_grammars:
                for rule in given_grammar[i]:
                    if len(rule) > 0 and rule not in given_parts_i:
                        given_parts_i.append(rule)
            given_parts.append(given_parts_i)
        cost = [
            dictionary_conditional_compressed_size_bits(
                flattened_target_grammar[i], given_parts[i]
            )["total_bits"]
            for i in range(len(target_grammar))
        ]
        return torch.tensor(cost, dtype=torch.float32, device=self.device)

    def _rule_diff_cost(
        self,
        target_rules: torch.Tensor,
        given_rules: torch.Tensor,
        bits_per_symbol: torch.Tensor,
        target_rules_mask: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        """Compute rule-wise diff cost between two (..., N, L) tensors as Hamming * bits_per_symbol * target_rules_mask.

        We take the minimum Hamming distance over several possible shifts of the given_rules, to allow for left/right insertion/deletion costs.
        """
        x, y = torch.broadcast_tensors(target_rules, given_rules)  # (..., N, L)
        L, K, lead = x.shape[-1], x.shape[:-1].numel(), x.shape[:-1]
        S = self.max_shift
        M = 2 * S + 1  # number of shifts

        # Pad both sides with pad_id so uncovered tokens count as ins/del
        x, y = x.reshape(K, L), y.reshape(K, L)
        x, y = (
            F.pad(x, (S, S), value=pad_id),
            F.pad(y, (S, S), value=pad_id),
        )  # (K, L+2S)

        # Expand with M shifts
        idxL = torch.arange(L + 2 * S, device=x.device)  # (L+2S,)
        shifts = torch.arange(-S, S + 1, device=x.device)  # (M,)
        x_idx = idxL.expand(M, L + 2 * S)  # (M, L + 2S)
        y_idx = (idxL[None] - shifts[:, None]).clamp(0, L + 2 * S - 1)  # (M, L + 2S)
        x = (
            x.unsqueeze(1)
            .expand(K, M, L + 2 * S)
            .gather(2, x_idx.unsqueeze(0).expand(K, M, L + 2 * S))
        )  # (K, M, L+2S)
        y = (
            y.unsqueeze(1)
            .expand(K, M, L + 2 * S)
            .gather(2, y_idx.unsqueeze(0).expand(K, M, L + 2 * S))
        )  # (K, M, L+2S)

        # Minimum Hamming distance over shifts
        mismatches = (x != y).sum(dim=2)  # (K, M)
        mismatches = mismatches.min(dim=1).values  # (K,), best over shifts
        mismatches = mismatches.reshape(lead).to(torch.float32)  # (..., N)

        # Bit-encoding cost
        bits_per_symbol = bits_per_symbol.view(
            bits_per_symbol.shape[0], *(1,) * (mismatches.dim() - 1)
        )
        target_rules_mask = target_rules_mask.view(
            target_rules_mask.shape[0],
            *(1,) * (mismatches.dim() - 2),
            target_rules_mask.shape[1],
        )
        return mismatches * bits_per_symbol * target_rules_mask

    # ---------------------------
    # Alignment core
    # ---------------------------
    def _find_best_alignment(
        self,
        target_t: torch.Tensor,  # (B,N,L)
        given_t: torch.Tensor,  # (B,N,L)
        pad_id: int,
        vocab_size: int,
        bits_per_symbol: torch.Tensor,  # (B,)
        target_rules_mask: torch.Tensor,  # (B,N) mask of real target rules
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (best_perms, best_rule_costs) for aligning 'given' to 'target'.

        best_perms: (B, N) permutation mapping target rule index -> given rule index
        best_rule_costs: (B, N) exact diff cost per target rule against its aligned given rule
        """
        device = target_t.device
        B, N, L = target_t.shape

        if N == 0:
            return torch.empty(B, 0, dtype=torch.long, device=device), torch.zeros(
                B, 0, dtype=torch.float32, device=device
            )

        # 1) Tiny grammars: exhaustive permutations
        if N <= self.max_nonterminals_for_exhaustive_search:
            perms = list(itertools.permutations(range(N)))  # interpret as old->new
            perms_tensor = torch.tensor(perms, dtype=torch.long, device=device)  # (P,N)
            P = perms_tensor.shape[0]
            # inverse perms for row gather (new->old)
            perms_inv = torch.empty_like(perms_tensor)
            arP = torch.arange(N, device=device).expand_as(perms_tensor)
            perms_inv.scatter_(1, perms_tensor, arP)  # perms_inv[new] = old
            # Row permute: put old rows into new positions
            row_perm = given_t[:, None, :, :].expand(B, P, N, L)
            row_perm = row_perm.gather(
                2, perms_inv[None, :, :, None].expand(B, P, N, L)
            )
            # Content remap via LUT using forward (old->new)
            lut = (
                torch.arange(vocab_size, device=device)
                .unsqueeze(0)
                .expand(P, -1)
                .clone()
            )
            lut[:, 1 : N + 1] = perms_tensor + 1  # old j -> new p[j]
            lut = lut[None, :, None, :].expand(B, P, N, vocab_size)
            fully_perm = lut.gather(3, row_perm)  # (B,P,N,L)

            # Exact diff costs
            per_rule_costs = self._rule_diff_cost(
                target_t[:, None, :, :],
                fully_perm,
                bits_per_symbol,
                target_rules_mask,
                pad_id,
            )  # (B,P,N)
            total = per_rule_costs.sum(-1)  # (B,P)
            best_idx = total.argmin(dim=1)

            # Normalize return permutation to new->old
            best_old2new = perms_tensor[best_idx]  # (B,N)
            best_new2old = torch.empty_like(best_old2new)  # (B,N)
            arB = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # (B,N)
            best_new2old.scatter_(1, best_old2new, arB)  # inverse: new2old[p[j]]=j

            best_rule_costs = per_rule_costs[
                torch.arange(B, device=device), best_idx
            ]  # (B,N)
            return best_new2old, best_rule_costs

        # 2) Large grammars: structure-aware assignment

        # 2a) Build WL embeddings for target & given
        h_target = _wl_rule_embeddings(
            target_t,
            pad_id,
            self.wl_embed_dim,
            self.wl_iters,
            self.wl_pos_scale,
            self.wl_smooth,
        )
        h_given = _wl_rule_embeddings(
            given_t,
            pad_id,
            self.wl_embed_dim,
            self.wl_iters,
            self.wl_pos_scale,
            self.wl_smooth,
        )

        # 2b) Cost matrix = emb_dist
        C = self._build_cost_matrix(h_target, h_given)  # (B, N, N)
        # Data-driven pad penalty: a bit worse than the worst *real* column.
        # Keeps` value range tight and avoids pathological bidding on padded cols.
        given_len = (given_t != pad_id).sum(dim=-1)  # (B, N)
        nonpad = given_len > 0  # (B, N)
        # Max cost among real columns per batch (fallback to overall max if none)
        real_col_max = C.masked_fill(~nonpad[:, None, :], float("-inf")).amax(
            dim=(1, 2)
        )
        fallback = C.amax(dim=(1, 2))
        real_col_max = torch.where(torch.isfinite(real_col_max), real_col_max, fallback)
        pad_penalty = (real_col_max + 1.0)[:, None, None].expand_as(
            C
        )  # +1 to stay strictly worse
        C = torch.where(nonpad[:, None, :], C, pad_penalty)

        # There is no cost for assigning a given rule to an empty target rule (which is all pad);
        # it simply doesn't help compress the target.
        C = C * target_rules_mask[:, :, None]

        # 2c) Solve assignment per batch item
        perms_out = torch.empty((B, N), dtype=torch.long, device=device)
        for b in range(B):
            perms_out[b] = self.linear_sum_assignment_min(C[b])

        # 2d) Compute exact costs given final permutation
        best_rule_costs = self._costs_with_perm(
            target_t,
            given_t,
            perms_out,
            bits_per_symbol,
            target_rules_mask,
            vocab_size,
            pad_id,
        )

        return perms_out, best_rule_costs

    def _build_cost_matrix(
        self,
        h_target: torch.Tensor,
        h_given: torch.Tensor,
    ) -> torch.Tensor:
        """Combine embedding distance, terminal-only mismatches, and length penalty."""
        # Embedding squared distance
        # (B,N,1,d) - (B,1,N,d) -> (B,N,N)
        diff = h_target[:, :, None, :] - h_given[:, None, :, :]
        cost = (diff * diff).sum(dim=-1)
        return cost

    def linear_sum_assignment_min(self, cost: torch.Tensor) -> torch.Tensor:
        """Linear sum assignment (min) via SciPy's Hungarian algorithm.

        Args:
            cost: (N,N) cost matrix (any device).
        Returns:
            assign: (N,) long tensor where assign[i] = column j chosen for row i.  (new->old)
        """
        # Force CPU; do math in float64 for stability
        c = cost.detach().to("cpu", dtype=torch.float64).numpy()
        row_ind, col_ind = linear_sum_assignment(c)

        # Expect square; row_ind should be [0..N-1] in order, but guard anyway.
        assign = torch.full((c.shape[0],), -1, dtype=torch.long)
        assign[torch.from_numpy(row_ind)] = torch.from_numpy(col_ind)
        return assign.to(device=cost.device)

    def _costs_with_perm(
        self,
        target_t: torch.Tensor,
        given_t: torch.Tensor,
        perms: torch.Tensor,  # (B,N)
        bits_per_symbol: torch.Tensor,
        target_rules_mask: torch.Tensor,
        vocab_size: int,
        pad_id: int,
    ) -> torch.Tensor:
        """Exact per-rule diff costs after applying row+content permutation.

        Conventions:
          - perms[i] = old given row to place at target row i (new->old).
          - Content remap must map old nonterminal ids -> new ids.
        """

        device = target_t.device
        B, N, L = target_t.shape

        # 1) Row permutation with FORWARD mapping (new->old)
        given_perm = given_t.gather(1, perms[:, :, None].expand(B, N, L))  # (B,N,L)

        # 2) Build forward mapping (old->new) for content remap: forward[old] = new
        forward = torch.empty_like(perms)  # (B,N)
        arB = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        forward.scatter_(1, perms, arB)

        # 3) LUT: map token j+1 (old nonterminal) -> forward[j] + 1 (new nonterminal)
        lut = torch.arange(vocab_size, device=device).unsqueeze(0).expand(B, -1).clone()
        lut[:, 1 : N + 1] = forward + 1  # nonterminals only
        lut_exp = lut[:, None, None, :].expand(B, N, L, vocab_size)

        remapped = lut_exp.gather(3, given_perm.unsqueeze(-1)).squeeze(-1)

        costs = self._rule_diff_cost(
            target_t,
            remapped,
            bits_per_symbol,
            target_rules_mask,
            pad_id,
        )  # (B, N)

        return costs


def _analyze_and_build_vocab(
    model: DeterministicGrammarModel,
    given_models: list[DeterministicGrammarModel],
) -> tuple[int, int, dict, int, int, torch.Tensor, torch.Tensor]:
    """Scan target+givens to compute shapes and a unified terminal vocabulary."""
    # normalize given_models to lists of grammars
    all_batches = []
    for g in model:
        all_batches.append(g)
    for gm in given_models:
        all_batches.extend(gm)

    max_rules = 0
    max_rule_len = 0
    terminals_set = set()
    target_num_nonterminals = []
    target_num_terminals = []
    # For target-specific per-item counts (terminals present in that grammar)
    for grammar in model:
        max_rules = max(max_rules, len(grammar))
        target_num_nonterminals.append(len(grammar))
        term_set = set()
        for rule in grammar:
            max_rule_len = max(max_rule_len, len(rule))
            for sym in rule:
                if sym < 0:
                    terminals_set.add(sym)
                    term_set.add(sym)
        target_num_terminals.append(len(term_set))

    # Also ensure shape covers givens
    for grammar in all_batches[len(model) :]:
        max_rules = max(max_rules, len(grammar))
        for rule in grammar:
            max_rule_len = max(max_rule_len, len(rule))
            for sym in rule:
                if sym < 0:
                    terminals_set.add(sym)

    # Assign terminal ids after nonterminals block
    terminal_list = sorted(terminals_set)  # deterministic
    terminal_to_id = {t: (i + max_rules + 1) for i, t in enumerate(terminal_list)}
    vocab_size = max_rules + 1 + len(terminal_list)  # +1 for pad_id=0
    pad_id = 0

    return (
        max_rules,
        max_rule_len,
        terminal_to_id,
        vocab_size,
        pad_id,
        torch.tensor(target_num_nonterminals, dtype=torch.long),
        torch.tensor(target_num_terminals, dtype=torch.long),
    )


def _tensorize(
    batched_grammars: DeterministicGrammarModel,
    max_rules: int,
    max_rule_len: int,
    terminal_to_id: dict,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map batched grammars to a padded tensor of token ids.

    Nonterminals s>=0 map to s+1 (in [1..max_rules]); terminals t<0 map via terminal_to_id[t] (>max_rules).
    Returns:
        tensor: (B, N, L)
        rules_mask: (B, N) 1 where rule exists, 0 for padded rules
    """
    B = len(batched_grammars)
    tensor = torch.full(
        (B, max_rules, max_rule_len), pad_id, dtype=torch.long, device=device
    )
    rules_mask = torch.zeros((B, max_rules), dtype=torch.float32, device=device)

    for i, grammar in enumerate(batched_grammars):
        n = len(grammar)
        rules_mask[i, :n] = 1.0
        for j, rule in enumerate(grammar):
            l = min(len(rule), max_rule_len)
            for k in range(l):
                s = rule[k]
                if s >= 0:
                    # nonterminal id in [0..n-1], map to 1..max_rules
                    if s >= max_rules:
                        # if someone references a nonterminal >= max_rules, clamp (shouldn't happen if consistent)
                        s = min(s, max_rules - 1)
                    tensor[i, j, k] = s + 1
                else:
                    tensor[i, j, k] = terminal_to_id.get(s, pad_id)
    return tensor, rules_mask


def _wl_rule_embeddings(
    x: torch.Tensor,
    pad_id: int,
    wl_embed_dim: int,
    wl_iters: int,
    wl_pos_scale: float,
    wl_smooth: bool,
    seed: int = 0,
) -> torch.Tensor:
    """Compute WL-style embeddings per rule for a (B,N,L) grammar tensor."""
    device = x.device
    B, N, L = x.shape
    d = wl_embed_dim
    T = wl_iters

    # Terminal ids are > N; we need a terminal embedding table covering up to max token seen.
    # We use fixed random terminal embeddings (same for target & given)
    max_token = int(x.max().item()) if x.numel() > 0 else (N + 1)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    E_term = torch.randn((max_token + 1, d), generator=rng, device=device) * (
        1.0 / math.sqrt(d)
    )
    # We'll zero-out pad and overwrite nonterminal slots per-iter; terminals remain random.
    E_term[0] = 0.0  # pad

    # Sinusoidal positional encodings (shift-smooth)
    pos_enc = _sinusoidal_positional_encoding(L, d, device) * wl_pos_scale  # (L,d)

    # Initialize rule embeddings (t=0) with terminal-only aggregate
    # Build symbol embeddings: at t=0, nonterminals emb = 0, terminals emb = E_term[idx]
    term_mask = (x > N).unsqueeze(-1)  # (B,N,L,1)
    not_pad = (x != pad_id).unsqueeze(-1)
    emb0 = torch.where(term_mask, E_term[x], torch.zeros((B, N, L, d), device=device))
    # Apply positional hash and mask out pads
    emb0 = (emb0 + pos_enc[None, None, :, :]) * not_pad
    # Optional smoothing along positions to reduce sensitivity to tiny shifts
    if wl_smooth:
        emb0 = _smooth_along_positions(emb0)
        emb0 = emb0 * not_pad
    R = emb0.sum(dim=2)  # (B,N,d)
    denom = not_pad.sum(dim=2).clamp_min(1.0).to(torch.float32)
    R = R / denom

    # Iterative refinement: at each step, nonterminal symbol embedding is previous step's rule embedding
    for _ in range(T):
        # Build symbol embeddings from previous R for nonterminals, terminals from E_term
        sym_is_nt = (x >= 1) & (x <= N)
        # Gather previous R by nonterminal id x-1
        nt_idx = (x - 1).clamp_min(0)
        # Advanced indexing per batch to gather rule embeddings for nonterminal symbols
        # For terminal positions, set a safe index (0) since we'll mask them out anyway.
        nt_idx_safe = nt_idx.clone()
        nt_idx_safe[~sym_is_nt] = 0
        nt_emb_list = []
        for b in range(B):
            nt_emb_list.append(R[b][nt_idx_safe[b]])  # (N,L,d)
        nt_emb = torch.stack(nt_emb_list, dim=0)  # (B,N,L,d)
        sym_emb = torch.where(sym_is_nt.unsqueeze(-1), nt_emb, E_term[x])
        # Mask pads and add positional encoding
        sym_emb = (sym_emb + pos_enc[None, None, :, :]) * not_pad
        if wl_smooth:
            sym_emb = _smooth_along_positions(sym_emb)
            sym_emb = sym_emb * not_pad
        R_new = sym_emb.sum(dim=2)
        denom = not_pad.sum(dim=2).clamp_min(1.0).to(torch.float32)
        R_new = R_new / denom
        R = R_new

    return R  # (B,N,d)


def _smooth_along_positions(X: torch.Tensor) -> torch.Tensor:
    """Apply a tiny 1D smoothing (kernel 3) along the position dimension."""
    # X: (B,N,L,d)
    _, _, L, _ = X.shape
    if L == 1:
        return X
    pad_left = X[:, :, :1, :]
    pad_right = X[:, :, -1:, :]
    Xp = torch.cat([pad_left, X, pad_right], dim=2)  # (B,N,L+2,d)
    # kernel [0.25, 0.5, 0.25]
    return 0.25 * Xp[:, :, :-2, :] + 0.5 * Xp[:, :, 1:-1, :] + 0.25 * Xp[:, :, 2:, :]


def _sinusoidal_positional_encoding(
    L: int, d: int, device: torch.device
) -> torch.Tensor:
    """Standard transformer sinusoidal encoding of length L and dim d."""
    pe = torch.zeros(L, d, device=device)
    position = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _permute_grammar(
    grammar: list[list[int]], perm: list[int], is_new_to_old: bool = False
) -> list[list[int]]:
    """Return a new grammar whose nonterminals are relabeled by 'perm' (list mapping old->new)."""
    n = len(grammar)
    assert n <= len(perm), "perm must cover all nonterminals in grammar"
    if n < len(perm):  # pad grammar with empty rules if perm is longer
        grammar = grammar + [[] for _ in range(len(perm) - n)]
        n = len(grammar)

    if is_new_to_old:
        # Invert perm to get old->new
        inv_perm = [0] * n
        for new_i, old_i in enumerate(perm):
            inv_perm[old_i] = new_i
        perm = inv_perm

    # Row permutation: new_rules[new_i] = old_rules[old_i] where new_i = perm[old_i]
    new_rules = [None] * n
    # Build LUT for symbols: j -> perm[j]
    lut = {j: perm[j] for j in range(n)}
    for old_i, rule in enumerate(grammar):
        new_i = perm[old_i]
        new_rule = []
        for s in rule:
            if s >= 0 and s < n:
                new_rule.append(lut[s])
            else:
                new_rule.append(s)
        new_rules[new_i] = new_rule
    return new_rules


# -------------- Demo & unit tests --------------


def _perturb_grammar(
    grammar: list[list[int]],
    dropout_prob: float = 0.0,
    shift_range: tuple[int, int] = (0, 1),
    permute: bool = True,
    rng: random.Random = random.Random(0),
) -> list[list[int]]:
    """Return a new grammar with some random perturbations: dropout of symbols and sequence shifts."""
    terminals = set()
    for rule in grammar:
        for s in rule:
            if s < 0:
                terminals.add(s)
    terminals = list(terminals)

    new_grammar = []
    for rule in grammar:
        rule = rule.copy()

        # Shift
        shift = min(rng.randint(shift_range[0], shift_range[1]), len(rule) - 1)
        if shift > 0 and rng.random() < 0.5:  # left side
            if rng.random() < 0.5:  # cut
                rule = rule[shift:]
            else:  # pad
                rule = [random.choice(terminals) for _ in range(shift)] + rule
        elif shift > 0:  # right side
            if rng.random() < 0.5:  # cut
                rule = rule[:-shift]
            else:  # pad
                rule = rule + [random.choice(terminals) for _ in range(shift)]

        # Dropout
        rule = [s for s in rule if rng.random() > dropout_prob]

        if len(rule) > 0:
            new_grammar.append(rule)

    if permute:
        n = len(new_grammar)
        perm = list(range(n))
        rng.shuffle(perm)
        new_grammar = _permute_grammar(new_grammar, perm)

    return new_grammar


if __name__ == "__main__":
    # Basic sanity tests
    compressor = DeterministicGrammarPatternMatchingConditionalCompressor()

    # Example from the prompt, but batched
    # g_target:
    # 0 -> 1 1 2 1
    # 1 -> 3 2
    # 2 -> -1 -2
    # 3 -> -4 -3
    g_target = [[[1, 1, 2, 1], [3, 2], [-1, -2], [-4, -3]]]

    # A slightly permuted version of g_target
    # 0 -> -4 -3       (was rule 3)
    # 1 -> -1 -2       (was rule 2)
    # 2 -> 3 3 1 3       (was rule 0, with 1->3, 2->1)
    # 3 -> 0 1           (was rule 1, with 3->0, 2->1)
    g_given1 = [
        [[-4, -3], [-1, -2], [3, 3, 1, 3], [0, 1]],
    ]

    # A grammar with different structure
    g_given2 = [
        [[-10, -11, -12], [0, 0]],
    ]

    # A completely identical grammar
    g_identical = [
        [[1, 1, 2, 1], [3, 2], [-1, -2], [-4, -3]],
    ]

    print(
        "### Test Case 1: Compressing a grammar given a permuted version and a different one."
    )
    # For g_given1, the alignment should be found, and conditional cost should be 0.
    # For g_given2, the conditional cost will be high, so it shouldn't be used.
    cost1 = compressor.compress(g_target, [g_given1, g_given2])
    print(f"Compression cost: {cost1.item():.2f} bits\n")

    print("### Test Case 2: Compressing a grammar given itself.")
    # Conditional cost should be 0.
    cost2 = compressor.compress(g_target, [g_identical])
    print(f"Compression cost (given identical): {cost2.item():.2f} bits\n")

    print("### Test Case 3: Compressing a grammar given only the different one.")
    # Both conditional and raw compression costs should be large.
    raw_cost_direct = compressor.compress(g_target, [])
    cost3 = compressor.compress(g_target, [g_given2])
    print(f"Raw cost: {raw_cost_direct.item():.2f} bits")
    print(f"Compression cost (given different): {cost3.item():.2f} bits\n")

    print("### Test Case 4: Batched execution with two different targets.")
    g_target_batch = [
        g_target[0],
        g_given2[0],
    ]  # A batch containing g_target and g_given2
    g_given_batch = [
        g_given1[0],
        g_identical[0],
    ]  # g_given1 for the first, g_identical for the second
    costs_batch = compressor.compress(g_target_batch, [g_given_batch])
    print(f"Batched compression costs: {costs_batch.tolist()} bits\n")

    print("\n### Test Case 5: Increasing target complexity.")
    # Conditional costs should get increasingly large.
    # g_target:
    # 0 -> 1 2 3
    # 1 -> 2 4
    # 2 -> 3 4
    # 3 -> -1 -1
    # 4 -> -2 -2
    g_target = [
        [1, 2, 3],
        [2, 4],
        [3, 4],
        [-1, -1],
        [-2, -2],
    ]
    g_given = [
        g_target,
        [
            [1, 3],
            [2, 3],
            [-1, -1],
            [-2, -2],
        ],
        [
            [1, 2],
            [-1, -1],
            [-2, -2],
        ],
    ]
    cost5 = compressor.compress([g_target] * 3, [g_given])
    print(f"Compression cost (increasing complexity): {cost5.tolist()} bits\n")

    print("### Test Case 6: Target as union of two given grammars.")
    # Conditional costs should be 0, for both concatenation and overlap union.
    # g_target:
    # 0 -> 1 1 2 1
    # 1 -> -1 -2 -3
    # 2 -> -4 -3 -4
    # 3 -> 4 5
    # 4 -> -5 -6
    # 5 -> -7 -8 -9
    g_target = [
        [
            [1, 1, 2, 1],
            [-1, -2, -3],
            [-4, -3, -4],
            [4, 5],
            [-5, -6],
            [-7, -8, -9],
        ],
        [
            [2, 2, 3, 2],
            [3, 3, 2, 3],
            [-1, -2, -3],
            [-7, -8, -9],
        ],
    ]
    g_given_a = [
        [
            [1, 1, 2, 1],
            [-1, -2, -3],
            [-4, -3, -4],
        ],
        [
            [1, 1, 2, 1],
            [-1, -2, -3],
            [-7, -8, -9],
        ],
    ]
    g_given_b = [
        [
            [1, 2],
            [-5, -6],
            [-7, -8, -9],
        ],
        [
            [2, 2, 1, 2],
            [-1, -2, -3],
            [-7, -8, -9],
        ],
    ]
    cost6 = compressor.compress(g_target, [g_given_a, g_given_b])
    print(f"Compression cost (target as union): {cost6.tolist()} bits\n")

    print("### Test Case 7: Given grammar has extra rules compared to target.")
    # Conditional cost should still be 0.
    # g_target:
    # 0 -> 1 2
    # 1 -> 3 3 3
    # 2 -> -1 -1 -1
    # 3 -> -2 b'
    g_target = [
        [1, 2],
        [3, 3, 3],
        [-1, -1, -1],
        [-2, -2],
    ]
    g_given = g_target + [
        [3, 1],
        [6, 2],
        [-3, -3],
    ]
    cost7 = compressor.compress([g_target], [[g_given]])
    print(f"Compression cost (given has extra rules): {cost7.item():.2f} bits\n")

    print(
        "### Test Case 8: Large grammar to test non-exhaustive search, and large batch size for speed."
    )
    # Create a large grammar with 10 rules (> max_nonterminals_for_exhaustive_search = 6)
    bs = 100
    g_target = [
        [1, 2, 3],
        [2, 4, 5, 4, 3],
        [4, 3, 5],
        [5, 4, 3],
        [6, 5, 4, 2, 3, 3, 3],
        [-1, -1, -1],
        [-2, -2, -2],
        [-3, -3, -3],
        [-4, -4, -4],
        [-5, -5, -5, -6, -6, -6],
    ]
    g_given = [
        [-5, -5, -5, -6, -6, -6],
        [-4, -4, -4],
        [-3, -3, -3],
        [-2, -2, -2],
        [-1, -1, -1],
        [3, 4, 5, 7, 6, 6, 6],
        [4, 5, 6],
        [5, 6, 4],
        [7, 5, 4, 5, 6],
        [8, 7, 6],
    ]
    raw_cost8 = compressor.compress([g_target], [])
    start_time = time.time()
    cost8 = compressor.compress([g_target] * bs, [[g_given] * bs] * 2)
    end_time = time.time()
    print(
        f"Performance test (batch_size={bs}) completed in {end_time - start_time:.3f} seconds"
    )
    print(f"Raw cost (large grammar): {raw_cost8.item():.2f} bits")
    print(f"Compression avg. cost (large grammar): {cost8.mean().item():.2f} bits\n")

    print("### Test 9: Much larger grammars (N=40 rules)")
    rng = random.Random(42)
    N = 40
    TERMS = [-1, -2, -3, -4, -5, -6, -7]
    g_big = []
    for i in range(N):
        # each rule: 3-6 symbols, mix of terminals and nonterminals
        L = rng.randint(3, 6)
        rule = []
        for _ in range(L):
            if rng.random() < 0.6 and N > 1:
                rule.append(rng.randint(0, N - 1))
            else:
                rule.append(rng.choice(TERMS))
        g_big.append(rule)
    perm_big = list(range(N))
    rng.shuffle(perm_big)
    g_big_perm = _permute_grammar(g_big, perm_big)

    t0 = time.time()
    c_big_raw = compressor.compress([g_big], [])
    c_big = compressor.compress([g_big], [[g_big_perm]])
    t1 = time.time()
    print(
        f"Raw: {float(c_big_raw.item()):.2f}\nConditional: {float(c_big.item()):.2f} | time {t1-t0:.3f}s\n"
    )

    print(
        "### Test 10: Large grammars (N=40 rules) with only terminal-generating rules"
    )
    rng = random.Random(42)
    N = 40
    TERMS = [-1, -2, -3, -4, -5, -6, -7]
    g_big = []
    for i in range(N):
        # each rule: 10-16 symbols, mix of terminals and nonterminals
        L = rng.randint(10, 16)
        rule = []
        for _ in range(L):
            rule.append(rng.choice(TERMS))
        g_big.append(rule)
    perm_big = list(range(N))
    rng.shuffle(perm_big)
    g_big_perm = _permute_grammar(g_big, perm_big)

    t0 = time.time()
    c_big_raw = compressor.compress([g_big], [])
    c_big = compressor.compress([g_big], [[g_big_perm]])
    t1 = time.time()
    print(
        f"Raw: {float(c_big_raw.item()):.2f}\nConditional: {float(c_big.item()):.2f} | time {t1-t0:.3f}s\n"
    )

    print("### Test 11: Perturbations of a small base grammar")
    g_target = [
        [1, 1, 2, 1, 2, 1],
        [3, 2, 3, 1, 1],
        [-1, -2, -3, -3, -3, -2, -1, -2, -2],
        [-2, -3, -2, -1, -3, -3, -3, -2],
    ]
    g_given = _perturb_grammar(g_target, rng=rng)
    raw_cost = compressor.compress([g_target], [])
    cond_cost = compressor.compress([g_target], [[g_given]])
    print(
        f"Raw: {float(raw_cost.item()):.2f}\nConditional: {float(cond_cost.item()):.2f}\n"
    )

    print("### Test 12: Perturbations of a large base grammar")
    g_target = [
        [1, 2, 3, 2, 1, 3, 1],
        [2, 4, 5, 4, 3, 4, 2, 3, 5],
        [4, 3, 5, 2, 1, 4, 2, 4, 2],
        [5, 4, 3, 2, 4, 1, 1, 3, 5, 3],
        [6, 5, 4, 2, 3, 3, 3, 1, 6],
        [-1, -1, -1, -1, -2, -2, -1, -1],
        [-2, -2, -2, -2, -3, -3, -2, -2],
        [-3, -3, -3, -3, -4, -5, -6],
        [-4, -4, -4, -4, -4, -4, -4],
        [-5, -5, -5, -6, -6, -6, -5, -5],
    ]
    g_given = _perturb_grammar(g_target, rng=rng)
    raw_cost = compressor.compress([g_target], [])
    cond_cost = compressor.compress([g_target], [[g_given]])
    print(
        f"Raw: {float(raw_cost.item()):.2f}\nConditional: {float(cond_cost.item()):.2f}\n"
    )

    print("### Test 13: Significant (dropout) perturbations of a large base grammar")
    g_given = _perturb_grammar(g_target, dropout_prob=0.5, rng=rng)
    raw_cost = compressor.compress([g_target], [])
    cond_cost = compressor.compress([g_target], [[g_given]])
    print(
        f"Raw: {float(raw_cost.item()):.2f}\nConditional: {float(cond_cost.item()):.2f}\n"
    )
