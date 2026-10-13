from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import heapq, math
import time

Symbol = int
RHS = Tuple[Symbol, ...]


def _log2(x: float) -> float:
    return math.log2(x) if x > 0 else 0.0


@dataclass(slots=True)
class Tape:
    sym: List[int]
    prev: List[int]
    next: List[int]
    alive: List[bool]
    pos: List[int]
    head: int
    free_list: List[int]


def build_tape(seq_ids: List[int]) -> Tape:
    n = len(seq_ids)
    sym = list(seq_ids)
    prev = [-1] * n
    next_ = [-1] * n
    alive = [True] * n
    pos = list(range(n))
    for i in range(n - 1):
        next_[i] = i + 1
        prev[i + 1] = i
    head = 0 if n > 0 else -1
    return Tape(sym, prev, next_, alive, pos, head, [])


def new_node(t: Tape, sym_val: int, left: int, right: int, pos_val: int) -> int:
    if t.free_list:
        i = t.free_list.pop()
        t.sym[i] = sym_val
        t.prev[i] = left
        t.next[i] = right
        t.alive[i] = True
        t.pos[i] = pos_val
        return i
    else:
        i = len(t.sym)
        t.sym.append(sym_val)
        t.prev.append(left)
        t.next.append(right)
        t.alive.append(True)
        t.pos.append(pos_val)
        return i


def kill_node(t: Tape, i: int):
    t.alive[i] = False
    t.prev[i] = -1
    t.next[i] = -1
    t.free_list.append(i)


@dataclass(slots=True)
class PairIndex:
    pair_lefts: Dict[Tuple[int, int], Set[int]]
    heap: List[Tuple[int, Tuple[int, int]]]
    first_pos: Dict[Tuple[int, int], int]


def pair_index_from_tape(t: Tape) -> PairIndex:
    pair_lefts: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    first_pos: Dict[Tuple[int, int], int] = {}
    i = t.head
    while i != -1:
        j = t.next[i]
        if j != -1 and t.alive[i] and t.alive[j]:
            p = (t.sym[i], t.sym[j])
            pair_lefts[p].add(i)
            if p not in first_pos or t.pos[i] < first_pos[p]:
                first_pos[p] = t.pos[i]
        i = t.next[i]
    heap = [(-len(v), k) for k, v in pair_lefts.items()]
    heapq.heapify(heap)
    return PairIndex(pair_lefts, heap, first_pos)


def remove_occ_at(t: Tape, P: PairIndex, left_idx: int):
    if left_idx == -1 or not t.alive[left_idx]:
        return
    right = t.next[left_idx]
    if right == -1 or not t.alive[right]:
        return
    p = (t.sym[left_idx], t.sym[right])
    s = P.pair_lefts.get(p)
    if s is None:
        return
    if left_idx in s:
        s.remove(left_idx)
        if not s:
            del P.pair_lefts[p]
            if p in P.first_pos:
                del P.first_pos[p]
        else:
            if P.first_pos.get(p, None) == t.pos[left_idx]:
                P.first_pos[p] = min(t.pos[x] for x in s)
            heapq.heappush(P.heap, (-len(s), p))


def add_occ_at(t: Tape, P: PairIndex, left_idx: int):
    if left_idx == -1 or not t.alive[left_idx]:
        return
    right = t.next[left_idx]
    if right == -1 or not t.alive[right]:
        return
    p = (t.sym[left_idx], t.sym[right])
    s = P.pair_lefts.get(p)
    if s is None:
        s = set()
        P.pair_lefts[p] = s
    if left_idx not in s:
        s.add(left_idx)
        fp = P.first_pos.get(p)
        if fp is None or t.pos[left_idx] < fp:
            P.first_pos[p] = t.pos[left_idx]
        heapq.heappush(P.heap, (-len(s), p))


def valid_top_pair(P: PairIndex) -> Optional[Tuple[Tuple[int, int], int]]:
    while P.heap:
        negf, p = heapq.heappop(P.heap)
        f = -negf
        cur = len(P.pair_lefts.get(p, ()))
        if cur == f and f > 1:
            return p, f
    return None


def iter_pairs_scan_order(t: Tape, P: PairIndex):
    seen: Set[Tuple[int, int]] = set()
    i = t.head
    while i != -1:
        j = t.next[i]
        if j != -1 and t.alive[i] and t.alive[j]:
            p = (t.sym[i], t.sym[j])
            if p not in seen and p in P.pair_lefts and P.pair_lefts[p]:
                seen.add(p)
                yield p, P.pair_lefts[p]
        i = t.next[i]


def extend_to_maximal_repeat_indices(
    t: Tape, left_nodes: List[int]
) -> Tuple[List[int], List[int]]:
    starts = [i for i in left_nodes]
    ends = [t.next[i] for i in left_nodes]
    changed = True
    while changed:
        changed = False
        while True:
            first = None
            ok = True
            for s in starts:
                p = t.prev[s]
                if p == -1 or not t.alive[p]:
                    ok = False
                    break
                if first is None:
                    first = t.sym[p]
                elif t.sym[p] != first:
                    ok = False
                    break
            if not ok:
                break
            for i in range(len(starts)):
                starts[i] = t.prev[starts[i]]
            changed = True
        while True:
            first = None
            ok = True
            for e in ends:
                n = t.next[e]
                if n == -1 or not t.alive[n]:
                    ok = False
                    break
                if first is None:
                    first = t.sym[n]
                elif t.sym[n] != first:
                    ok = False
                    break
            if not ok:
                break
            for i in range(len(ends)):
                ends[i] = t.next[ends[i]]
            changed = True
    return starts, ends


def drop_first_if_needed_indices(
    t: Tape, starts: List[int], ends: List[int]
) -> Tuple[List[int], List[int]]:
    if not starts:
        return starts, ends
    s0, e0 = starts[0], ends[0]
    length = 1
    cur = s0
    while cur != e0:
        cur = t.next[cur]
        length += 1
    if length > 2 and t.sym[s0] == t.sym[e0]:
        for i in range(len(ends)):
            ends[i] = t.prev[ends[i]]
    return starts, ends


def extract_pattern_from_span_indices(t: Tape, start: int, end: int) -> Tuple[int, ...]:
    pat = []
    cur = start
    while True:
        pat.append(t.sym[cur])
        if cur == end:
            break
        cur = t.next[cur]
    return tuple(pat)


def count_nonoverlap_pairs(t: Tape, lefts: List[int]) -> int:
    if not lefts:
        return 0
    occs = sorted((t.pos[i], i, t.next[i]) for i in lefts)
    chosen = 0
    last_end_pos = -1
    for sp, s, e in occs:
        ep = t.pos[e]
        if sp > last_end_pos:
            chosen += 1
            last_end_pos = ep
    return chosen


def count_nonoverlap_from_spans_indices(
    t: Tape, starts: List[int], ends: List[int]
) -> Tuple[int, List[Tuple[int, int]]]:
    if not starts:
        return 0, []
    occs = sorted(
        ((t.pos[starts[i]], starts[i], ends[i]) for i in range(len(starts))),
        key=lambda x: x[0],
    )
    chosen: List[Tuple[int, int]] = []
    last_end_pos = -1
    for sp, s, e in occs:
        ep = t.pos[e]
        if sp > last_end_pos:
            chosen.append((s, e))
            last_end_pos = ep
    return len(chosen), chosen


def replace_non_overlapping_indices(
    t: Tape, P: PairIndex, starts: List[int], ends: List[int], new_sym: int
) -> Tuple[int, int]:
    if not starts:
        return -1, 0
    occs = sorted(
        ((t.pos[starts[i]], starts[i], ends[i]) for i in range(len(starts))),
        key=lambda x: x[0],
    )
    replaced = 0
    first_new = -1
    for sp, s, e in occs:
        if not t.alive[s] or not t.alive[e]:
            continue
        before = t.prev[s]
        after = t.next[e]
        if before != -1 and t.alive[before] and t.next[before] == s:
            remove_occ_at(t, P, before)
        cur = s
        while True:
            remove_occ_at(t, P, cur)
            if cur == e:
                break
            cur = t.next[cur]
        new_idx = new_node(t, new_sym, before, after, t.pos[s])
        if before != -1 and t.alive[before]:
            t.next[before] = new_idx
        else:
            t.head = new_idx
        if after != -1 and t.alive[after]:
            t.prev[after] = new_idx
        cur = s
        while True:
            nxt = t.next[cur]
            kill_node(t, cur)
            if cur == e:
                break
            cur = nxt
        if before != -1 and t.alive[before]:
            add_occ_at(t, P, before)
        add_occ_at(t, P, new_idx)
        if first_new == -1:
            first_new = new_idx
        replaced += 1
    return first_new, replaced


def tape_iter_alive(t: Tape):
    i = t.head
    while i != -1:
        if t.alive[i]:
            yield i
        i = t.next[i]


def c_logc(c: int) -> float:
    return c * _log2(c) if c > 0 else 0.0


def sequence_only_gain_delta(
    pattern: Tuple[int, ...], k: int, counts_seq: Dict[int, int], N: int, new_sym: int
) -> float:
    m = len(pattern)
    if m < 2 or k <= 1:
        return float("-inf")
    Np = N - k * (m - 1)
    base_term = N * _log2(N) - Np * _log2(Np)
    delta_counts = 0.0
    pc = Counter(pattern)
    for s, tms in pc.items():
        c = counts_seq.get(s, 0)
        c2 = c - k * tms
        if c2 < 0:
            c2 = 0
        delta_counts += c_logc(c) - c_logc(c2)
    cA = counts_seq.get(new_sym, 0)
    delta_counts += c_logc(cA) - c_logc(cA + k)
    seq_savings = base_term - delta_counts
    L_rhs = sum((_log2(N) - _log2(counts_seq.get(s, 1))) for s in pattern)
    L_lhs = _log2(N) - _log2(counts_seq.get(new_sym, 1))
    return seq_savings - (L_rhs + L_lhs)


def global_static_gain_delta(
    pattern: Tuple[int, ...],
    k: int,
    counts_seq: Dict[int, int],
    counts_gram: Dict[int, int],
    new_sym: int,
) -> float:
    m = len(pattern)
    if m < 2 or k <= 1:
        return float("-inf")
    pc = Counter(pattern)
    T = sum(counts_seq.values()) + sum(counts_gram.values())
    dT = (m + 1) - k * (m - 1)
    Tp = T + dT
    base_term = T * _log2(T) - Tp * _log2(Tp)
    delta_counts = 0.0
    for s, tms in pc.items():
        C = counts_seq.get(s, 0) + counts_gram.get(s, 0)
        C2 = C + tms * (1 - k)
        if C2 < 0:
            C2 = 0
        delta_counts += c_logc(C) - c_logc(C2)
    CA = counts_seq.get(new_sym, 0) + counts_gram.get(new_sym, 0)
    delta_counts += c_logc(CA) - c_logc(CA + k + 1)
    return base_term - delta_counts


def collect_top_k_unique(P: PairIndex, K: int) -> List[Tuple[Tuple[int, int], int]]:
    pulled: List[Tuple[Tuple[int, int], int]] = []
    seen: Set[Tuple[int, int]] = set()
    while len(pulled) < K:
        v = valid_top_pair(P)
        if not v:
            break
        p, f = v
        if p in seen:
            continue
        seen.add(p)
        pulled.append((p, f))
    for p, _ in pulled:
        s = P.pair_lefts.get(p, None)
        if s:
            heapq.heappush(P.heap, (-len(s), p))
    return pulled


def suffix_array_int(seq: List[int]) -> List[int]:
    n = len(seq)
    if n == 0:
        return []
    vals = sorted(set(seq))
    rank_map = {v: i for i, v in enumerate(vals)}
    rank = [rank_map[x] for x in seq]
    sa = list(range(n))
    tmp = [0] * n
    k = 1
    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        tmp[sa[0]] = 0
        for idx in range(1, n):
            i_prev = sa[idx - 1]
            i_cur = sa[idx]
            prev_key = (rank[i_prev], rank[i_prev + k] if i_prev + k < n else -1)
            cur_key = (rank[i_cur], rank[i_cur + k] if i_cur + k < n else -1)
            tmp[i_cur] = tmp[i_prev] + (1 if cur_key != prev_key else 0)
        rank, tmp = tmp, rank
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1
    return sa


def kasai_lcp(seq: List[int], sa: List[int]) -> List[int]:
    n = len(seq)
    if n == 0:
        return []
    inv = [0] * n
    for i, sidx in enumerate(sa):
        inv[sidx] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = inv[i]
        if r == 0:
            h = 0
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and seq[i + h] == seq[j + h]:
            h += 1
        lcp[r] = h
        if h > 0:
            h -= 1
    return lcp


def top_lcp_intervals(
    lcp: List[int], min_len: int, limit: int
) -> List[Tuple[float, int, int, int, int]]:
    n = len(lcp)
    if n <= 1 or limit <= 0:
        return []

    prev_smaller = [0] * n
    stack: List[int] = []
    for i in range(1, n):
        while stack and lcp[stack[-1]] >= lcp[i]:
            stack.pop()
        prev_smaller[i] = stack[-1] if stack else 0
        stack.append(i)

    next_smaller = [n] * n
    stack = []
    for i in range(n - 1, 0, -1):
        while stack and lcp[stack[-1]] > lcp[i]:
            stack.pop()
        next_smaller[i] = stack[-1] if stack else n
        stack.append(i)

    heap: List[Tuple[float, int, int, int, int]] = []
    seen: Set[Tuple[int, int, int]] = set()
    for i in range(1, n):
        h = lcp[i]
        if h < min_len:
            continue
        left = prev_smaller[i] + 1
        right = next_smaller[i] - 1
        key = (left, right, h)
        if key in seen:
            continue
        seen.add(key)
        occ = right - left + 2
        score = float(h * (occ - 1))
        item = (score, h, i, left, right)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True)


def kmp_prefix_table(pattern: List[int]) -> List[int]:
    lps = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
    return lps


def kmp_find_all(text: List[int], pattern: List[int]) -> List[int]:
    if not pattern or len(pattern) > len(text):
        return []
    lps = kmp_prefix_table(pattern)
    out: List[int] = []
    j = 0
    m = len(pattern)
    for i, val in enumerate(text):
        while j > 0 and val != pattern[j]:
            j = lps[j - 1]
        if val == pattern[j]:
            j += 1
            if j == m:
                out.append(i - m + 1)
                j = lps[j - 1]
    return out


def nonoverlap_starts(starts: List[int], pat_len: int) -> List[int]:
    chosen: List[int] = []
    last_end = -1
    for s in starts:
        if s > last_end:
            chosen.append(s)
            last_end = s + pat_len - 1
    return chosen


def sa_repeat_candidates(
    t: Tape,
    min_len: int,
    max_len: int,
    top_k_patterns: int,
) -> List[Tuple[Tuple[int, ...], List[int], List[int], int, int]]:
    if top_k_patterns <= 0:
        return []
    alive_nodes = list(tape_iter_alive(t))
    seq = [t.sym[i] for i in alive_nodes]
    n = len(seq)
    if n < 2 * max(2, min_len):
        return []
    max_len = min(max_len, n)
    if max_len < min_len:
        return []

    sa = suffix_array_int(seq)
    lcp = kasai_lcp(seq, sa)
    interval_limit = max(top_k_patterns * 8, 16)
    intervals = top_lcp_intervals(lcp, min_len=min_len, limit=interval_limit)
    if not intervals:
        return []

    candidates: List[Tuple[Tuple[int, ...], List[int], List[int], int, int]] = []
    seen_patterns: Set[Tuple[int, ...]] = set()
    for _score, lcp_len, sa_idx, _l, _r in intervals:
        m = min(lcp_len, max_len)
        if m < min_len:
            continue
        start = sa[sa_idx]
        if start + m > n:
            continue
        pattern = tuple(seq[start : start + m])
        if pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        start_offsets = kmp_find_all(seq, list(pattern))
        if len(start_offsets) <= 1:
            continue
        k = len(nonoverlap_starts(start_offsets, m))
        if k <= 1:
            continue
        starts = [alive_nodes[s] for s in start_offsets]
        ends = [alive_nodes[s + m - 1] for s in start_offsets]
        best_pos = t.pos[starts[0]] if starts else (1 << 60)
        candidates.append((pattern, starts, ends, k, best_pos))
    return candidates


def mr_repair_encode(
    input_text: Union[str, List[int]],
    mdl: bool = False,
    mdl_mode: str = "global_static",
    mdl_lazy_top_k: int = 0,
    mdl_sa_top_k: int = 24,
    mdl_sa_every: int = 4,
    mdl_sa_min_len: int = 2,
    mdl_sa_max_len: int = 256,
) -> Tuple[Dict[Symbol, RHS], RHS, Dict[str, Any]]:
    """Encode a sequence into a grammar and start rule using MR-RePair.

    Args:
        input_text: Input token sequence.
        mdl: If True, accept candidate rules only when MDL gain is positive.
        mdl_mode: MDL approximation mode (`global_static` or `sequence_only`).
        mdl_lazy_top_k: If > 0, evaluate only the top-k frequent pairs per
            iteration. This speeds up pair scanning at possible quality loss.
        mdl_sa_top_k: Number of global repeat candidates proposed by suffix-array
            mining when triggered. Higher values improve robustness to non-local
            repeats, but increase per-trigger matching cost. 0 disables suffix-array mining.
        mdl_sa_every: Run suffix-array mining every this many accepted rules
            (plus always when pair-only search stalls). Smaller values improve
            candidate quality but spend more time in O(n log n) mining.
        mdl_sa_min_len: Minimum candidate length from suffix-array mining.
            Larger values reduce noise and speed up matching, but may miss short
            useful repeats.
        mdl_sa_max_len: Maximum candidate length from suffix-array mining.
            Larger values capture longer motifs, but increase matching cost.
    """
    term2id: Dict[Any, int] = {}
    id2term: Dict[int, Any] = {}
    seq_ids: List[int] = []
    next_id = 0
    for tkn in input_text:
        if tkn not in term2id:
            term2id[tkn] = next_id
            id2term[next_id] = tkn
            next_id += 1
        seq_ids.append(term2id[tkn])
    t = build_tape(seq_ids)
    P = pair_index_from_tape(t)
    grammar: Dict[int, Tuple[int, ...]] = {}
    var_id = next_id

    if not mdl:
        while True:
            v = valid_top_pair(P)
            if not v:
                break
            (a, b), freq = v
            if freq <= 1:
                break
            lefts = [
                i
                for i in P.pair_lefts.get((a, b), set())
                if t.alive[i]
                and t.next[i] != -1
                and t.alive[t.next[i]]
                and t.sym[i] == a
                and t.sym[t.next[i]] == b
            ]
            if len(lefts) <= 1:
                continue
            starts, ends = extend_to_maximal_repeat_indices(t, lefts)
            starts, ends = drop_first_if_needed_indices(t, starts, ends)
            if not starts:
                continue
            pattern = extract_pattern_from_span_indices(t, starts[0], ends[0])
            if len(pattern) == 0:
                break
            grammar[var_id] = pattern
            _, replaced = replace_non_overlapping_indices(t, P, starts, ends, var_id)
            if replaced == 0:
                break
            var_id += 1
    else:
        counts_seq = Counter(t.sym[i] for i in tape_iter_alive(t))
        N = sum(counts_seq.values())
        counts_gram = Counter()
        applied_rules = 0

        while True:
            best_gain = float("-inf")
            best_pack = None
            best_pos = None

            if mdl_lazy_top_k > 0:
                cand_pairs = collect_top_k_unique(P, mdl_lazy_top_k)
                for (a, b), _f in cand_pairs:
                    lefts_set = P.pair_lefts.get((a, b), set())
                    if not lefts_set:
                        continue
                    lefts = [
                        i
                        for i in lefts_set
                        if t.alive[i]
                        and t.next[i] != -1
                        and t.alive[t.next[i]]
                        and t.sym[i] == a
                        and t.sym[t.next[i]] == b
                    ]
                    if len(lefts) <= 1:
                        continue
                    k_pair = count_nonoverlap_pairs(t, lefts)
                    if k_pair <= 1:
                        continue
                    starts, ends = extend_to_maximal_repeat_indices(t, lefts)
                    starts, ends = drop_first_if_needed_indices(t, starts, ends)
                    if not starts:
                        continue
                    k, _chosen = count_nonoverlap_from_spans_indices(t, starts, ends)
                    if k <= 1:
                        continue
                    pattern = extract_pattern_from_span_indices(t, starts[0], ends[0])
                    if len(pattern) < 2:
                        continue
                    gain = (
                        global_static_gain_delta(
                            pattern, k, counts_seq, counts_gram, var_id
                        )
                        if mdl_mode == "global_static"
                        else sequence_only_gain_delta(pattern, k, counts_seq, N, var_id)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_pack = (pattern, starts, ends, k)
                        best_pos = P.first_pos.get((a, b), 1 << 60)
                    elif gain == best_gain:
                        fp = P.first_pos.get((a, b), 1 << 60)
                        if best_pos is None or fp < best_pos:
                            best_pack = (pattern, starts, ends, k)
                            best_pos = fp
            else:
                for (a, b), lefts_set in iter_pairs_scan_order(t, P):
                    lefts = [
                        i
                        for i in lefts_set
                        if t.alive[i]
                        and t.next[i] != -1
                        and t.alive[t.next[i]]
                        and t.sym[i] == a
                        and t.sym[t.next[i]] == b
                    ]
                    if len(lefts) <= 1:
                        continue
                    k_pair = count_nonoverlap_pairs(t, lefts)
                    if k_pair <= 1:
                        continue
                    starts, ends = extend_to_maximal_repeat_indices(t, lefts)
                    starts, ends = drop_first_if_needed_indices(t, starts, ends)
                    if not starts:
                        continue
                    k, _chosen = count_nonoverlap_from_spans_indices(t, starts, ends)
                    if k <= 1:
                        continue
                    pattern = extract_pattern_from_span_indices(t, starts[0], ends[0])
                    if len(pattern) < 2:
                        continue
                    gain = (
                        global_static_gain_delta(
                            pattern, k, counts_seq, counts_gram, var_id
                        )
                        if mdl_mode == "global_static"
                        else sequence_only_gain_delta(pattern, k, counts_seq, N, var_id)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_pack = (pattern, starts, ends, k)
                        best_pos = P.first_pos.get((a, b), 1 << 60)
                    # ties broken implicitly by scan order

            run_sa_search = mdl_sa_top_k > 0 and (
                best_pack is None
                or best_gain <= 0
                or (mdl_sa_every > 0 and ((applied_rules + 1) % mdl_sa_every == 0))
            )
            if run_sa_search:
                sa_cands = sa_repeat_candidates(
                    t=t,
                    min_len=max(2, mdl_sa_min_len),
                    max_len=max(2, mdl_sa_max_len),
                    top_k_patterns=mdl_sa_top_k,
                )
                for pattern, starts, ends, k, first_pos in sa_cands:
                    gain = (
                        global_static_gain_delta(
                            pattern, k, counts_seq, counts_gram, var_id
                        )
                        if mdl_mode == "global_static"
                        else sequence_only_gain_delta(pattern, k, counts_seq, N, var_id)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_pack = (pattern, starts, ends, k)
                        best_pos = first_pos
                    elif gain == best_gain and (
                        best_pos is None or first_pos < best_pos
                    ):
                        best_pack = (pattern, starts, ends, k)
                        best_pos = first_pos

            if not best_pack or best_gain <= 0:
                break
            pattern, starts, ends, k_est = best_pack
            grammar[var_id] = pattern
            _, k_applied = replace_non_overlapping_indices(t, P, starts, ends, var_id)
            pc = Counter(pattern)
            for s, times in pc.items():
                counts_seq[s] -= k_applied * times
                if counts_seq[s] < 0:
                    counts_seq[s] = 0
            counts_seq[var_id] += k_applied
            N -= k_applied * (len(pattern) - 1)
            if mdl_mode == "global_static":
                counts_gram.update(pc)
                counts_gram[var_id] += 1
            var_id += 1
            applied_rules += 1

    start_list: List[int] = []
    i = t.head
    while i != -1:
        if t.alive[i]:
            start_list.append(t.sym[i])
        i = t.next[i]
    start_rule = tuple(start_list)
    vocab = {"term2id": term2id, "id2term": id2term, "next_id": var_id}
    return grammar, start_rule, vocab


def mr_repair_decode(
    grammar: Dict[Symbol, RHS], start_rule: RHS, id2term: Dict[int, Any]
) -> Union[str, List[int]]:
    memo: Dict[int, List[int]] = {}

    def expand_symbol(s: int) -> List[int]:
        if s in memo:
            return memo[s]
        if s in grammar:
            out: List[int] = []
            for x in grammar[s]:
                out.extend(expand_symbol(x))
            memo[s] = out
            return out
        else:
            memo[s] = [s]
            return memo[s]

    expanded: List[int] = []
    for s in start_rule:
        expanded.extend(expand_symbol(s))
    if id2term and all(
        isinstance(id2term.get(k), str) and len(id2term.get(k, "")) == 1
        for k in id2term
    ):
        return "".join(id2term[i] for i in expanded)
    else:
        return expanded


@dataclass(slots=True)
class RePairCase:
    name: str
    sequence: List[int]
    ground_truth_grammar: Dict[int, RHS]
    ground_truth_start: RHS
    structured: bool


def _expand_from_grammar(grammar: Dict[int, RHS], start_rule: RHS) -> List[int]:
    memo: Dict[int, List[int]] = {}

    def expand_symbol(sym: int) -> List[int]:
        if sym in memo:
            return memo[sym]
        if sym in grammar:
            out: List[int] = []
            for x in grammar[sym]:
                out.extend(expand_symbol(x))
            memo[sym] = out
            return out
        return [sym]

    out: List[int] = []
    for s in start_rule:
        out.extend(expand_symbol(s))
    return out


def _two_part_code_length(grammar: Dict[int, RHS], start_rule: RHS) -> float:
    grammar_stream = [sym for rule in grammar.values() for sym in rule]
    lhs_stream = list(grammar.keys())
    all_stream = grammar_stream + list(start_rule) + lhs_stream
    if not all_stream:
        return 0.0
    counts = Counter(all_stream)
    total = float(sum(counts.values()))
    bits = {sym: -_log2(cnt / total) for sym, cnt in counts.items()}
    model = sum(bits[sym] for sym in grammar_stream) + sum(
        bits[sym] for sym in lhs_stream
    )
    data = sum(bits[sym] for sym in start_rule)
    return model + data


def _decode_to_input_space(
    grammar: Dict[int, RHS], start_rule: RHS, vocab: Dict[str, Any]
) -> Union[str, List[int]]:
    out = mr_repair_decode(grammar, start_rule, vocab["id2term"])
    if isinstance(out, str):
        return out
    if out and out[0] in vocab["id2term"]:
        return [vocab["id2term"][x] for x in out]
    return out


def _make_benchmark_cases(seed: int = 123) -> List[RePairCase]:
    import random

    rng = random.Random(seed)
    nt_base = 10_000
    cases: List[RePairCase] = []

    small_pattern = [0, 1, 2, 1]
    small_rule = nt_base + 1
    g_small = {small_rule: tuple(small_pattern)}
    s_small = tuple([small_rule] * 64)
    cases.append(
        RePairCase(
            name="small_repeated_no_noise",
            sequence=_expand_from_grammar(g_small, s_small),
            ground_truth_grammar=g_small,
            ground_truth_start=s_small,
            structured=True,
        )
    )

    small_rule_noise = nt_base + 2
    g_small_noise = {small_rule_noise: tuple(small_pattern)}
    s_small_noise: List[int] = []
    for i in range(64):
        s_small_noise.append(small_rule_noise)
        if i != 63:
            for _ in range(rng.randrange(0, 4)):
                s_small_noise.append(rng.randrange(0, 8))
    cases.append(
        RePairCase(
            name="small_repeated_with_noise",
            sequence=_expand_from_grammar(g_small_noise, tuple(s_small_noise)),
            ground_truth_grammar=g_small_noise,
            ground_truth_start=tuple(s_small_noise),
            structured=True,
        )
    )

    large_pattern = [rng.randrange(0, 12) for _ in range(80)]
    large_rule = nt_base + 3
    g_large = {large_rule: tuple(large_pattern)}
    s_large = tuple([large_rule] * 12)
    cases.append(
        RePairCase(
            name="large_repeated_no_noise",
            sequence=_expand_from_grammar(g_large, s_large),
            ground_truth_grammar=g_large,
            ground_truth_start=s_large,
            structured=True,
        )
    )

    large_rule_noise = nt_base + 4
    g_large_noise = {large_rule_noise: tuple(large_pattern)}
    s_large_noise: List[int] = []
    for i in range(12):
        s_large_noise.append(large_rule_noise)
        if i != 11:
            for _ in range(rng.randrange(0, 9)):
                s_large_noise.append(rng.randrange(0, 12))
    cases.append(
        RePairCase(
            name="large_repeated_with_noise",
            sequence=_expand_from_grammar(g_large_noise, tuple(s_large_noise)),
            ground_truth_grammar=g_large_noise,
            ground_truth_start=tuple(s_large_noise),
            structured=True,
        )
    )

    r1 = nt_base + 10
    r2 = nt_base + 11
    r3 = nt_base + 12
    r4 = nt_base + 13
    r5 = nt_base + 14
    g_hier = {
        r1: (0, 1, 0, 2),
        r2: (3, 4, 3, 5),
        r3: (6, 7, 6, 8),
        r4: (r1, r2, r1, r3),
        r5: (r1, r3, r2, r1),
    }
    s_hier = tuple([r4, r5] * 35)
    cases.append(
        RePairCase(
            name="hierarchical_no_noise",
            sequence=_expand_from_grammar(g_hier, s_hier),
            ground_truth_grammar=g_hier,
            ground_truth_start=s_hier,
            structured=True,
        )
    )

    s_hier_noise: List[int] = []
    for i in range(70):
        s_hier_noise.append(r4 if i % 2 == 0 else r5)
        if i != 69:
            for _ in range(rng.randrange(0, 4)):
                s_hier_noise.append(rng.randrange(0, 10))
    cases.append(
        RePairCase(
            name="hierarchical_with_noise",
            sequence=_expand_from_grammar(g_hier, tuple(s_hier_noise)),
            ground_truth_grammar=g_hier,
            ground_truth_start=tuple(s_hier_noise),
            structured=True,
        )
    )

    sparse_prefix = [rng.randrange(0, 4) for _ in range(1000)]
    sparse_pattern = [rng.randrange(0, 4) for _ in range(100)]
    sparse_rule = nt_base + 20
    g_sparse = {sparse_rule: tuple(sparse_pattern)}
    s_sparse = tuple(sparse_prefix + [sparse_rule, sparse_rule])
    cases.append(
        RePairCase(
            name="sparse_long_repeat_random_prefix",
            sequence=_expand_from_grammar(g_sparse, s_sparse),
            ground_truth_grammar=g_sparse,
            ground_truth_start=s_sparse,
            structured=True,
        )
    )

    random_seq = [rng.randrange(0, 8) for _ in range(20_000)]
    cases.append(
        RePairCase(
            name="random_unstructured",
            sequence=random_seq,
            ground_truth_grammar={},
            ground_truth_start=tuple(random_seq),
            structured=False,
        )
    )

    return cases


def _run_single_mode(
    sequence: List[int],
    mode_cfg: Dict[str, Any],
    repeats: int,
) -> Tuple[Dict[int, RHS], RHS, Dict[str, Any], float]:
    elapsed = 0.0
    out: Optional[Tuple[Dict[int, RHS], RHS, Dict[str, Any]]] = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = mr_repair_encode(sequence, **mode_cfg)
        elapsed += time.perf_counter() - t0
    assert out is not None
    return out[0], out[1], out[2], elapsed / max(1, repeats)


def _run_self_tests_and_benchmarks() -> None:
    modes: List[Tuple[str, Dict[str, Any]]] = [
        ("no_mdl", {"mdl": False}),
        (
            "mdl_no_sa",
            {"mdl": True, "mdl_mode": "global_static", "mdl_sa_top_k": 0},
        ),
        (
            "mdl_sa",
            {
                "mdl": True,
                "mdl_mode": "global_static",
                "mdl_sa_top_k": 24,
                "mdl_sa_every": 4,
                "mdl_sa_min_len": 2,
                "mdl_sa_max_len": 256,
            },
        ),
    ]
    cases = _make_benchmark_cases(seed=123)
    speed_repeats = 2
    tol = 1e-6

    print("MR-RePair regression tests and benchmarks")
    print("Modes: no_mdl, mdl_no_sa, mdl_sa")
    print(f"Timing: mean of {speed_repeats} run(s) per mode/case")
    print()

    for case in cases:
        empty_bits = _two_part_code_length({}, tuple(case.sequence))
        gt_bits = _two_part_code_length(
            case.ground_truth_grammar, case.ground_truth_start
        )
        results: Dict[str, Dict[str, float | int]] = {}
        print(f"[CASE] {case.name}  n={len(case.sequence)}")
        print(f"  empty_bits={empty_bits:.3f}  ground_truth_bits={gt_bits:.3f}")

        for mode_name, cfg in modes:
            grammar, start_rule, vocab, avg_sec = _run_single_mode(
                case.sequence, cfg, repeats=speed_repeats
            )
            decoded = _decode_to_input_space(grammar, start_rule, vocab)
            if isinstance(decoded, str):
                ok = isinstance(case.sequence, str) and decoded == case.sequence
            else:
                ok = list(decoded) == list(case.sequence)
            assert ok, f"Round-trip failed for case={case.name}, mode={mode_name}"

            total_bits = _two_part_code_length(grammar, start_rule)
            results[mode_name] = {
                "bits": total_bits,
                "rules": float(len(grammar)),
                "ms": avg_sec * 1000.0,
            }
            print(
                f"  {mode_name:<10} rules={len(grammar):>3}  bits={total_bits:>10.3f}  "
                f"time_ms={avg_sec * 1000.0:>8.3f}"
            )

        mdl_sa_bits = float(results["mdl_sa"]["bits"])
        mdl_no_sa_bits = float(results["mdl_no_sa"]["bits"])

        if case.structured:
            assert (
                mdl_sa_bits <= empty_bits + tol
            ), f"MDL+SA failed to beat empty grammar on structured case {case.name}"
            assert (
                float(results["mdl_sa"]["rules"]) > 0 or mdl_sa_bits <= gt_bits + tol
            ), f"MDL+SA neither found grammar nor beat ground truth on {case.name}"

        if case.name == "sparse_long_repeat_random_prefix":
            assert (
                mdl_sa_bits + tol < mdl_no_sa_bits
            ), "Expected SA to strictly improve sparse long-repeat case"

        speed_no_mdl = float(results["no_mdl"]["ms"])
        speed_mdl_no_sa = float(results["mdl_no_sa"]["ms"])
        speed_mdl_sa = float(results["mdl_sa"]["ms"])
        print(
            "  trend"
            f"  bits(no_mdl, mdl_no_sa, mdl_sa)=({float(results['no_mdl']['bits']):.3f}, "
            f"{mdl_no_sa_bits:.3f}, {mdl_sa_bits:.3f})"
            f"  time_ms=({speed_no_mdl:.3f}, {speed_mdl_no_sa:.3f}, {speed_mdl_sa:.3f})"
        )
        print()

    print("All regression checks passed.")


if __name__ == "__main__":
    _run_self_tests_and_benchmarks()
