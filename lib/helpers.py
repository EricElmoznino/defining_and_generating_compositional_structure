import heapq
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from wandb import Image


def positional_encode(
    coords: torch.Tensor,
    num_bands: int,
    include_raw: bool,
) -> torch.Tensor:
    """Apply powers-of-two sinusoidal positional encoding to coordinates."""
    if num_bands <= 0:
        if include_raw:
            return coords
        raise ValueError(
            "At least one coordinate feature is required. "
            "Set include_raw=True or num_bands>0."
        )
    frequencies = (2.0 ** torch.arange(num_bands, device=coords.device)) * math.pi
    frequencies = frequencies.to(coords.dtype).view(1, 1, num_bands)
    expanded = coords.unsqueeze(-1) * frequencies
    features: list[torch.Tensor] = []
    if include_raw:
        features.append(coords)
    features.append(torch.sin(expanded).reshape(coords.shape[0], -1))
    features.append(torch.cos(expanded).reshape(coords.shape[0], -1))
    return torch.cat(features, dim=-1)


def uses_shared_coordinate_indices(coord_indices: torch.Tensor) -> bool:
    return (
        coord_indices.ndim == 1
        or int(coord_indices.shape[0]) <= 1
        or int(coord_indices.stride(0)) == 0
    )


def select_active_coordinate_indices(
    coord_indices: torch.Tensor,
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    if coord_indices.ndim == 1:
        return coord_indices.unsqueeze(0).expand(int(sample_indices.numel()), -1)
    if uses_shared_coordinate_indices(coord_indices):
        return coord_indices[:1].expand(int(sample_indices.numel()), -1)
    return coord_indices.index_select(0, sample_indices)


def lookup_coordinate_batch(
    encoded_coords: torch.Tensor,
    coord_indices: torch.Tensor,
) -> torch.Tensor:
    if coord_indices.ndim == 1:
        return encoded_coords.index_select(0, coord_indices)
    if uses_shared_coordinate_indices(coord_indices):
        return encoded_coords.index_select(0, coord_indices[0])
    return encoded_coords[coord_indices]


# ------------------------------
# Visualization utilities
# ------------------------------


def songs_from_strings(
    strings: np.ndarray, vocab_size: int, sample_rate: int = 16000
) -> np.ndarray:
    """Generate a batch of audio songs in minor key from a batch of symbolic strings.

    Args:
        strings (np.ndarray): (B, L) integer array with values in [0, vocab_size)
        vocab_size (int): size of the vocabulary
        sample_rate (int, optional): audio sample rate. Defaults to 16000.

    Returns:
        np.ndarray: batch of audio waveforms, shape (B, T)
    """
    minor_intervals = np.array([0, 2, 3, 5, 7, 8, 10])
    base_freq = 220.0
    seconds_per_quarter = 0.26
    samples_per_note = int(sample_rate * seconds_per_quarter)
    note_time = np.linspace(0, seconds_per_quarter, samples_per_note, endpoint=False)
    attack_samples = max(1, int(0.2 * samples_per_note))
    release_samples = max(1, int(0.1 * samples_per_note))
    sustain_level = 0.9

    def cosine_ramp(start: float, end: float, n: int) -> np.ndarray:
        if n <= 1:
            return np.array([end], dtype=np.float32)
        t = np.linspace(0.0, np.pi, n, endpoint=True, dtype=np.float32)
        return start + (end - start) * 0.5 * (1 - np.cos(t))

    if samples_per_note > attack_samples + release_samples:
        sustain_samples = samples_per_note - attack_samples - release_samples
        attack_curve = cosine_ramp(0.0, sustain_level, attack_samples)
        release_curve = cosine_ramp(sustain_level, 0.0, release_samples)
        envelope = np.concatenate(
            [
                attack_curve,
                np.full(sustain_samples, sustain_level, dtype=np.float32),
                release_curve,
            ]
        )
    else:
        fade_samples = max(1, samples_per_note // 2)
        fade = cosine_ramp(0.0, 1.0, fade_samples)
        envelope = np.ones(samples_per_note, dtype=np.float32)
        envelope[:fade_samples] = fade
        envelope[-fade_samples:] = fade[::-1]
    vocab_steps = (
        np.arange(vocab_size) // len(minor_intervals)
    ) * 12 + minor_intervals[np.arange(vocab_size) % len(minor_intervals)]
    note_freqs = base_freq * 2 ** (vocab_steps / 12)
    vibrato_freq = 5.5
    vibrato_depth = 0.003
    waveforms = []
    for seq in strings:
        freqs = note_freqs[seq.astype(int)]
        freq_mod = freqs[:, None] * (
            1 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * note_time)
        )
        phase = 2 * np.pi * freq_mod * note_time
        harmonics = (
            np.sin(phase)
            + 0.6 * np.sin(2 * phase)
            + 0.4 * np.sin(3 * phase)
            + 0.25 * np.sin(4 * phase)
            + 0.15 * np.sin(5 * phase)
        )
        harmonics *= envelope
        waveform = (harmonics / 2.4).ravel().astype(np.float32) * 0.2
        waveforms.append(waveform)
    return np.array(waveforms)


def dot_plots_from_strings(strings: np.ndarray) -> list[Image]:
    """Generate a batch of dot plots from a batch of symbolic strings.

    Args:
        strings (np.ndarray): (B, L) integer array with values in [0, vocab_size)

    Returns:
        list[Image]: list of matplotlib figures as W&B images containing the dot plots
    """
    dot_plots = (strings[:, :, None] == strings[:, None, :]).astype(int)
    figures = []
    for i in range(dot_plots.shape[0]):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(dot_plots[i], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        figures.append(Image(fig))
    return figures


def hilbert_curves_from_strings(strings: np.ndarray) -> list[Image]:
    """Generate a batch of hilbert curves from a batch of symbolic strings.

    Args:
        strings (np.ndarray): (B, L) integer array with values in [0, vocab_size)

    Returns:
        list[Image]: list of matplotlib figures as W&B images containing the hilbert curves
    """

    def d2xy(n: int, d: int) -> tuple[int, int]:
        """
        Map distance d to (x, y) coordinates on a Hilbert curve of order n.
        Grid size is 2^n x 2^n.

        Args:
            n (int): Order of the curve (side length 2^n).
            d (int): Distance along the curve (0 to 4^n - 1).

        Returns:
            tuple: (x, y) coordinates.
        """
        x = 0
        y = 0
        s = 1
        while s < (1 << n):
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)

            # Rotate/flip quadrant
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                # Swap x and y
                x, y = y, x

            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    B, L = strings.shape

    # Determine the order of the Hilbert curve needed
    # Area must be >= L. Side length = 2^order. Area = 4^order.
    # log_4(L) = log_2(L) / 2
    if L > 0:
        order = math.ceil(math.log(L, 4))
    else:
        order = 1

    side_length = 1 << order

    # Pre-calculate coordinates for the string indices (0..L-1)
    coords = [d2xy(order, i) for i in range(L)]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Determine colormap based on max symbol value to ensure distinct colors
    max_val = np.max(strings) if strings.size > 0 else 0
    if max_val < 10:
        cmap_name = "tab10"
    elif max_val < 20:
        cmap_name = "tab20"
    else:
        cmap_name = "nipy_spectral"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white", alpha=0)  # Make unused pixels transparent

    figures = []
    for i in range(B):
        # Create grid filled with NaNs (for transparency of unused slots)
        grid = np.full((side_length, side_length), np.nan)
        # Fill grid with symbol values
        # We treat y as row index and x as column index
        grid[ys, xs] = strings[i]

        # Plot
        # vmin/vmax ensures consistent coloring across the batch
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(
            grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(max_val, 1)
        )
        ax.set_xticks([])
        ax.set_yticks([])
        # Optional: Add a subtle border to show the empty space if string < grid size
        for spine in ax.spines.values():
            spine.set_edgecolor("lightgray")
        fig.tight_layout()
        figures.append(Image(fig))

    return figures


def paths_from_strings(
    strings: np.ndarray,
    vocab_size: int,
    angle_range: tuple[float, float] = (-60.0, 60.0),
) -> list[Image]:
    """Generate a batch of path plots from a batch of symbolic strings.

    Args:
        strings (np.ndarray): (B, L) integer array with values in [0, vocab_size)
        vocab_size (int): size of the vocabulary
        angle_range (tuple[float, float], optional): range of angles for the path plot in degrees.

    Returns:
        list[Image]: list of matplotlib figures as W&B images containing the path plots
    """
    angles = np.linspace(angle_range[0], angle_range[1], vocab_size)
    angles = np.deg2rad(angles)
    figures = []
    for i in range(strings.shape[0]):
        string = strings[i]
        x, y, a = [0.0], [0.0], 0.0
        for token in string:
            a += angles[token] % (2 * np.pi)
            x.append(x[-1] + np.cos(a))
            y.append(y[-1] + np.sin(a))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(x, y)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        figures.append(Image(fig))
    return figures


def circle_paths_from_strings(strings: np.ndarray, vocab_size: int) -> list[Image]:
    """Generate a batch of circle paths from a batch of symbolic strings.

    Args:
        strings (np.ndarray): (B, L) integer array with values in [0, vocab_size)
        vocab_size (int): size of the vocabulary

    Returns:
        list[Image]: list of matplotlib figures as W&B images containing the dot plots
    """
    vocab_indices = np.arange(vocab_size, dtype=np.float32)
    angles = 2 * np.pi * vocab_indices / vocab_size
    circle_coordinates = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    figures = []
    for i in range(strings.shape[0]):
        path_coordinates = circle_coordinates[strings[i]]
        segments = np.concatenate(
            [path_coordinates[:-1, None], path_coordinates[1:, None]], axis=1
        )
        segments, counts = np.unique(segments, return_counts=True, axis=0)
        weights = counts / counts.max() * 5.0  # Scale line widths
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                1.0,
                edgecolor="lightgray",
                fill=False,
                linewidth=2.0,
            )
        )
        ax.scatter(
            circle_coordinates[:, 0],
            circle_coordinates[:, 1],
            s=10,
            color="lightgray",
            zorder=2,
        )
        lc = LineCollection(segments, linewidths=weights)
        ax.add_collection(lc)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        fig.tight_layout()
        figures.append(Image(fig))
    return figures


# ------------------------------
# Compression utilities
# ------------------------------


def huffman_code_lengths(freq: dict[Any, int]) -> dict[Any, int]:
    """
    Given symbol->count, return symbol->code_length (in bits).
    Uses classic Huffman (ties broken stably).
    """
    if not freq:
        return {}
    if len(freq) == 1:
        # Single symbol: assign 1 bit (can't be 0 in practical coders)
        k = next(iter(freq))
        return {k: 1}

    # Heap of (weight, unique_id, {symbols}) as leaf/merged nodes
    uid = 0
    heap = []
    for sym, w in freq.items():
        heap.append((w, uid, {sym}))
        uid += 1
    heapq.heapify(heap)

    # Build tree; track depths via symbol sets
    # We don't need explicit pointers: at each merge, depth++ for all symbols in that set.
    depths = Counter({sym: 0 for sym in freq})
    while len(heap) > 1:
        w1, _, s1 = heapq.heappop(heap)
        w2, _, s2 = heapq.heappop(heap)
        merged = s1 | s2
        # Every symbol in merged increases depth by 1
        for s in merged:
            depths[s] += 1
        heapq.heappush(heap, (w1 + w2, uid, merged))
        uid += 1
    return dict(depths)


def zigzag(n: int) -> int:
    # Maps ...,-2,-1,0,1,2,... -> 3,1,0,2,4,...
    return (
        (n << 1) ^ (n >> 63)
        if -(1 << 62) <= n < (1 << 62)
        else (2 * (-n) - 1 if n < 0 else 2 * n)
    )


def fixed_width_for(values: list[int]) -> int:
    if not values:
        return 0
    m = max(zigzag(v) for v in values)
    # width = ceil(log2(m+1)), but ensure at least 1 bit if m==0
    return max(1, (m + 1).bit_length())


def tokenize_with_parts(
    whole: list[int],
    parts: list[list[int]],
) -> tuple[list[tuple[str, int | None, int | None]], dict[str, Any]]:
    """
    Produce a token stream using only `parts`, greedy longest match.
    Events:
      - ("TOK", part_index, length)
      - ("LIT", value, 1)
    Returns (events, stats)
    """
    # index by first integer to prune candidates
    by_first: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for idx, p in enumerate(parts):
        if not p:
            continue
        by_first[p[0]].append((idx, tuple(p)))

    # sort candidates for determinism (prefer longer, then smaller part_index)
    for k in by_first:
        by_first[k].sort(key=lambda it: (-len(it[1]), it[0]))

    n = len(whole)
    i = 0
    events: list[tuple[str, int | None, int | None]] = []
    while i < n:
        first = whole[i]
        best = None  # (part_idx, length)
        if first in by_first:
            cand = by_first[first]
            # candidates already sorted by decreasing length; pick first that matches
            for part_idx, seq in cand:
                L = len(seq)
                if i + L <= n and tuple(whole[i : i + L]) == seq:
                    best = (part_idx, L)
                    break
        if best is None:
            events.append(("LIT", whole[i], 1))
            i += 1
        else:
            part_idx, L = best
            events.append(("TOK", part_idx, L))
            i += L

    stats = {
        "token_count": sum(1 for e in events if e[0] == "TOK"),
        "literal_count": sum(1 for e in events if e[0] == "LIT"),
    }
    return events, stats


def dictionary_conditional_compressed_size_bits(
    whole: list[int],
    parts: list[list[int]],
    literal_coding: str = "huffman",  # "fixed" | "huffman"
) -> dict[str, Any]:
    """
    Compute compressed bit size using:
      - Huffman over event kinds: TOK(part_index) for each part, plus one LIT marker.
      - For LIT values:
          'fixed'    -> fixed-width ZigZag integers
          'huffman'  -> second Huffman over literal values
    Returns breakdown and total bits.
    """
    events, stats = tokenize_with_parts(whole, parts)

    # Build frequency table for event symbols: TOK_i and LIT
    event_freq: Counter = Counter()
    literal_values: list[int] = []
    for kind, arg, _ in events:
        if kind == "TOK":
            event_freq[("TOK", arg)] += 1
        else:
            event_freq[("LIT", None)] += 1
            literal_values.append(arg)

    event_code_len = huffman_code_lengths(dict(event_freq))
    event_bits = 0
    for kind, arg, _ in events:
        sym = ("TOK", arg) if kind == "TOK" else ("LIT", None)
        event_bits += event_code_len[sym]

    # Literal value bits
    literal_bits = 0
    literal_model: dict[Any, int] = {}
    if event_freq.get(("LIT", None), 0) > 0:
        if literal_coding == "huffman":
            lit_freq = Counter(literal_values)
            lit_code_len = huffman_code_lengths(dict(lit_freq))
            literal_model = lit_code_len
            for v in literal_values:
                literal_bits += lit_code_len[v]
        else:
            width = fixed_width_for(literal_values)
            literal_model = {"fixed_width": width}
            literal_bits = width * len(literal_values)

    total_bits = event_bits + literal_bits

    return {
        "total_bits": total_bits,
        "event_bits": event_bits,
        "literal_bits": literal_bits,
        "event_alphabet_size": len(event_freq),
        "event_code_lengths": event_code_len,  # e.g., {("TOK", 5): 7, ("LIT", None): 3, ...}
        "literal_model": literal_model,  # fixed width or per-value code lengths
        "stats": stats,
        "events_sample": events[:50],  # peek first 50 events for sanity
    }
