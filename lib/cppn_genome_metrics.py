import math
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Iterable, Sequence

import neat

ConnectionKey = tuple[int, int]


@dataclass(frozen=True)
class CPPNGenomeMetrics:
    """Structural summaries of the input-dependent part of a CPPN genome."""

    nodes: int
    enabled_connections: int
    input_output_hidden_nodes: int
    input_output_connections: int
    input_output_hidden_fraction: float
    depth: int
    reused_hidden_fraction: float
    combining_hidden_fraction: float
    hidden_fan_in: int
    hidden_fan_out: int
    hidden_activation_diversity: int
    log10_one_plus_input_output_paths: float


def compute_cppn_genome_metrics(
    genome: neat.DefaultGenome,
    input_keys: Sequence[int],
    output_keys: Sequence[int],
) -> CPPNGenomeMetrics:
    """Measure hierarchy and feature reuse along enabled input-to-output paths."""
    enabled_connections = [
        connection.key
        for connection in genome.connections.values()
        if connection.enabled
    ]
    successors = _adjacency(enabled_connections)
    predecessors = _adjacency((target, source) for source, target in enabled_connections)

    input_descendants = _reachable_nodes(input_keys, successors)
    output_ancestors = _reachable_nodes(output_keys, predecessors)
    input_output_nodes = input_descendants & output_ancestors
    input_output_connections = [
        (source, target)
        for source, target in enabled_connections
        if source in input_output_nodes and target in input_output_nodes
    ]

    output_key_set = set(output_keys)
    hidden_nodes = set(genome.nodes) - output_key_set
    input_output_hidden_nodes = hidden_nodes & input_output_nodes

    fan_in: dict[int, int] = defaultdict(int)
    fan_out: dict[int, int] = defaultdict(int)
    for source, target in input_output_connections:
        fan_out[source] += 1
        fan_in[target] += 1

    reused_hidden_nodes = sum(
        fan_out[node] > 1 for node in input_output_hidden_nodes
    )
    combining_hidden_nodes = sum(
        fan_in[node] > 1 for node in input_output_hidden_nodes
    )
    depth, n_input_output_paths = _dag_path_statistics(
        nodes=input_output_nodes,
        connections=input_output_connections,
        input_keys=input_keys,
        output_keys=output_keys,
    )

    return CPPNGenomeMetrics(
        nodes=len(genome.nodes),
        enabled_connections=len(enabled_connections),
        input_output_hidden_nodes=len(input_output_hidden_nodes),
        input_output_connections=len(input_output_connections),
        input_output_hidden_fraction=_fraction(
            len(input_output_hidden_nodes), len(hidden_nodes)
        ),
        depth=depth,
        reused_hidden_fraction=_fraction(
            reused_hidden_nodes, len(input_output_hidden_nodes)
        ),
        combining_hidden_fraction=_fraction(
            combining_hidden_nodes, len(input_output_hidden_nodes)
        ),
        hidden_fan_in=max(
            (fan_in[node] for node in input_output_hidden_nodes), default=0
        ),
        hidden_fan_out=max(
            (fan_out[node] for node in input_output_hidden_nodes), default=0
        ),
        hidden_activation_diversity=len(
            {genome.nodes[node].activation for node in input_output_hidden_nodes}
        ),
        log10_one_plus_input_output_paths=math.log10(1 + n_input_output_paths),
    )


def summarize_cppn_genomes(
    genomes: Sequence[neat.DefaultGenome],
    input_keys: Sequence[int],
    output_keys: Sequence[int],
) -> dict[str, float]:
    """Return population mean/max summaries suitable for scalar logging."""
    if not genomes:
        raise ValueError("Cannot summarize an empty CPPN genome population.")

    genome_metrics = [
        compute_cppn_genome_metrics(
            genome=genome,
            input_keys=input_keys,
            output_keys=output_keys,
        )
        for genome in genomes
    ]
    summaries: dict[str, float] = {}
    for metric_field in fields(CPPNGenomeMetrics):
        values = [
            float(getattr(metrics, metric_field.name)) for metrics in genome_metrics
        ]
        summaries[f"mean_{metric_field.name}"] = sum(values) / len(values)
        summaries[f"max_{metric_field.name}"] = max(values)
    return summaries


def _adjacency(connections: Iterable[ConnectionKey]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for source, target in connections:
        adjacency[source].add(target)
    return adjacency


def _reachable_nodes(
    start_nodes: Iterable[int], adjacency: dict[int, set[int]]
) -> set[int]:
    reachable = set(start_nodes)
    pending = list(reachable)
    while pending:
        node = pending.pop()
        for adjacent_node in adjacency.get(node, set()):
            if adjacent_node not in reachable:
                reachable.add(adjacent_node)
                pending.append(adjacent_node)
    return reachable


def _dag_path_statistics(
    nodes: set[int],
    connections: Sequence[ConnectionKey],
    input_keys: Sequence[int],
    output_keys: Sequence[int],
) -> tuple[int, int]:
    """Return longest input-output path length and total input-output path count."""
    if not nodes:
        return 0, 0

    successors = _adjacency(connections)
    in_degrees = {node: 0 for node in nodes}
    for _, target in connections:
        in_degrees[target] += 1

    pending = [node for node, degree in in_degrees.items() if degree == 0]
    depths = {node: 0 for node in nodes}
    path_counts = {node: int(node in input_keys) for node in nodes}
    n_processed = 0

    while pending:
        source = pending.pop()
        n_processed += 1
        for target in successors.get(source, set()):
            depths[target] = max(depths[target], depths[source] + 1)
            path_counts[target] += path_counts[source]
            in_degrees[target] -= 1
            if in_degrees[target] == 0:
                pending.append(target)

    if n_processed != len(nodes):
        raise ValueError("CPPN structural metrics require a feed-forward genome.")

    return (
        max((depths.get(output, 0) for output in output_keys), default=0),
        sum(path_counts.get(output, 0) for output in output_keys),
    )


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
