import torch

from lib.cppn_coordinates import CPPNCoordinateFrame

######################################
############### Models ###############
######################################

# A deterministic grammar is represented as a list of production rules.
# The index in the list corresponds to the nonterminal symbol being expanded by the production rule.
# Each production rule is represented as a list of symbol expansions (>=0 for nonterminal, <0 for terminal).
# Outermost list represents a batch of these grammars.
DeterministicGrammarModel = list[list[list[int]]]

# The Model type for CPPNs: A list of PyTorch modules where each module
# corresponds to one batch item, plus the per-item object shape
# (*spatial_dims, channels), i.e. without the batch dimension. Models fitted
# in a shared full-image coordinate system additionally carry their frame.
CPPNModel = (
    tuple[list[torch.nn.Module], tuple[int, ...]]
    | tuple[list[torch.nn.Module], tuple[int, ...], CPPNCoordinateFrame]
)
