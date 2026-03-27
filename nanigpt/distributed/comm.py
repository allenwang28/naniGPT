"""Communication primitives: autograd.Function conjugate pairs.

Each class pairs a forward collective with its mathematically correct
backward dual. These are the atoms that TP, EP, and PP build on.

Planned pairs (from Megatron's mappings.py):
    CopyToParallelRegion:            forward=identity,       backward=all-reduce
    ReduceFromParallelRegion:        forward=all-reduce,     backward=identity
    GatherFromSequenceParallelRegion:   forward=all-gather,  backward=reduce-scatter
    ReduceScatterToSequenceParallelRegion: forward=reduce-scatter, backward=all-gather
    AllToAll:                        forward=all-to-all,     backward=inverse all-to-all

Not implemented yet — will be added when tensor parallelism lands.
"""
