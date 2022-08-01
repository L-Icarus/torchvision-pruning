from torch import nn


PRUNE_CONV_OPTYPE = (
    nn.Conv2d,
)

PRUNE_NORMALIZATION_OPTYPE = (
    nn.BatchNorm2d,
    nn.LayerNorm,
    nn.GroupNorm,
)

PRUNE_FC_OPTYPE = (
    nn.Linear,
)


ALL_PRUNE_OPTYPE = PRUNE_CONV_OPTYPE + PRUNE_FC_OPTYPE + PRUNE_NORMALIZATION_OPTYPE
