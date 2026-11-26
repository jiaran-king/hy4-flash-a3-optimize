# SPDX-License-Identifier: Apache-2.0
"""Preserve finite E4M3FN rounding in a floating tensor on Ascend A3."""

import torch


def round_e4m3fn(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs().clamp_max(448.0).float().contiguous()

    # FP32 has 23 mantissa bits and E4M3FN has 3. The integer bias
    # rounds to nearest, with the retained bit deciding midpoint ties.
    bits = magnitude.view(torch.int32)
    retained = (bits >> 20) & 1
    normal = ((bits + 0x7FFFF + retained) & ~0xFFFFF).view(torch.float32)

    # E4M3FN subnormals are spaced at exactly 2**-9.
    units = magnitude * 512.0
    lower = units.floor()
    fraction = units - lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.long() & 1) != 0))
    subnormal = (lower + increment) * (2.0**-9)
    rounded = torch.where(magnitude < 2.0**-6, subnormal, normal)
    return rounded * values.sign()
