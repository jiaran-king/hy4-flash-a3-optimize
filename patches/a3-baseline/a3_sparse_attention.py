# SPDX-License-Identifier: Apache-2.0

"""Paged sparse attention with learned sinks using the A3 BF16 KV cache."""

from collections.abc import Sequence

import torch


QUERY_CHUNK = 16


def attend_paged_with_sink(
    query_latent: torch.Tensor,
    query_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    selected_tokens: torch.Tensor,
    block_tables: torch.Tensor,
    query_ends: Sequence[int],
    sequence_lengths: Sequence[int],
    sinks: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # Keep the original logical token selection, including SWA and IndexPool.
    # Each sink adds a logit to the denominator and has a zero value.
    block_size = latent_cache.shape[1]
    flat_latent = latent_cache.reshape(-1, latent_cache.shape[-1])
    flat_rope = rope_cache.reshape(-1, rope_cache.shape[-1])
    output = torch.empty_like(query_latent)
    request_start = 0
    for request, (request_end, sequence_length) in enumerate(
        zip(query_ends, sequence_lengths, strict=True)
    ):
        for start in range(request_start, request_end, QUERY_CHUNK):
            end = min(start + QUERY_CHUNK, request_end)
            indexes = selected_tokens[start:end].long()
            valid = (indexes >= 0) & (indexes < sequence_length)
            indexes = indexes.masked_fill(~valid, 0)
            blocks = block_tables[request].index_select(
                0, (indexes // block_size).flatten()
            ).reshape_as(indexes)
            slots = (blocks.long() * block_size + indexes % block_size).flatten()
            vectors = flat_latent.index_select(0, slots).reshape(
                end - start, indexes.shape[-1], flat_latent.shape[-1]
            ).float()
            rope = flat_rope.index_select(0, slots).reshape(
                end - start, indexes.shape[-1], flat_rope.shape[-1]
            ).float()
            logits = (
                torch.bmm(query_latent[start:end].float(), vectors.transpose(1, 2))
                + torch.bmm(query_rope[start:end].float(), rope.transpose(1, 2))
            ) * scale
            logits.masked_fill_(~valid[:, None, :], -torch.inf)
            sink = sinks.float().reshape(1, -1, 1).expand(end - start, -1, 1)
            weights = torch.cat((logits, sink), dim=-1).softmax(dim=-1)[..., :-1]
            output[start:end].copy_(torch.bmm(weights, vectors))
        request_start = request_end
    return output
