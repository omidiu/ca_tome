import torch
from typing import Tuple, Callable, Dict, Any
import time


def do_nothing(x: torch.Tensor):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1), dim - 1 if dim < 0 else dim, index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    similarity_threshold: float,
    cache_info: Dict[str, Any],
    inference_step: int,
    no_rand: bool = False,
    generator: torch.Generator = None,
) -> Tuple[Callable, Callable]:
    """
    Performs bipartite soft matching to merge tokens in a 2D grid layout, with caching for efficiency.

    This function partitions tokens into source (src) and destination (dst) sets, then merges similar tokens
    from src into dst based on a similarity threshold. The dst tokens are selected by choosing one token
    randomly (or deterministically) within each (sx, sy) sized region of the grid.

    Args:
        metric (torch.Tensor): Tensor of shape [B, N, C] containing token features/embeddings used to compute similarities.
            B is batch size, N is number of tokens, C is feature dimension.
        w (int): Width of the token grid in number of tokens
        h (int): Height of the token grid in number of tokens
        sx (int): Stride in x-dimension for selecting dst tokens. Controls granularity of merging.
        sy (int): Stride in y-dimension for selecting dst tokens. Controls granularity of merging.
        similarity_threshold (float): Cosine similarity threshold in [-1, 1] that determines whether tokens can be merged.
            Higher values (e.g. 0.7) require tokens to be more similar for merging.
        cache_info (Dict[str, Any]): Dictionary containing caching configuration and state:
            - compute_on: List of timesteps where merging should be recomputed
            - merge: Cached merge information (None in first iteration)
            - unmerge: Cached unmerge information (None in first iteration)
        inference_step (int): Current inference timestep, used for cache management
        no_rand (bool, optional): If True, disable randomness and always select top-left token as dst.
            Defaults to False.
        generator (torch.Generator, optional): Random number generator for reproducibility.
            If None, will initialize new generator.

    Returns:
        Tuple[Callable, Callable]: Tuple of (merge_func, unmerge_func) that handle the token merging/unmerging.
            merge_func: Merges similar tokens according to computed matching
            unmerge_func: Reverses the merging operation to restore original token arrangement
            If similarity_threshold >= 1, returns (do_nothing, do_nothing).
    """

    # Return cached functions if not at a checkpoints
    if (
        inference_step not in cache_info["compute_on"]
        and cache_info["merge"] is not None
    ):
        return cache_info["merge"], cache_info["unmerge"]

    B, N, _ = metric.shape

    # Skip merging if threshold too high
    if similarity_threshold >= 1:
        return do_nothing, do_nothing

    # Handle MPS device edge case
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        # Calculate grid dimensions
        hsy, wsx = h // sy, w // sx

        # Initialize destination token selection
        if no_rand:
            # Deterministic: always select top-left token
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            # Random: select one token per grid cell randomly
            rand_idx = torch.randint(
                sy * sx,
                size=(hsy, wsx, 1),
                device=generator.device,
                generator=generator,
            ).to(metric.device)

        # Create token assignment buffer
        idx_buffer_view = torch.zeros(
            hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64
        )
        # Mark destination tokens with -1
        idx_buffer_view.scatter_(
            dim=2,
            index=rand_idx,
            src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype),
        )
        # Reshape to match image grid
        idx_buffer_view = (
            idx_buffer_view.view(hsy, wsx, sy, sx)
            .transpose(1, 2)
            .reshape(hsy * sy, wsx * sx)
        )

        # Handle non-divisible image dimensions
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[: (hsy * sy), : (wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # Sort to get dst|src ordering
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # Free temporary buffers
        del idx_buffer, idx_buffer_view

        # Split into source and destination indices
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src tokens
        b_idx = rand_idx[:, :num_dst, :]  # dst tokens

        def split(x):
            """Split tensor into source and destination tokens"""
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Compute cosine similarities
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Find best matches greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Determine number of tokens to merge based on similarity threshold
        similarity_mask = node_max >= similarity_threshold
        similarity_count = similarity_mask.sum(dim=-1)
        r = similarity_count.max().item()

        # Split tokens into merged and unmerged sets
        unm_idx = edge_idx[..., r:, :]  # Unmerged tokens
        src_idx = edge_idx[..., :r, :]  # Tokens to be merged
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # Store merged pairs for analysis
        pairs = torch.cat((src_idx, dst_idx), dim=-1)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            """Merge tokens according to computed CA-ToMe matching"""
            src, dst = split(x)
            n, t1, c = src.shape

            # Handle unmerged tokens
            unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            # Merge similar tokens
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=1)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            """Reverse the merging operation"""
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            _, _, c = unm.shape

            # Retrieve merged source tokens
            src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

            # Reconstruct original token arrangement
            out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
            # Place destination tokens
            out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
            # Place unmerged source tokens
            out.scatter_(
                dim=-2,
                index=gather(
                    a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
                ).expand(B, unm_len, c),
                src=unm,
            )
            # Place merged source tokens
            out.scatter_(
                dim=-2,
                index=gather(
                    a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
                ).expand(B, r, c),
                src=src,
            )

            return out

        # Cache the merge/unmerge functions
        cache_info["merge"] = merge
        cache_info["unmerge"] = unmerge

        return merge, unmerge
