import torch
import math
from typing import Type, Dict, Any, Tuple, Callable, List

from . import merge as merge
from .utils import isinstance_str, init_generator


def compute_merge(
    x: torch.Tensor,
    ca_tome_info: Dict[str, Any],
    cache_info: Dict[str, Any],
    inference_step,
) -> Tuple[Callable, ...]:
    original_h, original_w = ca_tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = ca_tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        similarity_threshold = args["similarity_threshold"]

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(
            x,
            w,
            h,
            args["sx"],
            args["sy"],
            similarity_threshold,
            cache_info,
            inference_step,
            no_rand=not use_rand,
            generator=args["generator"],
        )
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"] else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (
        (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    )
    m_m, u_m = (m, u) if args["merge_mlp"] else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good


def make_ca_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies CA-ToMe to the forward function of the block.
    """

    class AdaptiveToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(
            self, x: torch.Tensor, context: torch.Tensor = None
        ) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(
                x, self._ca_tome_info, self._cache_info, self.inference_step
            )

            # This is where the meat of the computation happens
            x = (
                u_a(
                    self.attn1(
                        m_a(self.norm1(x)),
                        context=context if self.disable_self_attn else None,
                    )
                )
                + x
            )
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x

    return AdaptiveToMeBlock


def make_diffusers_ca_tome_block(
    block_class: Type[torch.nn.Module],
) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies CA-ToMe to the forward function of the block.
    """

    class AdaptiveToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) CA-ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(
                hidden_states, self._ca_tome_info, self._cache_info, self.inference_step
            )

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.norm1(
                        hidden_states,
                        timestep,
                        class_labels,
                        hidden_dtype=hidden_states.dtype,
                    )
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) CA-ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=(
                    encoder_hidden_states if self.only_cross_attention else None
                ),
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) CA-ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )
                # (4) CA-ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) CA-ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            # (6) CA-ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) CA-ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return AdaptiveToMeBlock


def hook_ca_tome_model(model: torch.nn.Module):
    """Adds a forward pre hook to get the image size. This hook can be removed with remove_patch."""

    def hook(module, args):
        module._ca_tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._ca_tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def apply_patch(
    model: torch.nn.Module,
    compute_on: List[int],
    r: float = 0.7,
    max_downsample: int = 1,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
):
    """
    Patches a Stable Diffusion model with Cached Adaptive Token Merging (CA-ToMe) optimization.
    Apply this to the highest level Stable Diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - compute_on: List[int] specifying the timesteps where token merging patterns should be recomputed.
                  These checkpoints control when CA-ToMe updates its merging strategy during inference.
     - r: float between -1 and 1, cosine similarity threshold that determines whether tokens can be merged.
          Higher values (e.g. 0.7) require tokens to be more similar for merging.

    Optional Args:
     - max_downsample [1, 2, 4, or 8]: Controls which layers get CA-ToMe optimization based on downsampling.
                                       1 = only base resolution layers (4/15 layers)
                                       8 = all layers (15/15 layers)
                                       Recommended values: 1 or 2 for best quality/speed trade-off.
     - sx, sy: Stride values for partitioning tokens into source/destination sets.
               Higher strides allow more aggressive merging but may reduce quality.
               Default (2, 2) works well for most cases. Does not need to divide image size.
     - use_rand: Whether to add randomness when selecting destination tokens.
                 Recommended True for better results, set False if experiencing artifacts.
     - merge_attn: Whether to merge tokens in self-attention layers (recommended True).
     - merge_crossattn: Whether to merge tokens in cross-attention layers (recommended False).
     - merge_mlp: Whether to merge tokens in MLP layers (recommended False).

    Note: We carefully reasoned about these optional parameters and chose these specific values to
    provide an optimal balance between performance and quality. Please refer to our paper for more
    details about our reasoning.
    """

    # Make sure the module is not currently patched
    # Remove any existing CA-ToMe patches before applying new ones
    remove_patch(model)

    # Check if using diffusers library or base model
    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(
        model, "ModelMixin"
    )

    # Get reference to the core diffusion model
    if not is_diffusers:
        # Handle base Stable Diffusion model case
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            raise RuntimeError(
                "Provided model was not a Stable Diffusion / Latent Diffusion model, as expected."
            )
        diffusion_model = model.model.diffusion_model
    else:
        # Handle diffusers library case
        diffusion_model = model.unet if hasattr(model, "unet") else model

    # Initialize CA-ToMe configuration info
    diffusion_model._ca_tome_info = {
        "size": None,  # Will be set during forward pass
        "hooks": [],  # Store hooks for cleanup
        "args": {
            "similarity_threshold": r,
            "max_downsample": max_downsample,
            "sx": sx,
            "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
        },
    }

    # Add hook to capture input dimensions
    hook_ca_tome_model(diffusion_model)

    # Patch each transformer block with CA-ToMe functionality
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            # Select appropriate CA-ToMe block implementation
            make_ca_tome_block_fn = (
                make_diffusers_ca_tome_block if is_diffusers else make_ca_tome_block
            )
            # Replace block class with CA-ToMe-enabled version
            module.__class__ = make_ca_tome_block_fn(module.__class__)
            module._ca_tome_info = diffusion_model._ca_tome_info

            # Initialize caching info for adaptive token merging
            module._cache_info = {
                "compute_on": compute_on,  # Timesteps to recompute merging
                "merge": None,  # Cached merge information
                "unmerge": None,  # Cached unmerge information
            }

            # Handle model-specific compatibility flags
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False  # For SD 2.0 compatibility

            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                # For older diffusers versions
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model


def remove_patch(model: torch.nn.Module):
    """Removes a patch from a CA-ToMe Diffusion module if it was already patched."""
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_ca_tome_info"):
            for hook in module._ca_tome_info["hooks"]:
                hook.remove()
            module._ca_tome_info["hooks"].clear()

        if module.__class__.__name__ == "AdaptiveToMeBlock":
            module.__class__ = module._parent

    return model
