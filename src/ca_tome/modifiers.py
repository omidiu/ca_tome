from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from .patch import apply_patch

import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    BaseOutput,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


def register_inference_step(block, inference_step):
    block_names = [
        "CrossAttnDownBlock2D",
        "UNetMidBlock2DCrossAttn",
        "CrossAttnUpBlock2D",
    ]
    if block.__class__.__name__ in block_names:
        for attn in block.attentions:
            for transformer in attn.transformer_blocks:
                transformer.inference_step = inference_step


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def modify_unet(unet):
    def reg_new_forward(self):
        def forward(
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            inference_step: int,
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, Tuple]:
            r"""
            The [`UNet2DConditionModel`] forward method.

            Args:
                sample (`torch.Tensor`):
                    The noisy input tensor with the following shape `(batch, channel, height, width)`.
                timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.Tensor`):
                    The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
                class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                    Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
                timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                    Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                    through the `self.time_embedding` layer to obtain the timestep embeddings.
                attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                    An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                    is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                    negative values to the attention scores corresponding to "discard" tokens.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                added_cond_kwargs: (`dict`, *optional*):
                    A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                    are passed along to the UNet blocks.
                down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                    A tuple of tensors that if specified are added to the residuals of down unet blocks.
                mid_block_additional_residual: (`torch.Tensor`, *optional*):
                    A tensor that if specified is added to the residual of the middle unet block.
                down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                    additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
                encoder_attention_mask (`torch.Tensor`):
                    A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                    `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                    which adds large negative values to the attention scores corresponding to "discard" tokens.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                    tuple.

            Returns:
                [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                    If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is the sample tensor.
            """
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            for dim in sample.shape[-2:]:
                if dim % default_overall_up_factor != 0:
                    # Forward upsample size to force interpolation output size.
                    forward_upsample_size = True
                    break

            # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
            # expects mask of shape:
            #   [batch, key_tokens]
            # adds singleton query_tokens dimension:
            #   [batch,                    1, key_tokens]
            # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
            #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
            #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
            if attention_mask is not None:
                # assume that mask is expressed as:
                #   (1 = keep,      0 = discard)
                # convert mask into a bias that can be added to attention scores:
                #       (keep = +0,     discard = -10000.0)
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(sample.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            # 0. center input if necessary
            if self.config.center_input_sample:
                sample = 2 * sample - 1.0

            # 1. time
            t_emb = self.get_time_embed(sample=sample, timestep=timestep)
            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
            if class_emb is not None:
                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            aug_emb = self.get_aug_embed(
                emb=emb,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            )
            if self.config.addition_embed_type == "image_hint":
                aug_emb, hint = aug_emb
                sample = torch.cat([sample, hint], dim=1)

            emb = emb + aug_emb if aug_emb is not None else emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            encoder_hidden_states = self.process_encoder_hidden_states(
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            )

            # 2. pre-process
            sample = self.conv_in(sample)

            # 2.5 GLIGEN position net
            if (
                cross_attention_kwargs is not None
                and cross_attention_kwargs.get("gligen", None) is not None
            ):
                cross_attention_kwargs = cross_attention_kwargs.copy()
                gligen_args = cross_attention_kwargs.pop("gligen")
                cross_attention_kwargs["gligen"] = {
                    "objs": self.position_net(**gligen_args)
                }

            # 3. down
            # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
            # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
            if cross_attention_kwargs is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                lora_scale = cross_attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)

            is_controlnet = (
                mid_block_additional_residual is not None
                and down_block_additional_residuals is not None
            )
            # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
            is_adapter = down_intrablock_additional_residuals is not None
            # maintain backward compatibility for legacy usage, where
            #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
            #       but can only use one or the other
            if (
                not is_adapter
                and mid_block_additional_residual is None
                and down_block_additional_residuals is not None
            ):
                deprecate(
                    "T2I should not use down_block_additional_residuals",
                    "1.3.0",
                    "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                           and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                           for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                    standard_warn=False,
                )
                down_intrablock_additional_residuals = down_block_additional_residuals
                is_adapter = True

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:

                register_inference_step(downsample_block, inference_step)

                if (
                    hasattr(downsample_block, "has_cross_attention")
                    and downsample_block.has_cross_attention
                ):
                    # For t2i-adapter CrossAttnDownBlock2D
                    additional_residuals = {}
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        additional_residuals["additional_residuals"] = (
                            down_intrablock_additional_residuals.pop(0)
                        )

                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample, temb=emb
                    )
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        sample += down_intrablock_additional_residuals.pop(0)

                down_block_res_samples += res_samples

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = (
                        down_block_res_sample + down_block_additional_residual
                    )
                    new_down_block_res_samples = new_down_block_res_samples + (
                        down_block_res_sample,
                    )

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:

                register_inference_step(self.mid_block, inference_step)

                if (
                    hasattr(self.mid_block, "has_cross_attention")
                    and self.mid_block.has_cross_attention
                ):
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                sample = sample + mid_block_additional_residual

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):

                register_inference_step(upsample_block, inference_step)

                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[
                    : -len(upsample_block.resnets)
                ]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if (
                    hasattr(upsample_block, "has_cross_attention")
                    and upsample_block.has_cross_attention
                ):
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return (sample,)

            return UNet2DConditionOutput(sample=sample)

        return forward

    unet.forward = reg_new_forward(unet)


def modify_pipeline(pipe):
    def reg_new_call(self):
        @torch.no_grad
        def call(
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[
                Union[
                    Callable[[int, int, Dict], None],
                    PipelineCallback,
                    MultiPipelineCallbacks,
                ]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):
            r"""
            The call function to the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                timesteps (`List[int]`, *optional*):
                    Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                    in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                    passed will be used. Must be in descending order.
                sigmas (`List[float]`, *optional*):
                    Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                    their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                    will be used.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    A higher guidance scale value encourages the model to generate images closely linked to the text
                    `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                    pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                    to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.Tensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                    provided, text embeddings are generated from the `prompt` input argument.
                negative_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                    not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                    IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                    contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                    provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                    [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                guidance_rescale (`float`, *optional*, defaults to 0.0):
                    Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                    Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                    using zero terminal SNR.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                    A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                    each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                    DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                    list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.

            Examples:

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is a list with the generated images and the
                    second element is a list of `bool`s indicating whether the corresponding generated image contains
                    "not-safe-for-work" (nsfw) content.
            """

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )

            if isinstance(
                callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)
            ):
                callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None)
                if self.cross_attention_kwargs is not None
                else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas
            )

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 6.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
                else None
            )

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                    batch_size * num_images_per_prompt
                )
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=self.unet.config.time_cond_proj_dim,
                ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = (
                len(timesteps) - num_inference_steps * self.scheduler.order
            )
            self._num_timesteps = len(timesteps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        i + 1,  # Pass inference step start from zero
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop(
                            "prompt_embeds", prompt_embeds
                        )
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            if not output_type == "latent":
                image = self.vae.decode(
                    latents / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                image, has_nsfw_concept = self.run_safety_checker(
                    image, device, prompt_embeds.dtype
                )
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=do_denormalize
            )

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )

        return call

    pipe.call = reg_new_call(pipe)


def apply_CA_ToMe(pipe, cache_conf, r):
    """
    Sets up a diffusion pipeline with Cached Adaptive ToMe optimizations.

    Args:
        pipe: StableDiffusionPipeline, The diffusion pipeline to modify
        cache_conf: List[int], array of timestep checkpoints where new merge indices are computed.
                   These checkpoints specify when the CA-ToMe merging pattern is recalculated.
        r: float, cosine similarity threshold that determines whether tokens should be merged.
           Higher values mean tokens need to be more similar to be merged.

    Returns:
        The modified pipeline with CA-ToMe optimizations applied

    This function applies three modifications to the pipeline:
    1. Modifies the UNet to support cached adaptive ToMe during inference
    2. Modifies the pipeline's forward pass to handle CA-ToMe
    3. Applies the CA-ToMe patch with the specified configuration.
    """
    modify_unet(pipe.unet)
    modify_pipeline(pipe)
    apply_patch(pipe, cache_conf, r)
    return pipe
