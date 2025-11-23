"""ANE-oriented UNet implementation for SDXL.

The original project depends on the UNet implementation from
`python_coreml_stable_diffusion`. That implementation is tailored for SD 1.5
and leaves SDXL support in an "experimental" state. In practice the default
attention backend falls back to GPU, which prevents Core ML from targeting the
ANE. This module provides a small, SDXL-specific shim that forces an
ANE-friendly attention path and handles the SDXL conditioning inputs in a way
that is friendly to TorchScript tracing and Core ML conversion.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from python_coreml_stable_diffusion import attention
from python_coreml_stable_diffusion.unet import (
    AttentionImplementations,
    TimestepEmbedding,
    Timesteps,
    UNet2DConditionModelXL,
)


class UNet2DConditionModelXLANE(UNet2DConditionModelXL):
    """SDXL UNet variant with ANE defaults.

    The base ``UNet2DConditionModelXL`` class from ``python_coreml_stable_diffusion``
    does not force an ANE-compatible attention implementation, so a converted
    Core ML model can unexpectedly target the GPU. This subclass ensures that the
    split-einsum attention path is active and optionally casts all latent and
    conditioning inputs to ``float16`` before running the network. That casting
    keeps the TorchScript graph compatible with Core ML's ANE backend.
    """

    def __init__(
        self,
        *args: Any,
        attention_implementation: AttentionImplementations = AttentionImplementations.SPLIT_EINSUM_V2,
        cast_inputs_to_float16: bool = True,
        **kwargs: Any,
    ) -> None:
        # Force the attention backend before the parent initialises attention
        # processors. This mirrors the behaviour expected by the ANE compiler.
        attention.ATTENTION_IMPLEMENTATION_IN_EFFECT = attention_implementation
        super().__init__(*args, **kwargs)
        self.cast_inputs_to_float16 = cast_inputs_to_float16

        # ``python_coreml_stable_diffusion`` only constructs the extra SDXL time
        # embedding modules when ``addition_embed_type`` is set to "text_time" at
        # initialisation time. Some pipelines omit that field from the config,
        # which leaves ``add_time_proj``/``add_embedding`` undefined even though
        # the forward pass expects them. Recreate the pair when missing so
        # TorchScript tracing does not fail.
        add_type = getattr(self.config, "addition_embed_type", None)
        if add_type == "text_time":
            if not hasattr(self, "add_time_proj"):
                self.add_time_proj = Timesteps(
                    self.config.addition_time_embed_dim,
                    self.config.flip_sin_to_cos,
                    self.config.freq_shift,
                )

            if not hasattr(self, "add_embedding"):
                time_embed_dim = self.config.block_out_channels[0] * 4
                self.add_embedding = TimestepEmbedding(
                    self.config.projection_class_embeddings_input_dim,
                    time_embed_dim,
                )

    def _cast_if_needed(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None or not self.cast_inputs_to_float16:
            return tensor
        return tensor.to(torch.float16)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        time_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        *additional_residuals: torch.Tensor,
    ):
        # Make sure all conditioning pathways are explicitly float16 when
        # requested. The ANE kernels expect fp16 tensors and conversion inside
        # the graph avoids Core ML inserting GPU-only casts.
        sample = self._cast_if_needed(sample)
        encoder_hidden_states = self._cast_if_needed(encoder_hidden_states)
        text_embeds = self._cast_if_needed(text_embeds)
        time_ids = self._cast_if_needed(time_ids)
        additional_residuals = tuple(self._cast_if_needed(t) for t in additional_residuals)

        return super().forward(
            sample,
            timestep,
            encoder_hidden_states,
            time_ids,
            text_embeds,
            *additional_residuals,
        )


def build_sdxl_unet_for_ane(
    ref_unet_config: Any,
    attention_impl: AttentionImplementations = AttentionImplementations.SPLIT_EINSUM_V2,
    cast_inputs_to_float16: bool = True,
) -> UNet2DConditionModelXLANE:
    """Create an SDXL UNet pre-configured for ANE export.

    Args:
        ref_unet_config: The configuration from the reference Diffusers UNet.
        attention_impl: The attention implementation to inject before model
            construction. SPLIT_EINSUM_V2 yields the most ANE friendly graphs.
        cast_inputs_to_float16: Whether to eagerly cast latent and conditioning
            inputs to ``float16``.
    """

    return UNet2DConditionModelXLANE.from_config(
        ref_unet_config,
        attention_implementation=attention_impl,
        cast_inputs_to_float16=cast_inputs_to_float16,
    ).eval()
