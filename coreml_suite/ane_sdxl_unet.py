from __future__ import annotations

"""ANE-friendly SDXL UNet implementation.

The stock SDXL UNet shipped with ``apple/ml-stable-diffusion`` does not expose
hooks that make it easy to evolve towards Apple Neural Engine execution.  This
module introduces a thin subclass that keeps parity with the original network
while enabling two ANE-focused extensibility points:

* Optional ``timestep_cond`` support so schedulers that rely on time conditioning
  (e.g. LCM or other fast samplers) can feed the auxiliary embedding without
  rewriting the Core ML graph.
* Explicit ``support_controlnet`` toggling, allowing ControlNet residuals to be
  merged even when the upstream configuration omits the flag.

The forward pass mirrors ``UNet2DConditionModelXL`` and therefore can be swapped
in as a drop-in replacement during Core ML export.
"""

import torch
from overrides import overrides
from python_coreml_stable_diffusion.unet import (
    TimestepEmbedding,
    UNet2DConditionModelXL,
)


class UNet2DConditionModelXLANE(UNet2DConditionModelXL):
    """Drop-in SDXL UNet variant with ANE-oriented hooks."""

    def __init__(
        self,
        *,
        support_controlnet: bool | None = None,
        time_cond_proj_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Rebuild the time embedding block when a conditional projection is
        # requested.  This mirrors the LCM-specific UNet implementation.
        if time_cond_proj_dim is not None:
            timestep_input_dim = self.config.block_out_channels[0]
            time_embed_dim = timestep_input_dim * 4
            self.time_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim, cond_proj_dim=time_cond_proj_dim
            )

        # ``support_controlnet`` is set on the base class during conversion.  In
        # practice this flag is often missing for SDXL exports, so allow callers
        # to force-enable the residual path merging logic.
        if support_controlnet is not None:
            self.support_controlnet = support_controlnet

    @overrides(check_signature=False)
    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        time_ids,
        text_embeds,
        *additional_residuals,
        timestep_cond=None,
    ):
        # 0. Project time embeddings
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        aug_emb = None

        if self.config.addition_embed_type == "text":
            raise NotImplementedError
        elif self.config.addition_embed_type == "text_image":
            raise NotImplementedError
        elif self.config.addition_embed_type == "text_time":
            assert time_ids is not None
            assert text_embeds is not None

            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            raise NotImplementedError
        elif self.config.addition_embed_type == "image_hint":
            raise NotImplementedError

        emb = emb + aug_emb if aug_emb is not None else emb

        # 1. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if getattr(self, "support_controlnet", False):
            new_down_block_res_samples = ()
            for i, down_block_res_sample in enumerate(down_block_res_samples):
                down_block_res_sample = down_block_res_sample + additional_residuals[i]
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        if getattr(self, "support_controlnet", False):
            sample = sample + additional_residuals[-1]

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return (sample,)
