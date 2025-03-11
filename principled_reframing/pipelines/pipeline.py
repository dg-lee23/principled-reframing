# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union, Any, Dict
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import einops
import imageio
import matplotlib.pyplot as plt
import yaml


from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
import pdb

from ..utils.xformer_attention import *
from ..utils.conv_layer import *
from ..utils.util import *
from ..utils.util import _in_step, _classify_blocks, ddim_inversion

from .additional_components import *

logger = logging.get_logger(__name__) 



@dataclass
class PR_PipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class PR_Pipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)
        
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def invert(self,
            video = None,
            config: omegaconf.dictconfig = None,
            save_path = None,
            ):
        # perform DDIM inversion 
        import time
        start_time = time.time()
        generator = None
        video_latent = self.vae.encode(video.to(self.vae.dtype).to(self.vae.device)).latent_dist.sample(generator)
        video_latent = self.vae.config.scaling_factor * video_latent
        video_latent = video_latent.unsqueeze(0)
        video_latent = einops.rearrange(video_latent, "b f c h w -> b c f h w")                                                                 
        ddim_latents_dict, cond_embeddings = ddim_inversion(self, self.scheduler, video_latent, config.num_inference_step, config.inversion_prompt)
        
        end_time = time.time()
        print("Inversion time", end_time - start_time)

        video_data: Dict = {
            'inversion_prompt': config.inversion_prompt,
            'all_latents_inversion': ddim_latents_dict,
            'raw_video': video,
            'inversion_prompt_embeds': cond_embeddings,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(video_data, f)
                 
    
    def get_attn_features(self, index_select=None):

        # attn_prob_dic = {}
        query_dic = {}
        key_dic = {}
        value_dic = {}

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.temp_guidance.blocks, name):
                key = module.processor.key
                query = module.processor.query
                value = module.processor.value

                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                    index_all = torch.arange(key.shape[0])
                    index_picked = index_all[get_index.bool()]
                    key = key[index_picked]
                    query = query[index_picked]
                    value = value[index_picked]

                # * [hw, f, n_heads * head_dim] -> [hw * n_heads, f, head_dim]
                key = module.reshape_heads_to_batch_dim(key).contiguous()
                query = module.reshape_heads_to_batch_dim(query).contiguous()
                value = module.reshape_heads_to_batch_dim(value).contiguous()

                # * uncomment below to also retrieve the attention matrix
                # attention_probs = module.get_attention_scores(query, key, None)         
                # attention_probs = attention_probs.reshape(-1, module.heads,attention_probs.shape[1], attention_probs.shape[2])
                  
                # attn_prob_dic[name] = attention_probs
                key_dic[name] = key
                query_dic[name] = query
                value_dic[name] = value

        return query_dic, key_dic, value_dic


    def get_attn_component(self, index_select=None, component='v'):

        component_dict = {}
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__

            # filter temp attn blocks
            if "VersatileAttention" in module_name and _classify_blocks(self.input_config.temp_guidance.blocks, name):
                
                # f: feature component (among key, query, value, conv)
                if component == 'v':
                    f = module.processor.value
                elif component == 'q':
                    f = module.processor.query
                elif component == 'k':
                    f = module.processor.key
                elif component == 'h':
                    f = module.processor.hidden_state
                else: 
                    raise NotImplementedError

                if index_select is not None:
                    get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=f.shape[0]//len(index_select))
                    index_all = torch.arange(f.shape[0])
                    index_picked = index_all[get_index.bool()]
                    f = f[index_picked]
                
                component_dict[name] = f

        return component_dict
        
    def __call__(
        self,
        config: omegaconf.dictconfig = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noisy_latents: Optional[torch.FloatTensor] = None,
        inversion_data_path: str = None,
        transformation: str = None,
        shift_dir: str = None,
        resize_factor : float = 1.0,
        ac : str = None,
    ):
        # assert config is not None, "config is required for FreeControl pipeline"
        if not hasattr(self, 'config'):
            setattr(self, 'input_config', config)
        self.input_config = config
        if not hasattr(self, 'video_name'):
            setattr(self, 'video_name', config.video_path.split('/')[-1].split('.')[0])
        self.video_name = config.video_path.split('/')[-1].split('.')[0]

        self.unet = prep_unet_attention(self.unet)
        self.unet = prep_unet_conv(self.unet)
        
        # 0. Default height and width to unet
        height = config.height or self.unet.config.sample_size * self.vae_scale_factor
        width = config.width or self.unet.config.sample_size * self.vae_scale_factor
        video_length = config.video_length

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        
        # perform classifier_free_guidance in default
        cfg_scale = config.cfg_scale or 7.5
        do_classifier_free_guidance = True
        
        # 3. Encode input prompt
        new_prompt = config.new_prompt if isinstance(config.new_prompt, list) else [config.new_prompt] * batch_size
        negative_prompt = config.negative_prompt or ""
        negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(new_prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt)
        # [uncond_embeddings, text_embeddings] [2, 77, 768]
        
        global_app_guidance = True 
        token_index_example, token_index_app = None, None


        num_inference_step = config.num_inference_step or 300
        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        control_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            noisy_latents,
        )

        latents_group = torch.cat([control_latents, control_latents],dim=0)

        with open(inversion_data_path, "rb") as f:
            inverted_data = pickle.load(f)
            all_latents = inverted_data['all_latents_inversion']
            example_prompt_embeds = inverted_data['inversion_prompt_embeds'].to(device)

        # latents_group = torch.cat([all_latents[999], all_latents[999]], dim=0)

        self.temp_attn_prob_dic = {} # for record usage

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        example_latents = control_latents
        
        with self.progress_bar(total=num_inference_step) as progress_bar:
            for step_index, step_t in enumerate(self.scheduler.timesteps):
                global step_idx
                step_idx = step_index
                
                if step_index < self.input_config.guidance_step:
                    if step_t.item() not in all_latents.keys():
                        raise IndexError("The inference step does not match the inversion step")
                    
                    example_latents = all_latents[step_t.item()].to(device=device, dtype=text_embeddings.dtype)
                
                latents_group = self.single_step_video(latents_group, step_index, step_t, example_latents, text_embeddings, example_prompt_embeds,
                                                        cfg_scale, extra_step_kwargs, 
                                                        transformation, resize_factor=resize_factor, shift_dir=shift_dir, ac=ac)                              
                
                progress_bar.update()
            
            control_latents = latents_group[[1]]

            # 8. Post-processing
            video = self.decode_latents(control_latents)
        return video


    def single_step_video(self, latents_group, step_index, step_t, example_latents, text_embeddings, example_prompt_embeds, 
                                    cfg_scale, extra_step_kwargs,
                                    transformation=None, resize_factor=1.0, shift_dir=None, ac=None):

        # NOTE: 
        #   - latents_group = [uncondition_latent, example_latent, control_latent]
        #   - a_c stands for attention components
        
        # * guidance-steps
        if step_index < self.input_config.guidance_step:

            # TODO: unsure if below actually does anything
            with torch.no_grad():
                latent_model_input = torch.cat([latents_group[[0]], example_latents.detach(), latents_group[[1]]], dim=0)

            latent_model_input[-1].requires_grad = True
            # 

            step_prompt_embeds = torch.cat([text_embeddings[[0]], example_prompt_embeds, text_embeddings[[1]]], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_t).detach() 
            latent_model_input.requires_grad = True

            # feed [uncodition_latent, example_latent] to unet
            with torch.no_grad():
                noise_pred_no_grad = self.unet(
                        latent_model_input[[0,1]], step_t, 
                        encoder_hidden_states=step_prompt_embeds[[0,1]],
                    ).sample.to(dtype=latents_group.dtype)


            # gather attention feature --- (1)
            a_c = ac
            if a_c == "special":
                q_dict_ex, k_dict_ex, v_dict_ex = self.get_attn_features(index_select=[0, 1])
            else:
                v_example_dict = self.get_attn_component(index_select=[0, 1], component='v') if 'v' in a_c else None
                k_example_dict = self.get_attn_component(index_select=[0, 1], component='k') if 'k' in a_c else None
                q_example_dict = self.get_attn_component(index_select=[0, 1], component='q') if 'q' in a_c else None
                h_example_dict = self.get_attn_component(index_select=[0, 1], component='h') if 'h' in a_c else None


            # feed [control_latent] to unet
            torch.cuda.empty_cache()
            unet_in = latent_model_input[[2]]
            noise_pred_control = self.unet(
                        unet_in, 
                        step_t, 
                        encoder_hidden_states=step_prompt_embeds[[2]],
                            ).sample.to(dtype=latents_group.dtype)

            
            # gather attention feature --- (2)
            v_control_dict = self.get_attn_component(component='v') if 'v' in a_c else None
            k_control_dict = self.get_attn_component(component='k') if 'k' in a_c else None
            q_control_dict = self.get_attn_component(component='q') if 'q' in a_c else None
            h_control_dict = self.get_attn_component(component='h') if 'h' in a_c else None

            
            # compute loss
            loss_value = 0

            if a_c == 'special':
                q_dict_control, k_dict_control, v_dict_control = self.get_attn_features()

                loss_value += A_loss(q_dict_ex, k_dict_ex, v_dict_ex,
                                             q_dict_control, k_dict_control, v_dict_control)
            else:
                for c in a_c:
                    if c == 'v':
                        loss_value += compute_f_loss(v_example_dict, v_control_dict, transformation = transformation, resize_factor = resize_factor, shift_dir = shift_dir, step_index=step_index)
                    elif c == 'k':
                        loss_value += compute_f_loss(k_example_dict, k_control_dict, transformation = transformation, resize_factor = resize_factor, shift_dir = shift_dir, step_index=step_index)
                    elif c == 'q':
                        loss_value += compute_f_loss(q_example_dict, q_control_dict, transformation = transformation, resize_factor = resize_factor, shift_dir = shift_dir, step_index=step_index)
                    elif c == 'h':
                        loss_value += compute_f_loss(h_example_dict, h_control_dict, transformation = transformation, resize_factor = resize_factor, shift_dir = shift_dir, step_index=step_index)


            loss_total = 100.0 * (loss_value)
            print(f"loss: {loss_total.item():.4f}")


            # guidance scheduling
            scale = 1 - step_index / (self.input_config.guidance_step)
            loss_total *= scale


            # grad backprop
            torch.cuda.empty_cache()
            gradient = torch.autograd.grad(loss_total, latent_model_input, allow_unused=True)[0] 
            gradient = gradient[[2]] 
            assert gradient is not None, f"Step {step_index}: grad is None"
            
            if self.input_config.grad_guidance_threshold is not None:
                threshold = self.input_config.grad_guidance_threshold
                gradient_clamped = torch.where(
                        gradient.abs() > threshold,
                        torch.sign(gradient) * threshold,
                        gradient
                    )
                score = gradient_clamped.detach()
            else:
                score = gradient.detach()

            # diffusion step
            noise_pred = noise_pred_control + cfg_scale * (noise_pred_control - noise_pred_no_grad[[0]]) 
            control_latents = self.scheduler.customized_step(noise_pred, step_t, latents_group[[1]], score=score,
                                                                guidance_scale=self.input_config.grad_guidance_scale,
                                                                indices=[0],
                                                                **extra_step_kwargs, return_dict=False)[0].detach() 
            
            return torch.cat([control_latents , control_latents], dim=0)
            

        # * non-guidance steps
        else:
            with torch.no_grad():
                latent_model_input = self.scheduler.scale_model_input(latents_group, step_t)
                noise_pred = self.unet(
                    latent_model_input, step_t, 
                    encoder_hidden_states=text_embeddings,
                ).sample.to(dtype=latents_group.dtype)

                noise_pred = noise_pred[[1]] + cfg_scale * (noise_pred[[1]] - noise_pred[[0]])
                control_latents = self.scheduler.customized_step(noise_pred, step_t, latents_group[[1]], score=None,
                                                guidance_scale=self.input_config.grad_guidance_scale,
                                                indices=[0],
                                                **extra_step_kwargs, return_dict=False)[0]
            return torch.cat([control_latents , control_latents],dim=0)
        
