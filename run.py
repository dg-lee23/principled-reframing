import os
import json
import torch
import argparse
import imageio
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange, repeat
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from principled_reframing.models.unet import UNet3DConditionModel
from principled_reframing.pipelines.pipeline import PR_Pipeline
from principled_reframing.pipelines.additional_components import customized_step, set_timesteps
from principled_reframing.utils.util import load_weights
from principled_reframing.utils.util import set_all_seed
import datetime

from transformers import logging
logging.set_verbosity_error()


def main(args):

    if not os.path.exists("results"):
        os.makedirs("results")
    
    config  = OmegaConf.load(args.config)
    adopted_dtype = torch.float16
    device = "cuda"

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)

    config.width = config.get("W")
    config.height = config.get("H")
    config.video_length = config.get("L")

    model_config = OmegaConf.load(config.get("model_config", args.model_config))
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs)).to(device).to(dtype=adopted_dtype)
    
    if args.num_inference_steps is not None:
        config.num_inference_step = args.num_inference_steps

    controlnet = None

    if is_xformers_available() and (not args.without_xformers):
        unet.enable_xformers_memory_efficient_attention()
        if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()
    pipeline = PR_Pipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet,
        scheduler = DDIMScheduler(**OmegaConf.to_container(model_config.noise_scheduler_kwargs)),
    ).to(device)

    pipeline = load_weights(
        pipeline,
        motion_module_path         = config.get("motion_module", ""),
        dreambooth_model_path      = config.get("dreambooth_path", ""),
    ).to(device)
    
    pipeline.scheduler.customized_step = customized_step.__get__(pipeline.scheduler)
    pipeline.scheduler.added_set_timesteps = set_timesteps.__get__(pipeline.scheduler)
    
    seed = config.get("seed", args.default_seed)
    set_all_seed(seed)
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(seed)
    pipeline.scheduler.added_set_timesteps(config.num_inference_step, device=device)

    if args.examples == None:

        if args.resize_factor is not None:
            config.resize_factor = args.resize_factor

        video_name = config.video_path.split('/')[-1].split('.')[0]
        inversion_data_path =  os.path.join(args.inversion_save_dir, f"inverted_data_{video_name}.pkl")
        videos = pipeline(
                        config = config,
                        generator = generator,
                        inversion_data_path = inversion_data_path,
                        transformation = args.transformation,
                        shift_dir = args.shift_dir,
                        resize_factor = args.resize_factor,
                        ac = args.ac
                    )
        videos = rearrange(videos, "b c f h w -> b f h w c")
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(args.save_dir,  config.new_prompt.replace(' ', '_') + f"_seed{seed}_{time}" + '.mp4') 

        videos_uint8 = (videos[0] * 255).astype(np.uint8)
        imageio.mimwrite(save_path, videos_uint8, fps=8, quality=9)

    else:
        examples = json.load(args.examples)
        for example_infor in examples:
            config.video_path = example_infor["video_path"]
            config.inversion_prompt = example_infor["inversion_prompt"]
            config.new_prompt = example_infor["new_prompt"]
            inversion_data_path =  os.path.join(args.inversion_save_dir, config.new_prompt.replace(' ', '_') + ".pkl")
            videos = pipeline(
                        config = config,
                        generator = generator,
                        inversion_data_path = inversion_data_path,
                    )
            videos = rearrange(videos, "b c f h w -> b f h w c")
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir,  config.new_prompt.replace(' ', '_') + f"_seed{seed}" + '.mp4')
            videos_uint8 = (videos[0] * 255).astype(np.uint8)
            imageio.mimwrite(save_path, videos_uint8, fps=8, quality=9)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--model-config",      type=str, default="configs/model_config.yaml")    
    parser.add_argument("--config",            type=str, default="configs/inference_config/generate.yaml")
    parser.add_argument("--examples",          type=str, default=None)
    parser.add_argument("--save_dir",          type=str, default="results/")
    parser.add_argument("--inversion_save_dir",type=str, default="inversion/")
    parser.add_argument("--default-seed", type=int, default=42)
    parser.add_argument("--without-xformers", action="store_true")

    parser.add_argument("--transformation", type=str, default=None)
    parser.add_argument("--resize_factor", type=float, default=None)
    parser.add_argument("--shift_dir", type=str, default=None)

    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--ac", type=str, default="v")
    parser.add_argument("--video_name", type=str, default=None)

    args = parser.parse_args()
    main(args)

