from dataclasses import dataclass
import os
import pickle
import numpy as np
import torch
import omegaconf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Callable, List, Optional, Union, Any, Dict
from diffusers.utils import deprecate, logging, BaseOutput, randn_tensor
from ..utils.xformer_attention import *
from ..utils.conv_layer import *
from ..utils.util import *
import kornia
import gc
import torchvision.transforms as transforms
from einops import rearrange

pad_mode = 'center'

@torch.no_grad()
def customized_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,

        # Guidance parameters
        score=None,
        guidance_scale=0.0,
        indices=None, 

):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # Support IF models
    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = self._get_variance(timestep, prev_timestep)

    # setting eta = 0.0 is also fine
    eta = 0.025
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5) # [2, 4, 64, 64]

    # 6. apply guidance following the formula (14) from https://arxiv.org/pdf/2105.05233.pdf
    if score is not None and guidance_scale > 0.0: 

        # if eta = 0, use line (1), otherwise, use (2) - see (https://arxiv.org/abs/2312.14091)
        # score = std_dev_t * score                             # (1) 
        score = std_dev_t * score / (torch.std(score) + 1e-7)   # (2)

        if indices is not None:
            assert pred_epsilon[indices].shape == score.shape, "pred_epsilon[indices].shape != score.shape"
            pred_epsilon[indices] = pred_epsilon[indices] - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score 
        else:
            assert pred_epsilon.shape == score.shape
            pred_epsilon = pred_epsilon - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score

    # 7. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon 

    # 8. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction 

    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                " `variance_noise` stays `None`."
            )

        if variance_noise is None:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        variance = std_dev_t * variance_noise 

        prev_sample = prev_sample + variance 
    self.pred_epsilon = pred_epsilon
    if not return_dict:
        return (prev_sample,)

    return prev_sample, pred_original_sample

def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None,timestep_spacing_type= "linspace"):
    """
    Sets the discrete timesteps used for the diffusion chain (to be run before inference).

    Args:
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model.
    """

    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
            f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.config.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps

    # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
    if timestep_spacing_type == "linspace":
        timesteps = (
            np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
    elif timestep_spacing_type == "leading":
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps += self.config.steps_offset
    elif timestep_spacing_type == "trailing":
        step_ratio = self.config.num_train_timesteps / self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
        timesteps -= 1
    else:
        raise ValueError(
            f"{timestep_spacing_type} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
        )

    self.timesteps = torch.from_numpy(timesteps).to(device)

def compute_f_loss(f_example_dict, f_control_dict, transformation=None, 
                    resize_factor = 1.0, shift_dir=None, step_index=None):
    
    losses = []

    for name in f_example_dict.keys():

        f_example = f_example_dict[name].detach()
        f_control = f_control_dict[name]

        hw, f, dim = f_example.shape
        h = w = int(hw**0.5); assert h*w == hw

        # ? resize
        resize = False
        if transformation == 'resize': 
            resize = True
        if resize:
            scale_factor = resize_factor; s = scale_factor ** 0.5

            f_example = f_example.view(h, w, f, dim)
            f_example = f_example.permute(2, 3, 0, 1)
            f_example = F.interpolate(f_example, size=(int(h*s), int(w*s)), mode='bicubic', align_corners=True)
            
            if scale_factor <= 1:
                f_example = pad_tensor(f_example, target_size=(f, dim, h, w), mode=pad_mode) 
            else:
                f_example = crop_tensor(f_example, target_size=(f, dim, h, w), mode=pad_mode)

            f_example = f_example.permute(2, 3, 0, 1)

            if scale_factor <= 1:
                f_example = f_example.view(hw, f, dim)
            else:
                f_example = f_example.reshape(hw, f, dim)

        # ? shift
        shift = False
        if transformation == 'shift':
            shift = True
        if shift:

            if shift_dir == 'up':
                shift_by = (h//5, 0)

            elif shift_dir == 'left':
                shift_by = (0, w//4)

            elif shift_dir == 'down':
                shift_by = (-h//4, 0)
            
            elif shift_dir == 'right':
                shift_by = (0, -w//4)

            f_example = shift_tensor(f_example.unsqueeze(dim=-1), shift=shift_by).view(hw, f, dim)
        
        # ? perspective warp
        warp = False
        if transformation == 'warp':
            warp = True
            warp_dir = shift_dir
            warp_scale = resize_factor
        if warp:
            d = int(h*0.2)
            if warp_dir == 'right':
                dst_pts = [(0, 0), (h-1, 0), (h-1, h-1), (0, h-1)]
                src_pts = [(0, 0), (h-1, d), (h-1, h-1-d), (0, h-1)]
            elif warp_dir == 'left':
                dst_pts = [(0, 0), (h-1, 0), (h-1, h-1), (0, h-1)]
                src_pts = [(0, d), (h-1, 0), (h-1, h-1), (0, h-1 - d)]
            elif warp_dir == 'down':
                dst_pts = [(0, 0), (h-1, 0), (h-1, h-1), (0, h-1)]
                src_pts = [(0, 0), (h-1, 0), (h-1-d, h-1-d), (d, h-1 - d)]

            dst_pts = torch.tensor(dst_pts, dtype=torch.float, device='cuda').unsqueeze(dim=0)
            src_pts = torch.tensor(src_pts, dtype=torch.float, device='cuda').unsqueeze(dim=0)

            # get H
            H = kornia.geometry.get_perspective_transform(src_pts, dst_pts).float()

            # prepare f
            f_example = f_example.view(h, w, f, dim)
            f_example = f_example.permute(2, 3, 0, 1).view(f * dim, 1, h, w)

            # f <- H * f
            H_batch = H.repeat(f * dim, 1, 1)  
            for _ in range(int(warp_scale)):
                f_example = kornia.geometry.warp_perspective(f_example.float(), H_batch, dsize=(h,w), mode='bicubic', align_corners=True)
            f_example = f_example.view(f, dim, h, w)
            f_example = f_example.permute(2, 3, 0, 1).view(hw, f, dim).half()


        # compute loss
        nonzero_mask = f_example > 1e-3
        module_attn_loss = F.mse_loss(input=f_control[nonzero_mask], target=f_example[nonzero_mask])
        losses.append(module_attn_loss)
            

    loss_value = torch.stack(losses) 

    return 2*loss_value.mean()

def A_loss(q_example_dict, k_example_dict, v_example_dict,
                q_control_dict, k_control_dict, v_control_dict,
                ):
    
    losses = []
    n_heads = 8

    for name in q_example_dict.keys():

        # 1. get q, k
        q_example = q_example_dict[name].detach()      # [hw * n_heads, f, head_dim]
        q_control = q_control_dict[name]

        k_example = k_example_dict[name].detach()
        k_control = k_control_dict[name]

        v_example = v_example_dict[name].detach()
        v_control = v_control_dict[name]

        # 2. extract dim
        hw_, f, head_dim = q_example.shape
        hw = (hw_)//n_heads; h = w = int(hw**0.5)

        # ? shift
        shift = False
        shift_dir = 'right'
        if shift:
            if shift_dir == 'up':
                shift_by = (h//4, 0)
            elif shift_dir == 'left':
                shift_by = (0, w//5)
            elif shift_dir == 'down':
                shift_by = (-h//3, 0)
            elif shift_dir == 'right':
                shift_by = (0, -w//4)
            else:
                shift_by = (0,0)

            # 3. shift q,k
            q_example = q_example.view(hw, n_heads, f, head_dim)
            k_example = k_example.view(hw, n_heads, f, head_dim)
            v_example = v_example.view(hw, n_heads, f, head_dim)
            q_example = shift_tensor(q_example, shift=shift_by, fill=0).view(hw_, f, head_dim) # expects (hw, *, *, *)
            k_example = shift_tensor(k_example, shift=shift_by, fill=0).view(hw_, f, head_dim)
            v_example = shift_tensor(v_example, shift=shift_by, fill=0).view(hw_, f, head_dim)


        # ? resize
        resize = False
        resize_factor = 0.8
        if resize:
            s = resize_factor ** 0.5
            dim = n_heads * head_dim

            q_example = q_example.view(hw, n_heads, f, head_dim)
            k_example = k_example.view(hw, n_heads, f, head_dim)
            v_example = v_example.view(hw, n_heads, f, head_dim)

            q_example = rearrange(q_example, 'hw n_heads f head_dim -> f n_heads head_dim hw').contiguous()
            k_example = rearrange(k_example, 'hw n_heads f head_dim -> f n_heads head_dim hw').contiguous()
            v_example = rearrange(v_example, 'hw n_heads f head_dim -> f n_heads head_dim hw').contiguous()

            q_example = q_example.view(f, n_heads * head_dim, hw).view(f, dim, h, w)
            k_example = k_example.view(f, n_heads * head_dim, hw).view(f, dim, h, w)
            v_example = v_example.view(f, n_heads * head_dim, hw).view(f, dim, h, w)

            # expects (b, c, h, w)
            q_example = F.interpolate(q_example, size=(int(h*s), int(w*s)), mode='bicubic', align_corners=True)
            k_example = F.interpolate(k_example, size=(int(h*s), int(w*s)), mode='bicubic', align_corners=True)
            v_example = F.interpolate(v_example, size=(int(h*s), int(w*s)), mode='bicubic', align_corners=True)

            if resize_factor <= 1:
                q_example = pad_tensor(q_example, target_size=(f, dim, h, w), mode=pad_mode) 
                k_example = pad_tensor(k_example, target_size=(f, dim, h, w), mode=pad_mode) 
                v_example = pad_tensor(v_example, target_size=(f, dim, h, w), mode=pad_mode) 
            else:
                q_example = crop_tensor(q_example, target_size=(f, dim, h, w), mode=pad_mode)
                k_example = crop_tensor(k_example, target_size=(f, dim, h, w), mode=pad_mode)
                v_example = crop_tensor(v_example, target_size=(f, dim, h, w), mode=pad_mode)


            q_example = q_example.reshape(f, n_heads, head_dim, hw)
            k_example = k_example.reshape(f, n_heads, head_dim, hw)
            v_example = v_example.reshape(f, n_heads, head_dim, hw)

            q_example = rearrange(q_example, 'f n_heads head_dim hw -> hw n_heads f head_dim').contiguous()
            k_example = rearrange(k_example, 'f n_heads head_dim hw -> hw n_heads f head_dim').contiguous()
            v_example = rearrange(v_example, 'f n_heads head_dim hw -> hw n_heads f head_dim').contiguous()

            if resize_factor <= 1:
                q_example = q_example.view(hw_, f, head_dim)
                k_example = k_example.view(hw_, f, head_dim)
                v_example = v_example.view(hw_, f, head_dim)
            else:
                q_example = q_example.reshape(hw_, f, head_dim)
                k_example = k_example.reshape(hw_, f, head_dim)
                v_example = v_example.reshape(hw_, f, head_dim)


        # 3. compute A
        A_example = get_A(q_example, k_example)
        A_example = A_example.view(-1, n_heads, f, f)  # [hw, n_heads, f, f]

        A_control = get_A(q_control, k_control)
        A_control = A_control.view(-1, n_heads, f, f)

        # motionclone-style masking
        rank_k = 1
        _, sorted_indices = torch.sort(A_example, dim=-1)
        mask_indices = torch.cat(
            (torch.zeros([*A_example.shape[:-1], A_example.shape[-1] - rank_k], dtype=torch.bool),
            torch.ones([*A_example.shape[:-1], rank_k], dtype=torch.bool)),
            dim=-1
        )
        max_copy = sorted_indices[..., [-1]].expand_as(A_example)
        sorted_indices[~mask_indices] = max_copy[~mask_indices]
        mask = torch.zeros_like(A_example, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, True)

        nonzero_mask = (A_example != 0)
        final_mask = nonzero_mask & mask

        module_attn_loss = F.mse_loss(input=A_example[final_mask], target=A_control[final_mask])
        losses.append(3 * module_attn_loss)
            
    loss = torch.stack(losses) 

    return loss.mean()



# * ============================================ * #
# * utils (below)

def shift_tensor(tensor, shift=None, fill=0):
    
    # expects input shape (hw, __, __, __)

    # setup
    hw, n_head, f1, f2 = tensor.shape
    res = h = w = int(hw**0.5)

    # positive: up, left
    if shift is None: shift = (-h//4, 0)
    shift_x, shift_y = shift

    # roll but with no circular shifting
    tensor = tensor.view(res, res, n_head, f1, f2)
    if shift_x != 0:
        if shift_x > 0:
            tensor = torch.cat((tensor[shift_x:, :, :, :, :], torch.zeros_like(tensor[:shift_x, :, :, :, :]) + fill), dim=0)
        else:
            shift_x = -shift_x
            tensor = torch.cat((torch.zeros_like(tensor[-shift_x:, :, :, :, :]), tensor[:-shift_x, :, :, :, :] + fill), dim=0)

    if shift_y != 0:
        if shift_y > 0:
            tensor = torch.cat((tensor[:, shift_y:, :, :, :], torch.zeros_like(tensor[:, :shift_y, :, :, :]) + fill), dim=1)
        else:
            shift_y = -shift_y
            tensor = torch.cat((torch.zeros_like(tensor[:, -shift_y:, :, :, :]), tensor[:, :-shift_y, :, :, :] + fill), dim=1)

    # Reshape back to original shape
    shifted_tensor = tensor.view(res, res, n_head, f1, f2)
    
    return shifted_tensor
    
def pad_tensor(tensor, target_size, mode='center'):

    # ? check expected input shape: (B, C, H, W)
    _, _, H, W = tensor.shape
    target_H, target_W = target_size[2], target_size[3]

    if mode=='center':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    elif mode=='top-left':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = 0
        pad_right = target_W - W
    
    elif mode=='top-right':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = target_W - W
        pad_right = 0

    elif mode=='bottom-left':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = 0
        pad_right = target_W - W

    elif mode == 'bottom-right':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = target_W - W
        pad_right = 0

    elif mode == 'center-left':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = 0
        pad_right = target_W - W

    elif mode == 'center-right':
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = target_W - W
        pad_right = 0

    elif mode == 'center-top':
        pad_top = 0
        pad_bottom = target_H - H
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    elif mode == 'center-bottom':
        pad_top = target_H - H
        pad_bottom = 0
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)

    return F.pad(tensor, padding, mode='constant', value=0)

def crop_tensor(tensor, target_size, mode='center'):

    _, _, H, W = tensor.shape
    target_H, target_W = target_size[2], target_size[3]

    if mode == 'center':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    elif mode == 'top-left':
        crop_top = 0
        crop_bottom = target_H
        crop_left = 0
        crop_right = target_W
    
    elif mode == 'top-right':
        crop_top = 0
        crop_bottom = target_H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'bottom-left':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = 0
        crop_right = target_W

    elif mode == 'bottom-right':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'center-left':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = 0
        crop_right = target_W

    elif mode == 'center-right':
        crop_top = (H - target_H) // 2
        crop_bottom = crop_top + target_H
        crop_left = W - target_W
        crop_right = W

    elif mode == 'center-top':
        crop_top = 0
        crop_bottom = target_H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    elif mode == 'center-bottom':
        crop_top = H - target_H
        crop_bottom = H
        crop_left = (W - target_W) // 2
        crop_right = crop_left + target_W

    # Perform the cropping operation
    cropped_tensor = tensor[:, :, crop_top:crop_bottom, crop_left:crop_right]

    return cropped_tensor

def get_A(q, k):

    q = q.float()
    k = k.float()
    A = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1]).to('cuda').float(),
            q,
            k.transpose(-1, -2),
            beta=0,
            alpha=float(q.shape[-1])**(-0.5)
        ).float()
    # A = torch.softmax(A, dim=-1).half()    # optional to use A before/after softmax
    return A.float()
    




