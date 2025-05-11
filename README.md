# Principled Reframing in Motion Transfer with Video Diffusion Models  

Official implementation of **["Principled Reframing in Motion Transfer with Video Diffusion Models"](https://dg-lee23.github.io/principled-reframing-project-page/)**.

We implement *reframing* in existing **motion transfer** methods, such as **[MotionClone](https://github.com/LPengYang/MotionClone)**, **[Ctrl-x](https://github.com/genforce/ctrl-x)**, and **[DMT](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer)**. 

Currently, this repo supports incorporation of reframing into DMT and our own guidance-based motion transfer method. For Ctrl-x, see **[here](https://github.com/oh470102/ctrl-x-with-AnimateDiff)**. Further details and code will be released soon.

First, install all necessary dependencies by creating the environment,
```
conda env create -f environment.yaml
```
then installing additional libraries via pip.
```
pip install -r requirements.txt
```
Then, first install the necessary models.
1. create `/models` in the current repo (not to be confused with ./principled_reframing/models).
2. install **[RealisticVision](https://civitai.com/models/4201?modelVersionId=130072)**, and put it under `/models/DreamBooth_LoRA`
3. install `v3_sd15_adapter.ckpt` and `v3_sd15_mm.ckpt` from **[AnimateDiff](https://huggingface.co/guoyww/animatediff/tree/main)**, and put it under `/models/Motion_Module`
4. install **[Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)** and put it under `/models/StableDiffusion/stable-diffusion-v1-5`

See `runners/example.sh` for example usages.

Our code is largely based on **[MotionClone](https://github.com/LPengYang/MotionClone)**. We highly thank them for open-sourcing. 

---
