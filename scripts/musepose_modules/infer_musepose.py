import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import glob
import torch.nn.functional as F

from scripts.musepose_modules.musepose.models.pose_guider import PoseGuider
from scripts.musepose_modules.musepose.models.unet_2d_condition import UNet2DConditionModel
from scripts.musepose_modules.musepose.models.unet_3d import UNet3DConditionModel
from scripts.musepose_modules.musepose.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from scripts.musepose_modules.musepose.utils.util import get_fps, read_frames, save_videos_grid
from scripts.musepose_modules.paths import *
from modules import safe

def scale_video(video, width, height):
    video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
    scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
    scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                     width)  # [batch, frames, channels, height, width]

    return scaled_video


def infer_musepose(
    ref_image_path: str,
    pose_video_path: str,
    weight_dtype: str,
    W: int,
    H: int,
    L: int,
    S: int,
    O: int,
    cfg: float,
    seed: int,
    steps: int,
    fps: int,
    skip: int
):
    if weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        os.path.join(models_dir, "sd-vae-ft-mse"),
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        os.path.join(models_dir, "sd-image-variations-diffusers"),
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = os.path.join(musepose_module_dir, "configs", "inference_v2.yaml")
    infer_config = OmegaConf.load(inference_config_path)

    torch.load = safe.unsafe_torch_load
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        Path(os.path.join(models_dir, "sd-image-variations-diffusers")),
        Path(os.path.join(models_dir, "MusePose","motion_module.pth")),
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        os.path.join(models_dir, "image_encoder")
    ).to(dtype=weight_dtype, device="cuda")
    torch.load = safe.load

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(seed)

    width, height = W, H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(os.path.join(models_dir, "MusePose","denoising_unet.pth"), map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(os.path.join(models_dir, "MusePose","reference_unet.pth"), map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(os.path.join(models_dir, "MusePose","pose_guider.pth"), map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    print('handle===', ref_image_path, pose_video_path)
    ref_name = Path(ref_image_path).stem
    pose_name = Path(pose_video_path).stem.replace("_kps", "")

    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    pose_list = []
    pose_tensor_list = []
    pose_images = read_frames(pose_video_path)
    src_fps = get_fps(pose_video_path)
    print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
    L = min(L, len(pose_images))
    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    original_width, original_height = 0, 0

    pose_images = pose_images[::skip + 1]
    print("processing length:", len(pose_images))
    src_fps = src_fps // (skip + 1)
    print("fps", src_fps)
    L = L // ((skip + 1))

    for pose_image_pil in pose_images[: L]:
        pose_tensor_list.append(pose_transform(pose_image_pil))
        pose_list.append(pose_image_pil)
        original_width, original_height = pose_image_pil.size
        pose_image_pil = pose_image_pil.resize((width, height))

    # repeart the last segment
    last_segment_frame_num = (L - S) % (S - O)
    repeart_frame_num = (S - O - last_segment_frame_num) % (S - O)
    for i in range(repeart_frame_num):
        pose_list.append(pose_list[-1])
        pose_tensor_list.append(pose_tensor_list[-1])

    ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
    ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L)

    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)
    pose_tensor = pose_tensor.unsqueeze(0)

    video = pipe(
        ref_image_pil,
        pose_list,
        width,
        height,
        len(pose_list),
        steps,
        cfg,
        generator=generator,
        context_frames=S,
        context_stride=1,
        context_overlap=O,
    ).videos

    m1 = os.path.join(models_dir, "MusePose","pose_guider.pth").split('.')[0].split('/')[-1] #check
    m2 = os.path.join(models_dir, "MusePose","motion_module.pth").split('.')[0].split('/')[-1]

    save_dir_name = f"{time_str}-{cfg}-{m1}-{m2}"

    result = scale_video(video[:, :, :L], original_width, original_height)
    save_videos_grid(
        result,
        os.path.join(final_output_dir, f"{save_dir_name}.mp4"),
        n_rows=1,
        fps=src_fps if fps is None else fps,
    )

    video = torch.cat([ref_image_tensor, pose_tensor[:, :, :L], video[:, :, :L]], dim=0)
    video = scale_video(video, original_width, original_height)
    output_path = os.path.join(final_output_dir, f"{ref_name}_{pose_name}_{cfg}_{steps}_{skip}_{m1}_{m2}.mp4")
    save_videos_grid(
        video,
        output_path,
        n_rows=3,
        fps=src_fps if fps is None else fps,
    )

    return output_path


