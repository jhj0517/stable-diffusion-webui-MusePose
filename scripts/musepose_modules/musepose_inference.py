import os
import argparse
from datetime import datetime
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import glob
import torch.nn.functional as F
import gc

from scripts.musepose_modules.musepose.models.pose_guider import PoseGuider
from scripts.musepose_modules.musepose.models.unet_2d_condition import UNet2DConditionModel
from scripts.musepose_modules.musepose.models.unet_3d import UNet3DConditionModel
from scripts.musepose_modules.musepose.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from scripts.musepose_modules.musepose.utils.util import get_fps, read_frames, save_videos_grid
from scripts.musepose_modules.downloading_weights import download_models
from scripts.musepose_modules.paths import *
from modules import safe


class MusePoseInference:
    def __init__(self):
        self.models_paths = {
            "pretrained_base_model": os.path.join(models_dir, "sd-image-variations-diffusers"),
            "pretrained_vae": os.path.join(models_dir, "sd-vae-ft-mse"),
            "image_encoder": os.path.join(models_dir, "image_encoder"),
            "denoising_unet": os.path.join(models_dir, "MusePose","denoising_unet.pth"),
            "reference_unet": os.path.join(models_dir, "MusePose","reference_unet.pth"),
            "pose_guider": os.path.join(models_dir, "MusePose","pose_guider.pth"),
            "motion_module": os.path.join(models_dir, "MusePose","motion_module.pth"),
            "inference_config": os.path.join(musepose_module_dir, "configs", "inference_v2.yaml")
        }
        self.vae = None
        self.reference_unet = None
        self.denoising_unet = None
        self.pose_guider = None
        self.image_enc = None
        self.pipe = None

    def infer_musepose(
        self,
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
        download_models()
        print('Width:', W)
        print('Height:', H)
        print('Length:', L)
        print('Slice:', S)
        print('Overlap:', O)
        print('Classifier free guidance:', cfg)
        print('DDIM sampling steps :', steps)
        print("skip", skip)

        dt_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.abspath(os.path.join(final_output_dir, f'{dt_file_name}.mp4'))
        output_path_demo = os.path.abspath(os.path.join(final_output_dir, f'{dt_file_name}_demo.mp4'))

        if weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

        torch.load = safe.unsafe_torch_load
        self.vae = AutoencoderKL.from_pretrained(
            self.models_paths["pretrained_vae"],
        ).to("cuda", dtype=weight_dtype)

        self.reference_unet = UNet2DConditionModel.from_pretrained(
            self.models_paths["pretrained_base_model"],
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        inference_config_path = self.models_paths["inference_config"]
        infer_config = OmegaConf.load(inference_config_path)

        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            Path(self.models_paths["pretrained_base_model"]),
            Path(self.models_paths["motion_module"]),
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")

        self.pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype, device="cuda"
        )

        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(models_dir, "image_encoder")
        ).to(dtype=weight_dtype, device="cuda")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        generator = torch.manual_seed(seed)

        width, height = W, H

        # load pretrained weights
        self.denoising_unet.load_state_dict(
            torch.load(self.models_paths["denoising_unet"], map_location="cpu"),
            strict=False,
        )
        self.reference_unet.load_state_dict(
            torch.load(self.models_paths["reference_unet"], map_location="cpu"),
        )
        self.pose_guider.load_state_dict(
            torch.load(self.models_paths["pose_guider"], map_location="cpu"),
        )
        torch.load = safe.load
        self.pipe = Pose2VideoPipeline(
            vae=self.vae,
            image_encoder=self.image_enc,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=scheduler,
        )
        self.pipe = self.pipe.to("cuda", dtype=weight_dtype)

        print('image: ', ref_image_path, "pose_video: ", pose_video_path)

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

        video = self.pipe(
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

        result = self.scale_video(video[:, :, :L], original_width, original_height)
        save_videos_grid(
            result,
            output_path,
            n_rows=1,
            fps=src_fps if fps is None or fps < 0 else fps,
        )

        video = torch.cat([ref_image_tensor, pose_tensor[:, :, :L], video[:, :, :L]], dim=0)
        video = self.scale_video(video, original_width, original_height)
        save_videos_grid(
            video,
            output_path_demo,
            n_rows=3,
            fps=src_fps if fps is None or fps < 0 else fps,
        )
        self.release_vram()
        return output_path

    def release_vram(self):
        models = [
            'vae', 'reference_unet', 'denoising_unet',
            'pose_guider', 'image_enc', 'pipe'
        ]

        for model_name in models:
            model = getattr(self, model_name, None)
            if model is not None:
                del model
                setattr(self, model_name, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def scale_video(video, width, height):
        video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
        scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                         width)  # [batch, frames, channels, height, width]

        return scaled_video




