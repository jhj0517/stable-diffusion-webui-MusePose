# MusePose SD WebUI Extension
[MusePose](https://github.com/TMElyralab/MusePose) extension for the [SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## OverView
When running, this will download a **total of 9 models** to your `path_to_sd_webui\models\` directory if there are no models.
```
./models/
|-- MusePose
|   |-- denoising_unet.pth
|   |-- motion_module.pth
|   |-- pose_guider.pth
|   └── reference_unet.pth
|-- dwpose
|   |-- dw-ll_ucoco_384.pth
|   └── yolox_l_8x8_300e_coco.pth
|-- sd-image-variations-diffusers
|   └── unet
|       |-- config.json
|       └── diffusion_pytorch_model.bin
|-- image_encoder
|   |-- config.json
|   └── pytorch_model.bin
└── sd-vae-ft-mse
    |-- config.json
    └── diffusion_pytorch_model.bin
```
MusePose works through a two step process.
 
Step1 - Extract pose (skeleton) video from input dance video & input image.<br>
These models will be used for this step:
- `yolox_l_8x8_300e_coco.pth`
- `dw-ll_ucoco_384.pth`
  
The extracted pose video output will be saved in `path_to_sd_webui\outputs\MusePose\aligned_pose`.

Step2 - Make the image move from the input image & the extracted pose video.<br>
These models will be used for this step:
- `denoising_unet.pth`
- `motion_module.pth`
- `pose_guider.pth`
- `reference_unet.pth`
- `sd-image-variations-diffusers`
- `image_encoder`
- `sd-vae-ft-mse`

The output will be saved in 
`path_to_sd_webui\outputs\MusePose\inference_musepose`

# How to Install & Use
- Download & unzip this [repository](https://github.com/jhj0517/stable-diffusion-webui-MusePose/zipball/master) to `path_to_sd_webui\extensions\`
- Input image & input dancing video and click "ALIGN POSE" button in **Step1: Pose Alignment** tab.
![step1](https://github.com/jhj0517/stable-diffusion-webui-MusePose/assets/97279763/54a787ee-5bbc-4889-a9a9-453195fdab0b)
Once the process is done, the aligned pose video will be saved in `path_to_sd_webui\outputs\MusePose\pose_alignment`.
- Input image & input the extract pose video and click "GENERATE" button in **Step2: MusePose Inference** tab.
![step2](https://github.com/jhj0517/stable-diffusion-webui-MusePose/assets/97279763/30058906-06e2-4700-b622-bc023cb40d53)
Once the process is done, the output will be displayed in the right cell and saved in `path_to_sd_webui\outputs\MusePose\musepose_inference`. 
