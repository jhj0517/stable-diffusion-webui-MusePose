# MusePose SD WebUI Extension
[MusePose](https://github.com/TMElyralab/MusePose) extension for the [SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
![screenshot](https://github.com/jhj0517/stable-diffusion-webui-MusePose/assets/97279763/aa982503-50c2-4093-9319-38510d51160b)


## OverView
When running, this will download a **total of 9 models** (15GB total) to your `path_to_sd_webui\models\` directory if there are no models.
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
You can also manually download the models in the links [here](https://github.com/TMElyralab/MusePose?tab=readme-ov-file#download-weights), if you want.

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
- Input image & input dancing video and click "ALIGN POSE" button in **Pose Alignment** tab.<br>
The output will be saved in `path_to_sd_webui\outputs\MusePose\pose_alignment`.
- Input image & input the extract pose video from step1 and click "GENERATE" button in **MusePose Inference** tab.<br>
The output will be saved in `path_to_sd_webui\outputs\MusePose\musepose_inference`. 

# Troubleshooting For Installation
If you encounter error during installation and the MusePose tab doesn't appear, it's because WebUI's venv prevents installing some dependencies.<br>
To fix this, you need to manually activate the venv and install these packages. 
1. Open the terminal in the WebUI and activate the venv
```
C:\YourPath\To_SD_WebUI>venv\Scripts\activate
```
Then it will display (venv) in front of the terminal like this.
```
(venv) C:\YourPath\To_SD_WebUI>
```
2. In this state, run
```
pip uninstall opencv-python-headless
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip install opencv-python
pip install opencv-contrib-python
```

