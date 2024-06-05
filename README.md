# MusePose SD WebUI Extension
[MusePose](https://github.com/TMElyralab/MusePose) extension for the [SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
## OverView
When running, this will download **total 9 models** on your `path_to_sd_webui\models\` directory if there're no models.
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
MusePose works roughly through two process.
 
Step1 - Extract proper pose (skeleton) from input image & input dance video. 

For pose extraction, it will use:
- `yolox_l_8x8_300e_coco.pth`
- `dw-ll_ucoco_384.pth`

Step2 - Infer MusePose from extracted pose video & input image
For MusePose inference, it will use:
- rest of theme
