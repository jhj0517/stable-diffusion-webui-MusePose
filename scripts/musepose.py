import gradio as gr
import os

from scripts.installation import *
install_musepose()
from scripts.musepose_modules.ui_utils import *
from scripts.musepose_modules.paths import *
from scripts.musepose_modules.pose_align import align_pose
from scripts.musepose_modules.infer_musepose import infer_musepose

from modules import scripts, script_callbacks


def add_tab():
    with gr.Blocks() as tab:
        with gr.Tabs():
            with gr.TabItem('Pose Alignment'):
                with gr.Row():
                    with gr.Column(scale=3):
                        img_input = gr.Image(label="Input Image here", type="filepath", scale=5)
                        vid_dance_input = gr.Video(label="Input Dance Video", scale=5)
                    with gr.Column(scale=3):
                        vid_dance_output = gr.Video(label="Aligned Pose Output will be displayed here")
                    with gr.Column(scale=3):
                        with gr.Column():
                            nb_detect_resolution = gr.Number(label="Detect Resolution", value=512, precision=0)
                            nb_image_resolution = gr.Number(label="Image Resolution.", value=720, precision=0)
                            nb_align_frame = gr.Number(label="Align Frame", value=0, precision=0)
                            nb_max_frame = gr.Number(label="Max Frame", value=300, precision=0)

                        with gr.Row():
                            btn_algin_pose = gr.Button("ALIGN POSE",  variant="primary", scale=7)
                            btn_open_pose_output_folder = gr.Button("📁", scale=3)

            btn_open_pose_output_folder.click(fn=lambda: open_folder(pose_output_dir), inputs=None, outputs=None)
            btn_algin_pose.click(fn=align_pose,
                                 inputs=[vid_dance_input, img_input, nb_detect_resolution, nb_image_resolution, nb_align_frame, nb_max_frame],
                                 outputs=[vid_dance_output])

            with gr.TabItem('Inferring MusePose'):
                with gr.Row():
                    with gr.Column(scale=3):
                        img_input = gr.Image(label="Input Image here", scale=5)
                        vid_pose_input = gr.Video(label="Input Aligned Pose Video here", scale=5)
                    with gr.Column(scale=3):
                        vid_output = gr.Video(label="Output Video will be displayed here", scale=8)

                    with gr.Column(scale=3):
                        with gr.Column():
                            weight_dtype = gr.DropDown(label="Compute Type", choices=["float16", "float32"],
                                                       value="float16")
                            nb_width = gr.Number(label="Width", value=768)
                            nb_height = gr.Number(label="Height", value=768)
                            nb_video_frame_length = gr.Number(label="Video Frame Length", value=300)
                            nb_video_slice_frame_length = gr.Number(label="Video Slice Frame Number", value=48)
                            nb_video_slice_overlap_frame_number = gr.Number(label="Video Slice Overlap Frame Number", value=4)
                            nb_cfg = gr.Number(label="Classifier Free Guidance", value=3.5)
                            nb_seed = gr.Number(label="Random Seed", value=99)
                            nb_steps = gr.Number(label="DDIM Sampling Steps", value=20)
                            nb_fps = gr.Number(label="FPS (Frames Per Second)", value=30)  # Assuming a default value for fps
                            nb_skip = gr.Number(label="SKIP (Frame Sample Rate = SKIP+1)", value=1, info="frame sample rate = (skip+1)")

                        with gr.Row():
                            btn_generate = gr.Button("GENERATE", variant="primary", scale=7)
                            btn_open_final_output_folder = gr.Button("📁", scale=3)

            btn_open_final_output_folder.click(fn=lambda: open_folder(final_output_dir), inputs=None, outputs=None)
            btn_generate.click(fn=infer_musepose,
                               inputs=[img_input, vid_pose_input, weight_dtype, nb_width, nb_height, nb_video_frame_length,
                                       nb_video_slice_frame_length, nb_video_slice_overlap_frame_number, nb_cfg, nb_seed,
                                       nb_steps, nb_fps, nb_skip],
                               outputs=[vid_output])

        return [(tab, "MusePose", "musepose")]


def on_unload():
    pass

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_script_unloaded(on_unload)
