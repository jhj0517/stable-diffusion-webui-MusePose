import gradio as gr
import os

from scripts.installation import *
install_musepose()
from scripts.musepose_modules.ui_utils import *
from scripts.musepose_modules.paths import *
from scripts.musepose_modules.pose_align import run_align_video_with_filterPose_translate_smooth

from modules import scripts, script_callbacks


musepose_inf = None

def add_tab():
    with gr.Blocks() as tab:
        with gr.Tabs():
            with gr.TabItem('Align Pose'):
                with gr.Row():
                    with gr.Column(scale=3):
                        img_input = gr.Image(label="Input Image here", scale=5)
                        vid_dance_input = gr.Video(label="Input Dance Video", scale=5)
                    with gr.Column(scale=3):
                        vid_dance_output = gr.Video(label="Aligned Pose Output will be displayed here")
                    with gr.Column(scale=3):
                        with gr.Column():
                            nb_detect_resolution = gr.Number(label="Detect Resolution", value=512)
                            nb_image_resolution = gr.Number(label="Image Resolution", value=720)
                            nb_align_frame = gr.Number(label="Align Frame", value=0)
                            nb_max_frame = gr.Number(label="Max Frame", value=300)

                        with gr.Row():
                            btn_algin_pose = gr.Button("ALIGN POSE",  variant="primary", scale=7)
                            btn_open_pose_output_folder = gr.Button("📁", scale=3)

            btn_open_pose_output_folder.click(fn=lambda: open_folder(pose_output_dir), inputs=None, outputs=None)
            btn_algin_pose.click(fn=run_align_video_with_filterPose_translate_smooth,
                                 inputs=[vid_dance_input, img_input, nb_detect_resolution, nb_image_resolution, nb_align_frame, nb_max_frame],
                                 outputs=[vid_dance_output])

            with gr.TabItem('Inferring MusePose'):
                with gr.Row():
                    with gr.Column(scale=5):
                        img_input = gr.Image(label="Input Image here", scale=5)
                        vid_pose_input = gr.Video(label="Input Aligned Pose Video here", scale=5)
                    with gr.Column(scale=5):
                        vid_output = gr.Video(label="Output Video will be displayed here", scale=8)
                        with gr.Row(scale=2):
                            btn_align_pose = gr.Button("GENERATE", variant="primary", scale=7)
                            btn_open_final_output_folder = gr.Button("📁", scale=3)

            btn_open_final_output_folder.click(fn=lambda: open_folder(final_output_dir), inputs=None, outputs=None)

        return [(tab, "MusePose", "musepose")]


def on_unload():
    global musepose_inf
    musepose_inf = None

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_script_unloaded(on_unload)
