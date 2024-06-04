from scripts.installation import *
#install_musepose()
from scripts.musepose_modules.ui_utils import *

import gradio as gr
import os
from modules import scripts, script_callbacks

musepose_inf = 3

def add_tab():
    with gr.Blocks() as tab:
        with gr.Tabs():
            with gr.TabItem('Align Pose'):
                with gr.Row():
                    with gr.Column(scale=5):
                        img_input = gr.Image(label="Input Image here", scale=5)
                        vid_dance_input = gr.Video(label="Input Dance Video", scale=5)
                    with gr.Column(scale=5):
                        vid_dance_output = gr.Video(label="Aligned Pose Output will be displayed here", scale=8)
                        with gr.Row(scale=2):
                            btn_algin_pose = gr.Button("ALIGN POSE",  variant="primary", scale=7)
                            btn_open_pose_output_folder = gr.Button("📁", scale=3)

            with gr.TabItem('Inferring MusePose'):
                with gr.Row():
                    with gr.Column(scale=5):
                        img_input = gr.Image(label="Input Image here", scale=5)
                        vid_pose_input = gr.Video(label="Input Aligned Pose Video here", scale=5)
                    with gr.Column(scale=5):
                        vid_output = gr.Video(label="Output Video will be displayed here", scale=8)
                        with gr.Row(scale=2):
                            btn_algin_pose = gr.Button("GENERATE", variant="primary", scale=7)
                            btn_open_pose_output_folder = gr.Button("📁", scale=3)

        return [(tab, "MusePose", "musepose")]


def on_unload():
    global musepose_inf
    musepose_inf = None

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_script_unloaded(on_unload)
