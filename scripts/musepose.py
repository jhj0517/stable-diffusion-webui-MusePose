import gradio as gr
from packaging import version
import os

from scripts.installation import *
install_musepose()
from scripts.musepose_modules.ui_utils import *
from scripts.musepose_modules.paths import *
from scripts.musepose_modules.pose_align import PoseAlignmentInference
from scripts.musepose_modules.musepose_inference import MusePoseInference
from modules import scripts, script_callbacks

musepose_infer = MusePoseInference()
pose_alignment_infer = PoseAlignmentInference()


def add_tab():
    global musepose_infer
    global pose_alignment_infer

    with gr.Blocks() as tab:
        with gr.Tabs():
            with gr.TabItem('Step1: Pose Alignment'):
                with gr.Row():
                    with gr.Column(scale=3):
                        img_pose_input = gr.Image(label="Input Image", type="filepath", scale=5)
                        vid_dance_input = gr.Video(label="Input Dance Video", scale=5)
                    with gr.Column(scale=3):
                        vid_dance_output = gr.Video(label="Aligned Pose Output")
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
            btn_algin_pose.click(fn=pose_alignment_infer.align_pose,
                                 inputs=[vid_dance_input, img_pose_input, nb_detect_resolution, nb_image_resolution, nb_align_frame, nb_max_frame],
                                 outputs=[vid_dance_output])

            with gr.TabItem('Step2: MusePose Inference'):
                with gr.Row():
                    with gr.Column(scale=3):
                        img_musepose_input = gr.Image(label="Input Image", type="filepath", scale=5)
                        vid_pose_input = gr.Video(label="Input Aligned Pose", scale=5)
                    with gr.Column(scale=3):
                        vid_output = gr.Video(label="MusePose Output", scale=8)

                    with gr.Column(scale=3):
                        with gr.Column():
                            weight_dtype = gr.Dropdown(label="Compute Type", choices=["fp16", "fp32"],
                                                       value="fp16")
                            nb_width = gr.Number(label="Width.", value=512, precision=0)
                            nb_height = gr.Number(label="Height.", value=512, precision=0)
                            nb_video_frame_length = gr.Number(label="Video Frame Length", value=300, precision=0)
                            nb_video_slice_frame_length = gr.Number(label="Video Slice Frame Number", value=48, precision=0)
                            nb_video_slice_overlap_frame_number = gr.Number(label="Video Slice Overlap Frame Number", value=4, precision=0)
                            nb_cfg = gr.Number(label="CFG (Classifier Free Guidance)", value=3.5, precision=0)
                            nb_seed = gr.Number(label="Seed", value=99, precision=0)
                            nb_steps = gr.Number(label="DDIM Sampling Steps", value=20, precision=0)
                            nb_fps = gr.Number(label="FPS (Frames Per Second) ", value=-1, precision=0, info="Set to '-1' to use same FPS with pose's")
                            nb_skip = gr.Number(label="SKIP (Frame Sample Rate = SKIP+1)", value=1, precision=0)

                        with gr.Row():
                            btn_generate = gr.Button("GENERATE", variant="primary", scale=7)
                            btn_open_final_output_folder = gr.Button("📁", scale=3)

            btn_open_final_output_folder.click(fn=lambda: open_folder(final_output_dir), inputs=None, outputs=None)
            btn_generate.click(fn=musepose_infer.infer_musepose,
                               inputs=[img_musepose_input, vid_pose_input, weight_dtype, nb_width, nb_height, nb_video_frame_length,
                                       nb_video_slice_frame_length, nb_video_slice_overlap_frame_number, nb_cfg, nb_seed,
                                       nb_steps, nb_fps, nb_skip],
                               outputs=[vid_output])

            if version.parse(gr.__version__) >= version.parse("4.36.1"):
                vid_dance_output.change(fn=on_step1_complete,
                                        inputs=[img_pose_input, vid_dance_output],
                                        outputs=[img_musepose_input, vid_pose_input])


        return [(tab, "MusePose", "musepose")]


def on_unload():
    global musepose_infer
    global pose_alignment_infer
    musepose_infer = None
    pose_alignment_infer = None

script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_script_unloaded(on_unload)
