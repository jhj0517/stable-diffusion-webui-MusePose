import os
import gradio as gr

def open_folder(folder_path):
    if os.path.exists(folder_path):
        os.system(f'start "" "{folder_path}"')
    else:
        print(f"The folder '{folder_path}' does not exist.")

def on_step1_complete(input_img: str, input_pose_vid: str):
    return [gr.Image(label="Input Image", value=input_img, type="filepath", scale=5),
            gr.Video(label="Input Aligned Pose Video", value=input_pose_vid, scale=5)]