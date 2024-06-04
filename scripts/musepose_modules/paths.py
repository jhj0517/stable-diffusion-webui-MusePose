import os

extension_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sd_webui_dir = os.path.abspath(os.path.join(extension_dir, '..', '..'))
pose_output_dir = os.path.abspath(os.path.join(sd_webui_dir, 'outputs', 'MusePose', "pose"))
if not os.path.exists(pose_output_dir):
    os.makedirs(pose_output_dir)
final_output_dir = os.path.abspath(os.path.join(sd_webui_dir, 'outputs', 'MusePose', "final"))
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)
