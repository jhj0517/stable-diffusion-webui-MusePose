import os

extension_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sd_webui_dir = os.path.abspath(os.path.join(extension_dir, '..', '..'))
pose_output_dir = os.path.abspath(os.path.join(sd_webui_dir, 'outputs', 'MusePose', "pose"))
final_output_dir = os.path.abspath(os.path.join(sd_webui_dir, 'outputs', 'MusePose', "final"))
