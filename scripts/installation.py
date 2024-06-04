import os

extension_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sd_webui_dir = os.path.abspath(os.path.join(extension_dir, '..', '..'))


def install_musepose():
    req_file_path = os.path.join(extension_dir, 'musepose_requirements.txt')

    from launch import is_installed, run_pip
    with open(req_file_path) as file:
        for package in file:
            package = package.strip()
            if not is_installed(package):
                run_pip(f"install {package}", f"MusePose Extension: Installing {package}")