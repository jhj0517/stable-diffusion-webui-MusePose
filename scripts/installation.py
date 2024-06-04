import os
import subprocess
import sys

extension_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sd_webui_dir = os.path.abspath(os.path.join(extension_dir, '..', '..'))


def install_package(install_command):
    print(f"MusePose Extension: Installing {install_command}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + install_command.split())


def install_musepose():
    req_file_path = os.path.join(extension_dir, 'musepose_requirements.txt')

    from launch import is_installed, run_pip
    with open(req_file_path) as file:
        for package in file:
            package_name, version = package.strip().split("==")
            if not is_installed(package_name):
                run_pip(f"install {package_name}=={version}", f"MusePose Extension: Installing {package_name}=={version}")

    problematic_packages = [
        "wheel==0.43.0",
        "mmcv==2.1.0 --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html",
        "mmengine==0.10.4",
        "mmdet==3.3.0",
        "mmpose==1.3.1"
    ]

    for command in problematic_packages:
        package_name, version = command.strip().split("==")
        if not is_installed(package_name):
            install_package(command)