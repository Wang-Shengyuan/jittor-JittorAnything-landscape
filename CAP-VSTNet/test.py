import subprocess

subprocess.call(f'python image_transfer.py --mode photorealistic --content_dir data/synthesized_image --style_dir data/imgs --auto_seg', shell=True)