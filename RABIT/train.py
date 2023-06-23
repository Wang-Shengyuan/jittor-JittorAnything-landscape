import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)

args = parser.parse_args()

print('训练生成数据存储在./checkpoints/rabit-base中')
subprocess.call(f'python rabit_train.py --name rabit-base --dataset_mode custom --label_dir {args.input_path}/labels --image_dir {args.input_path}/imgs --label_nc 29 --batchSize 4', shell=True)