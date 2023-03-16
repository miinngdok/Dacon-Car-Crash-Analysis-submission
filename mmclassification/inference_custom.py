import os
import csv
import cv2
import argparse
from tqdm import tqdm

from mmcls.apis import init_model, inference_model

def inference_custom(args):
    work_dir = f'work_dirs/{args.config}'
    test_video_dir = '../data/test'
    test_csv = '../data/test.csv'

    test_csv_info=[]
    with open(test_csv, "r", encoding="utf8") as file:
        tmp = csv.reader(file)
        next(tmp)
        for row in tmp:
            test_csv_info.append(row)

    config_file = f'{work_dir}/{args.config}.py'
    epoch = args.epoch
    chekpoint_path = f'{work_dir}/epoch_{epoch}.pth'
    model = init_model(config=config_file, checkpoint=chekpoint_path)

    result_csv = []
    for i in tqdm(range(len(test_csv_info))):
        video_path = os.path.join(test_video_dir,test_csv_info[i][0]+'.mp4')
        video = cv2.VideoCapture(video_path)
        tmp_result = []
        for _ in range(11): # 앞쪽 이미지 일부 뽑아 hard voting
            tmp_result.append(inference_model(model, video.read()[1])['pred_label'])
        result_csv.append(f'{test_csv_info[i][0]},{max(set(tmp_result),key=tmp_result.count)}')
        
    with open(f'{work_dir}/{args.config}_epoch{epoch}.csv', 'w') as f:
        for row in result_csv:
            f.write(row+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()

    inference_custom(args)
