import os
import csv
import argparse
import cv2
from tqdm import tqdm

from mmaction.apis import init_recognizer, inference_recognizer
    
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

    config_file = f'./work_dirs/{args.config}/{args.config}.py'
    epoch = args.epoch
    chekpoint_path = f'./work_dirs/{args.config}/epoch_{epoch}.pth'
    label_path = '../tools/label.txt'
    model = init_recognizer(config=config_file, checkpoint=chekpoint_path)
    result_csv = []
    
    for i in tqdm(range(len(test_csv_info))):
        video_path = os.path.join(test_video_dir,test_csv_info[i][0]+'.mp4')

        output=inference_recognizer(model=model, video_path=video_path, label_path=label_path)[0][0]
        result_csv.append(f'{test_csv_info[i][0]},{output}')

    with open(os.path.join(work_dir, f'{args.config}_epoch{epoch}.csv'), 'w') as f:
        for row in result_csv:
            f.write(row+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()

    inference_custom(args)
