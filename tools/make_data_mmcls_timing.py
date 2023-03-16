import os
import csv
import random
import cv2
from tqdm import tqdm

def make_dataset():
    relabeling_path = './relabeling_timing.txt'
    relabel={}

    with open(relabeling_path, 'r') as f:
        for row in f:
            tmp = row.split()
            relabel[tmp[0]]=tmp[1]

    labeling_nocrash_path='./labeling_nocrash.txt'
    labeling_no_crash = []
    with open(labeling_nocrash_path, 'r') as f:
        for row in f:
            tmp = row.split()
            labeling_no_crash.append(tmp)


    csv_path = '../data/train.csv'
    output_dir = '../data/mmcls_timing_whole'
    video_dir = '../data/train'
    os.makedirs(output_dir, exist_ok=True)

    output_meta_dir = os.path.join(output_dir, 'meta')
    os.makedirs(output_meta_dir, exist_ok=True)
    output_train_img_dir=os.path.join(output_dir, 'train')
    os.makedirs(output_train_img_dir, exist_ok=True)

    imgs_day = []
    imgs_night = []

    with open(csv_path, 'r') as f:
        r = csv.reader(f, delimiter=',')
        next(r)
        for row in tqdm(r):
            video_name = row[0]
            video_path = f'{video_dir}/{video_name}.mp4'
            label = int(row[2])
            if label:
                file_num = row[0][-4:]
                timing = (label+1)%2
                if file_num in relabel:
                    label_fix = relabel[file_num]
                    if label_fix == '-1':
                        continue
                    elif label_fix == '0':
                        timing = 0
                    elif label_fix == '1':
                        timing = 1
                video = cv2.VideoCapture(video_path)
                tmp=[]
                if timing == 0:   # day
                    for i in range(3):
                        tmp.append((video.read()[1], f'{video_name}_{i}', 0))
                        video.read()
                    imgs_day.append(tmp)
                elif timing==1:   # night
                    for i in range(24):
                        tmp.append((video.read()[1], f'{video_name}_{i}', 1))
                    imgs_night.append(tmp)

    imgs_day_labelingnocrash = []
    imgs_night_labelingnocrash = []

    for file_num, _, label in tqdm(labeling_no_crash):

        video_path = f'{video_dir}/{file_num}.mp4'
        timing = int(label)
        if timing>=0:
            video = cv2.VideoCapture(video_path)
            tmp=[]
            if timing == 0:   # day
                for i in range(2):
                    tmp.append((video.read()[1], f'{file_num}_{i}', 0))
                    video.read()
                imgs_day_labelingnocrash.append(tmp)
            elif timing==1:   # night
                for i in range(2):
                    tmp.append((video.read()[1], f'{file_num}_{i}', 1))
                    video.read()
                imgs_night_labelingnocrash.append(tmp)

    random.seed(8)

    train_txt=[]
    count_day_imgs=0
    count_night_imgs=0

    for i in tqdm(range(len(imgs_day))):
        for j in range(len(imgs_day[i])):
            img, file_name, label = imgs_day[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}//train/{file_name}.jpg', img)
            count_day_imgs+=1

    for i in tqdm(range(len(imgs_night))):
        for j in range(len(imgs_night[i])):
            img, file_name, label = imgs_night[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/train/{file_name}.jpg', img)
            count_night_imgs+=1


    for i in tqdm(range(len(imgs_day_labelingnocrash))):
        for j in range(len(imgs_day_labelingnocrash[i])):
            img, file_name, label = imgs_day_labelingnocrash[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}//train/{file_name}.jpg', img)
            count_day_imgs+=1
    for i in tqdm(range(len(imgs_night_labelingnocrash))):
        for j in range(len(imgs_night_labelingnocrash[i])):
            img, file_name, label = imgs_night_labelingnocrash[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}//train/{file_name}.jpg', img)
            count_night_imgs+=1

    random.shuffle(train_txt)
    with open(os.path.join(output_dir, 'meta', 'train.txt'), 'w') as f:
        for row in train_txt:
            f.write(row+'\n')

    print(f'day : {count_day_imgs}')
    print(f'night : {count_night_imgs}')

if __name__ == '__main__':
    make_dataset()
