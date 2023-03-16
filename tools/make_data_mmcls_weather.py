import os
import csv
import random
import cv2
from tqdm import tqdm

def make_dataset():
    relabeling_path = './relabeling_weather.txt'
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
    output_dir = '../data/mmcls_weather_labelingnocrash'
    video_dir = '../data/train'
    os.makedirs(output_dir, exist_ok=True)

    output_meta_dir = os.path.join(output_dir, 'meta')
    os.makedirs(output_meta_dir, exist_ok=True)
    output_train_img_dir=os.path.join(output_dir, 'train')
    os.makedirs(output_train_img_dir, exist_ok=True)
    output_val_img_dir=os.path.join(output_dir, 'val')
    os.makedirs(output_val_img_dir, exist_ok=True)


    imgs_normal = []
    imgs_snowy = []
    imgs_rainy = []

    with open(csv_path, 'r') as f:
        r = csv.reader(f, delimiter=',')
        next(r)
        for row in tqdm(r):
            video_name = row[0]
            video_path = f'{video_dir}/{video_name}.mp4'
            label = int(row[2])
            if label:
                file_num = row[0][-4:]
                weather = (label-1)%6//2
                if file_num in relabel:
                    label_fix = relabel[file_num]
                    if label_fix == '-1':
                        continue
                    elif label_fix == '0':
                        weather = 0
                    elif label_fix == '1':
                        weather = 1
                    else:
                        weather = 2
                video = cv2.VideoCapture(video_path)
                tmp=[]
                if weather == 0:   # normal
                    for i in range(3):
                        tmp.append((video.read()[1], f'{video_name}_{i}', 0))
                        video.read()
                    imgs_normal.append(tmp)
                elif weather==1:   # snowy
                    for i in range(13):
                        tmp.append((video.read()[1], f'{video_name}_{i}', 1))
                        video.read()
                    imgs_snowy.append(tmp)
                else:            # rainy
                    for i in range(27):
                        tmp.append((video.read()[1], f'{video_name}_{i}', 2))
                    imgs_rainy.append(tmp)

    imgs_normal_labelingnocrash = []
    imgs_snowy_labelingnocrash = []
    imgs_rainy_labelingnocrash = []

    for file_num, label, _ in tqdm(labeling_no_crash):

        video_path = f'{video_dir}/{file_num}.mp4'
        weather = int(label)
        if weather>=0:
            video = cv2.VideoCapture(video_path)
            tmp=[]
            if weather == 0:   # normal
                for i in range(2):
                    tmp.append((video.read()[1], f'{file_num}_{i}', 0))
                    video.read()
                imgs_normal_labelingnocrash.append(tmp)
            elif weather==1:   # snowy
                for i in range(13):
                    tmp.append((video.read()[1], f'{file_num}_{i}', 1))
                    video.read()
                imgs_snowy_labelingnocrash.append(tmp)
            elif weather==2:   # rainy
                for i in range(10):
                    tmp.append((video.read()[1], f'{file_num}_{i}', 2))
                    video.read()
                imgs_rainy_labelingnocrash.append(tmp)

    random.seed(8)

    random.shuffle(imgs_normal)
    random.shuffle(imgs_snowy)
    random.shuffle(imgs_rainy)
    num_train_normal = int(len(imgs_normal)*0.8)
    num_train_snowy = int((len(imgs_snowy)*0.8))
    num_train_rainy = int((len(imgs_rainy)*0.8))

    train_txt=[]
    val_txt=[]

    count_normal_imgs=[0,0]
    count_snowy_imgs=[0,0]
    count_rainy_imgs=[0,0]

    for i in tqdm(range(num_train_normal)):
        for j in range(len(imgs_normal[i])):
            img, file_name, label = imgs_normal[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/train/{file_name}.jpg', img)
            count_normal_imgs[0]+=1
    for i in tqdm(range(num_train_normal, len(imgs_normal))):
        for j in range(len(imgs_normal[i])):
            img, file_name, label = imgs_normal[i][j]
            val_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/val/{file_name}.jpg', img)
            count_normal_imgs[1]+=1

    for i in tqdm(range(num_train_snowy)):
        for j in range(len(imgs_snowy[i])):
            img, file_name, label = imgs_snowy[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/train/{file_name}.jpg', img)
            count_snowy_imgs[0]+=1
    for i in tqdm(range(num_train_snowy, len(imgs_snowy))):
        for j in range(len(imgs_snowy[i])):
            img, file_name, label = imgs_snowy[i][j]
            val_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/val/{file_name}.jpg', img)
            count_snowy_imgs[1]+=1

    for i in tqdm(range(num_train_rainy)):
        for j in range(len(imgs_rainy[i])):
            img, file_name, label = imgs_rainy[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/train/{file_name}.jpg', img)
            count_rainy_imgs[0]+=1
    for i in tqdm(range(num_train_rainy, len(imgs_rainy))):
        for j in range(len(imgs_rainy[i])):
            img, file_name, label = imgs_rainy[i][j]
            val_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/val/{file_name}.jpg', img)
            count_rainy_imgs[1]+=1

    random.shuffle(imgs_normal_labelingnocrash)
    random.shuffle(imgs_snowy_labelingnocrash)
    random.shuffle(imgs_rainy_labelingnocrash)
    for i in tqdm(range(len(imgs_normal_labelingnocrash))):
        for j in range(len(imgs_normal_labelingnocrash[i])):
            img, file_name, label = imgs_normal_labelingnocrash[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/train/{file_name}.jpg', img)
            count_normal_imgs[0]+=1

    for i in tqdm(range(len(imgs_snowy_labelingnocrash))):
        for j in range(len(imgs_snowy_labelingnocrash[i])):
            img, file_name, label = imgs_snowy_labelingnocrash[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/train/{file_name}.jpg', img)
            count_snowy_imgs[0]+=1

    for i in tqdm(range(len(imgs_rainy_labelingnocrash))):
        for j in range(len(imgs_rainy_labelingnocrash[i])):
            img, file_name, label = imgs_rainy_labelingnocrash[i][j]
            train_txt.append(f'{file_name}.jpg {label}')
            cv2.imwrite(f'{output_dir}/train/{file_name}.jpg', img)
            count_rainy_imgs[0]+=1


    random.shuffle(train_txt)
    random.shuffle(val_txt)

    with open(os.path.join(output_dir, 'meta', 'train.txt'), 'w') as f:
        for row in train_txt:
            f.write(row+'\n')
    with open(os.path.join(output_dir, 'meta', 'val.txt'), 'w') as f:
        for row in val_txt:
            f.write(row+'\n')

    print(f'normal  -  train : {count_normal_imgs[0]} val : {count_normal_imgs[1]}')
    print(f'snowy  -  train : {count_snowy_imgs[0]} val : {count_snowy_imgs[1]}')
    print(f'rainy  -  train : {count_rainy_imgs[0]} val : {count_rainy_imgs[1]}')

if __name__ == '__main__':
    make_dataset()
