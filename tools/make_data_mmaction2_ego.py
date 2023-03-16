import os
import csv
import random

def make_dataset():
    csv_path = '../data/train.csv'
    output_txt_dir = '../data/mmaction_ego'
    os.makedirs(output_txt_dir, exist_ok=True)
    data_path = '../data/train'

    relabeling_path = './relabeling_ego.txt'
    relabel={}

    with open(relabeling_path, 'r') as f:
        for row in f:
            tmp = row.split()
            relabel[tmp[0]]=tmp[1]
            
    ego=[]
    non_ego=[]
    with open(csv_path, 'r') as f:
        r = csv.reader(f, delimiter=',')
        next(r)
        for row in r:
            if int(row[2]):
                file_num = row[0][-4:]
                if file_num in relabel:
                    label_fix = relabel[file_num]
                    if label_fix == '0':
                        ego.append(f'{data_path}/{row[0]}.mp4 0')
                    elif label_fix == '1':
                        non_ego.append(f'{data_path}/{row[0]}.mp4 1')
                else:
                    if int(row[2])<=6 :
                        ego.append(f'{data_path}/{row[0]}.mp4 0')
                    else:
                        non_ego.append(f'{data_path}/{row[0]}.mp4 1')


    train=[]
    train+=ego[:]
    train+=non_ego[:]

    random.seed(8)
    random.shuffle(train)

    print(f'ego : {len(ego)}, non_ego : {len(non_ego)}')

    with open(output_txt_dir+'/train_ego_whole.txt', 'w') as f:
        for row in train:
            f.write(row+'\n')


if __name__ == '__main__':
    make_dataset()
