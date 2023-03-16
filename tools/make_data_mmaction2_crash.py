import os
import csv
import random


def make_dataset():
    csv_path = '../data/train.csv'
    output_txt_dir = '../data/mmaction_crash'
    os.makedirs(output_txt_dir, exist_ok=True)
    data_path = '../data/train'

    relabeling_path = './relabeling_crash.txt'
    relabel={}

    with open(relabeling_path, 'r') as f:
        for row in f:
            tmp = row.split()
            relabel[tmp[0]]=tmp[1]
            
    no_crash=[]
    crash=[]
    with open(csv_path, 'r') as f:
        r = csv.reader(f, delimiter=',')
        next(r)
        for row in r:
            file_num = row[0][-4:]
            if file_num in relabel:
                label_fix = relabel[file_num]
                if label_fix == '0':
                    no_crash.append(f'{data_path}/{row[0]}.mp4 0')
                elif label_fix == '1':
                    crash.append(f'{data_path}/{row[0]}.mp4 1')
            else:
                if not int(row[2]) :
                    no_crash.append(f'{data_path}/{row[0]}.mp4 0')
                else:
                    crash.append(f'{data_path}/{row[0]}.mp4 1')
            
    
    train=[]
    train+=no_crash[:]
    for _ in range(2): # oversampling
        train+=crash[:]

    random.seed(8)
    random.shuffle(train)

    print(f'no_crash : {len(no_crash)}, crash : {len(crash)*2}')

    with open(output_txt_dir+'/train_crash_whole.txt', 'w') as f:
        for row in train:
            f.write(row+'\n')


if __name__ == '__main__':
    make_dataset()
