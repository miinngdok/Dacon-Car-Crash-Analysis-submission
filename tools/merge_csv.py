import os
import csv
from collections import Counter


def merge_csv():
    output_dir = '../csv_merged'
    os.makedirs(output_dir, exist_ok=True)

    result_csv = ['sample_id,label']

    csv_crash=[]
    csv_ego=[]
    csv_weather=[]
    csv_timing=[]


    with open('../Video-Swin-Transformer/work_dirs/swin_base_crash_final/swin_base_crash_final_epoch3.csv', "r", encoding="utf8") as file:
        tmp = csv.reader(file)
        for row in tmp:
            csv_crash.append(row)

    with open('../Video-Swin-Transformer/work_dirs/swin_base_ego_final/swin_base_ego_final_epoch20.csv', "r", encoding="utf8") as file:
        tmp = csv.reader(file)
        for row in tmp:
            csv_ego.append(row)

    with open('../mmclassification/work_dirs/convnext_large_weather_final/convnext_large_weather_final_epoch4.csv', "r", encoding="utf8") as file:
        tmp = csv.reader(file)
        for row in tmp:
            csv_weather.append(row)

    timing_path=[
        '../mmclassification/work_dirs/efficientnet_b0_timing_final_1/efficientnet_b0_timing_final_1_epoch4.csv',
        '../mmclassification/work_dirs/efficientnet_b0_timing_final_1/efficientnet_b0_timing_final_1_epoch1.csv',
        '../mmclassification/work_dirs/efficientnet_b0_timing_final_0/efficientnet_b0_timing_final_0_epoch1.csv',
    ]

    csvs_timing=[[] for _ in range(1800)]
    for i in range(len(timing_path)):
        with open(timing_path[i], "r", encoding="utf8") as file:
            tmp = csv.reader(file)
            idx=0
            for row in tmp:
                csvs_timing[idx].append(row[1])
                idx+=1

    csv_timing=[]
    for i in range(1800):
        count = Counter(csvs_timing[i]).most_common()
        if len(count)==1:
            csv_timing.append([f'TEST_{str(i).zfill(4)}', count[0][0]])
        else:
            if count[0][1]!=count[1][1]:
                csv_timing.append([f'TEST_{str(i).zfill(4)}', count[0][0]])
            else:
                csv_timing.append([f'TEST_{str(i).zfill(4)}', csv_timing[0][0]])

    for i in range(len(csv_crash)):
        video_name, label_crash = csv_crash[i]
        label_crash = int(label_crash)
        if not label_crash:
            result_csv.append(f'{video_name},{0}')
        else:
            label_ego = int(csv_ego[i][1])
            label_weather = int(csv_weather[i][1])
            label_timing = int(csv_timing[i][1])
            label_merged = 1 + label_ego*6 + label_weather*2 + label_timing  
            result_csv.append(f'{video_name},{label_merged}')

    file_name='submit_final'
    with open(os.path.join(output_dir, f'{file_name}.csv'), 'w') as f:
        for row in result_csv:
            f.write(row+'\n')


if __name__ == '__main__':
    merge_csv()