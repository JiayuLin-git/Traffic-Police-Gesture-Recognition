import os
import csv
import re

def read_pic(folder_path, file):
    txt_file = folder_path + '/' +file
    picture_data = []
    with open(txt_file, 'r') as f:
        for row in f:
            l = re.findall(r"\d+\.?\d*", row)
            if len(l) == 6:
                a = float(l[0]) * pow(10, float(l[1]))
                b = float(l[2]) * pow(10, float(l[3]))
                c = float(l[4]) * pow(10, float(l[5]))
            if len(l) == 3:
                a = float(l[0])
                b = float(l[1])
                c = float(l[2])
            picture_data.append(a)
            picture_data.append(b)
            picture_data.append(c)
    label = file[:-17]
    picture_data.append(label)
    index = file[:-14]
    picture_data.append(index)
    return picture_data

data = []
project_path = '/Users/zephyryau/Documents/study/INF552/Project/'

data_path = project_path + 'traffic_police_gesture_processed/'
folders = os.listdir(data_path)
#print(folders)

for gesture in folders:
    if gesture != '.DS_Store':
        for _ in os.listdir(data_path+gesture):
            if _[-9:] == 'BodyPoint':
                folder_path = data_path + gesture + '/' + _
                files = os.listdir(folder_path)
                for file in files:
                    if file != '.DS_Store':
                        picture_data = read_pic(folder_path, file)
                        data.append(picture_data)


with open("data.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    for row in data:
        writer.writerow(row)
