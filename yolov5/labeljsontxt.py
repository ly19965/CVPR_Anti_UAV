import json
import os
import numpy

path = '/vol/research/facer2vm_tracking/people/xuefeng/2023AntiUAV/test.json'
root = '/vol/research/facer2vm_tracking/people/xuefeng/2023AntiUAV/test'

file = open(path, 'r')
data = json.load(file)

for item in data['annotations']:
    image_id = item['image_id']
    if image_id % 1000 == 0:
        print(image_id)
    file_name = data['images'][image_id-1]['file_name']
    h = data['images'][image_id-1]['height']
    w = data['images'][image_id - 1]['width']
    rect = item['bbox']
    rect[0] = rect[0] / w
    rect[2] = rect[2] / w
    rect[1] = rect[1] / h
    rect[3] = rect[3] / h
    rect[0] = rect[0] + 0.5 * rect[2]
    rect[1] = rect[1] + 0.5 * rect[3]
    file_name_split = os.path.split(file_name)
    folder_name = os.path.join(root, file_name_split[0])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(os.path.join(root, file_name)[:-4]+'.txt', 'w') as f:
        f.write('0 %s %s %s %s\n' % (rect[0], rect[1], rect[2], rect[3]))
