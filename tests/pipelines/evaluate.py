#!/usr/bin/env python
import sys
import os
import os.path
import glob
import json
import numpy as np
import io

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def eval(out_res, label_res):
    measure_per_frame = []
    penalty_measure = []  # penalty for frames where the target exists but is not detected
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt) if len(_pred) > 1 else 0)
        if _exist:
            if (len(_pred) > 1 and iou(_pred, _gt) > 1e-5):
                penalty_measure.append(0)
            else:
                penalty_measure.append(1)

    return np.mean(measure_per_frame) - max(0, 0.2 * np.mean(penalty_measure)**0.3)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

#input_dir = "C:/Users/24790/Desktop/track1"
#output_dir = "C:/Users/24790/Desktop/track1"

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref', 'track1_test_labels')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    mode='IR'
    
    label_files = sorted(glob.glob(
        os.path.join(truth_dir, '*/IR_label.json')))

    video_num = len(label_files)
    overall_performance = []

    for video_id, label_file in enumerate(label_files, start=1):

        with open(label_file, 'r') as f:
            label_res = json.load(f)

            video_dirs=os.path.dirname(label_file)

            video_dirsbase = os.path.basename(video_dirs)

            pred_file = os.path.join(submit_dir, video_dirsbase+'_%s.txt' % mode)

            try:
                with open(pred_file, 'r') as f:
                    pred_res = json.load(f)
                    pred_res=pred_res['res']
            except:
                with open(pred_file, 'r') as f:
                    pred_res = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
                    pred_res[:, 2:] = pred_res[:, 2:] - pred_res[:, :2] + 1
                
            mixed_measure = eval(pred_res, label_res)
            overall_performance.append(mixed_measure)

    #output_file.write(str(np.mean(overall_performance)))
    #print(np.mean(overall_performance))
    #output_file.write("correct:1")
    output_file.write('correct:{:}'.format(np.mean(overall_performance)))

    output_file.close()
