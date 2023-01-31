"""
baseline for 3rd Anti-UAV
https://anti-uav.github.io/
"""
from __future__ import absolute_import
import os
import glob
import time
import json
import cv2
import numpy as np

from siamfc import TrackerSiamFC


def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'
    # setup tracker
    net_path = 'model.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # setup experiments
    video_paths = glob.glob(os.path.join('dataset', 'track1_test', '*'))
    video_num = len(video_paths)
    output_dir = os.path.join('results', tracker.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        img_files = glob.glob(video_path + "/*jpg")
        res_file = os.path.join(video_path, '%s_label.json'%mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)

        init_rect = label_res['gt_rect'][0]
        out_res = []
        start_time = time.time()
        for frame_id in range(len(img_files)):
            frame = cv2.imread(img_files[frame_id])
            if frame_id == 0:
                tracker.init(frame, init_rect)  # initialization
                out = init_rect
                out_res.append(init_rect)
            else:
                out = tracker.update(frame)  # tracking
                out_res.append(out.tolist())
            if visulization:
                _gt = label_res['gt_rect'][frame_id]
                _exist = label_res['exist'][frame_id]
                if _exist:
                    cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                                  (0, 255, 0))
                cv2.putText(frame, 'exist' if _exist else 'not exist',
                            (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)

                cv2.rectangle(frame, (int(out[0]), int(out[1])), (int(out[0] + out[2]), int(out[1] + out[3])),
                              (0, 255, 255))
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)
        if visulization:
            cv2.destroyAllWindows()
        end_time = time.time()
        FPS = len(img_files) / (end_time - start_time)
        print(str(video_num)+' - '+str(video_id)+' : '+str(FPS))
        # save result
        output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
        with open(output_file, 'w') as f:
            json.dump({'res': out_res}, f)


if __name__ == '__main__':
    main(mode='IR', visulization=False)
