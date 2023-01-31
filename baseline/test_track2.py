"""
baseline for 3rd Anti-UAV
https://anti-uav.github.io/
"""
from __future__ import absolute_import
import os
import glob
import json
import cv2
import time
import sys
sys.path.append('./yolov5/')
from detection_siamfc import TrackerSiamFC

def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'
    # setup tracker
    net_path = 'model.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    yolo_model = tracker.initialize_yolo()

    # setup experiments
    video_paths = glob.glob(os.path.join('dataset', 'track2_test', '*'))
    video_num = len(video_paths)
    output_dir = os.path.join('results', tracker.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        img_files = glob.glob(video_path + "/*jpg")

        out_res = []
        pred_bbox = [0] # no prection
        start_time = time.time()
        for frame_id in range(len(img_files)):
            frame = cv2.imread(img_files[frame_id])
            if len(pred_bbox) == 1:
                pred_bbox, im_vis = tracker.init(frame, yolo_model)  # initialization
                out_res.append(pred_bbox)

                cv2.putText(im_vis, str(frame_id), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if len(pred_bbox) == 1:
                    cv2.putText(im_vis, 'Fail to detect the UAV', (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                else:
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(im_vis, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 0, 255), 3)
            else:
                pred_bbox = tracker.update(frame)  # tracking
                pred_bbox = list(map(int, pred_bbox))
                out_res.append(pred_bbox)
                cv2.rectangle(im_vis, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 0, 255), 3)
                cv2.putText(im_vis, str(frame_id), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if visulization:
                cv2.imshow(video_name, im_vis)
                cv2.waitKey(1)

            frame_id += 1
        if visulization:
            cv2.destroyAllWindows()
        end_time = time.time()
        FPS = len(img_files) / (end_time - start_time)
        print(str(video_num) + ' - ' + str(video_id) + ' : ' + str(FPS))
        # save result
        output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
        with open(output_file, 'w') as f:
            json.dump({'res': out_res}, f)


if __name__ == '__main__':
    main(mode='IR', visulization=False)
