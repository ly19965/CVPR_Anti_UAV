# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2
import json
import os
import numpy as np

from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level
from modelscope.utils.constant import DownloadMode
import glob
import time

class AntiUavTrack2Test(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.video_multi_object_tracking
        self.model_id = 'damo/3rd_Anti-UAV_CVPR23'
        track1_test_set = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='validation')
        assert track1_test_set is not None, 'test set should be downloaded first'
        # set own path
        self.dataset_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f/validation'
        self.visulization = False


    def iou(self, bbox1, bbox2):
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

    def not_exist(self, pred):
        return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


    def eval(self, out_res, label_res):
        measure_per_frame = []
        penalty_measure = []  # penalty for frames where the target exists but is not detected
        for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
            measure_per_frame.append(self.not_exist(_pred) if not _exist else self.iou(_pred, _gt) if len(_pred) > 1 else 0)
            if _exist:
                if (len(_pred) > 1 and self.iou(_pred, _gt) > 1e-5):
                    penalty_measure.append(0)
                else:
                    penalty_measure.append(1)

        return np.mean(measure_per_frame) - max(0, 0.2 * np.mean(penalty_measure)**0.3)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        video_paths = glob.glob(os.path.join(self.dataset_dir, '*'))
        video_num = len(video_paths)
        overall_performance = []
        video_id = 1
        mode = 'IR'
        truth_dir = self.dataset_dir

        # run tracking experiments and report performance
        for video_id, video_path in enumerate(video_paths, start=1):
            uav_detection = pipeline(self.task, model=self.model_id)
            tracker = uav_detection.tracker_2
            yolo_model = tracker.initialize_yolo()
            output_dir = os.path.join('results', tracker.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            video_name = os.path.basename(video_path)
            img_files = glob.glob(video_path + "/*jpg")
            img_files.sort()

            out_res = []
            pred_bbox = [0] # no prection
            start_time = time.time()
            for frame_id in range(len(img_files)):
                if frame_id % 100 == 0:
                    print ('video_id: {}/{}, frame_id: {}'.format(video_id, video_num, frame_id))
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

                if self.visulization:
                    cv2.imshow(video_name, im_vis)
                    cv2.waitKey(1)

                frame_id += 1
            if self.visulization:
                cv2.destroyAllWindows()
            end_time = time.time()
            FPS = len(img_files) / (end_time - start_time)
            print(str(video_num) + ' - ' + str(video_id) + ' : ' + str(FPS))
            # save result
            output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
            with open(output_file, 'w') as f:
                json.dump({'res': out_res}, f)

        # use for validation set
        if True:
            if not os.path.isdir(output_dir):
                print("%s doesn't exist" % output_dir)

            if os.path.isdir(output_dir) and os.path.isdir(truth_dir):
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

                        pred_file = os.path.join(output_dir, video_dirsbase+'_%s.txt' % mode)

                        try:
                            with open(pred_file, 'r') as f:
                                pred_res = json.load(f)
                                pred_res=pred_res['res']
                        except:
                            with open(pred_file, 'r') as f:
                                pred_res = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
                                pred_res[:, 2:] = pred_res[:, 2:] - pred_res[:, :2] + 1
                            
                        mixed_measure = self.eval(pred_res, label_res)
                        overall_performance.append(mixed_measure)

                #output_file.write(str(np.mean(overall_performance)))
                #print(np.mean(overall_performance))
                #output_file.write("correct:1")
                output_file.write('correct:{:}'.format(np.mean(overall_performance)))
                print('correct:{:}'.format(np.mean(overall_performance)))

                output_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
