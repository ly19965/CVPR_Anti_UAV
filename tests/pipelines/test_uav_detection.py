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


class FaceDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.video_multi_object_tracking
        self.model_id = 'damo/cv_manual_uav-detection_uav'
        # set own path
        self.dataset_dir = '/root/.cache/modelscope/hub/datasets/damo/Anti_UAV/master/data_files/extracted/7c4621266fb54a0b0a70fafdaeb58e462a18124a5e7fdda4d342cf8a15e60ae2/Anti_UAV_test_dev/'

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
        for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
            measure_per_frame.append(self.not_exist(_pred) if not _exist else self.iou(_pred, _gt) if len(_pred) > 1 else 0)
        return np.mean(measure_per_frame)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):

        uav_validation = MsDataset.load('Anti_UAV', namespace='damo', split='validation')
        video_num = len(uav_validation)
        overall_performance = []
        video_id = 1
        mode = 'IR'
        for info in uav_validation:
            uav_detection = pipeline(self.task, model=self.model_id)
            tracker = uav_detection.tracker
            output_dir = os.path.join('results', tracker.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            video_name = os.path.basename(os.path.join(self.dataset_dir, info['Type']))
            video_file = os.path.join(self.dataset_dir, info['InputVideo:File'])
            res_file = os.path.join(self.dataset_dir, info['Category:File'])
            with open(res_file, 'r') as f:
                label_res = json.load(f)

            init_rect = label_res['gt_rect'][0]
            capture = cv2.VideoCapture(video_file)

            frame_id = 0
            out_res = []
            while True:
                ret, frame = capture.read()
                if not ret:
                    capture.release()
                    break
                if frame_id == 0:
                    tracker.init(frame, init_rect)  # initialization
                    out = init_rect
                    out_res.append(init_rect)
                else:
                    out = tracker.update(frame)  # tracking
                    out_res.append(out.tolist())
                frame_id += 1
            # save result
            output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
            with open(output_file, 'w') as f:
                json.dump({'res': out_res}, f)

            mixed_measure = self.eval(out_res, label_res)
            overall_performance.append(mixed_measure)
            video_id += 1
            print('[%03d/%03d] %20s %5s Fixed Measure: %.03f' % (video_id, video_num, video_name, mode, mixed_measure))

        print('[Overall] %5s Mixed Measure: %.03f\n' % (mode, np.mean(overall_performance)))



    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
