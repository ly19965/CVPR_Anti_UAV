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

class AntiUavTrack1Test(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.video_multi_object_tracking
        self.model_id = 'damo/3rd_Anti-UAV_CVPR23'
        track1_test_set = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='track1_test')
        assert track1_test_set is not None, 'test set should be downloaded first'
        # set own path
        self.dataset_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/4bc8e5d84d77bd5c4abb007b7a5861f4ae7833711d6ad390d214b6589f9956fb/track1_test'
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
        for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
            measure_per_frame.append(self.not_exist(_pred) if not _exist else self.iou(_pred, _gt) if len(_pred) > 1 else 0)
        return np.mean(measure_per_frame)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        video_paths = glob.glob(os.path.join(self.dataset_dir, '*'))
        video_num = len(video_paths)
        overall_performance = []
        video_id = 1
        mode = 'IR'

        # run tracking experiments and report performance
        for video_id, video_path in enumerate(video_paths, start=1):
            uav_detection = pipeline(self.task, model=self.model_id)
            tracker = uav_detection.tracker
            output_dir = os.path.join('results', tracker.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            video_name = os.path.basename(video_path)
            img_files = glob.glob(video_path + "/*jpg")
            res_file = os.path.join(video_path, '%s_label.json'%mode)
            with open(res_file, 'r') as f:
                label_res = json.load(f)

            init_rect = label_res['gt_rect'][0]
            out_res = []
            start_time = time.time()
            for frame_id in range(len(img_files)):
                if frame_id % 100 == 0:
                    print ('vidio_id: {}/{}, frame_id: {}'.format(video_id, video_num, frame_id))
                frame = cv2.imread(img_files[frame_id])
                if frame_id == 0:
                    tracker.init(frame, init_rect)  # initialization
                    out = init_rect
                    out_res.append(init_rect)
                else:
                    out = tracker.update(frame)  # tracking
                    out_res.append(out.tolist())
                if self.visulization:
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
            if self.visulization:
                cv2.destroyAllWindows()
            end_time = time.time()
            FPS = len(img_files) / (end_time - start_time)
            print(str(video_num)+' - '+str(video_id)+' : '+str(FPS))
            # save result
            output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, mode))
            with open(output_file, 'w') as f:
                json.dump({'res': out_res}, f)


    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
