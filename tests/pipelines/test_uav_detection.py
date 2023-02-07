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
from modelscope.utils.cv.image_utils import draw_face_detection_no_lm_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level
from modelscope.utils.constant import DownloadMode


class UavDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.task = Tasks.domain_specific_object_detection
        self.model_id = 'damo/cv_tinynas_uav-detection_damoyolo'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        uav_detection = pipeline(self.task, model=self.model_id)
        img_path = 'data/test/images/uav_detection.jpg'

        result = uav_detection(img_path)
        self.show_result(img_path, result)

    def show_result(self, img_path, detection_result):
        img = draw_face_detection_no_lm_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
