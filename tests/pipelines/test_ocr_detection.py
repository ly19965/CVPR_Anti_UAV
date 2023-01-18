# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class OCRDetectionTest(unittest.TestCase, DemoCompatibilityCheck):

    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet18_ocr-detection-line-level_damo'
        self.model_id_vlpt = 'damo/cv_resnet50_ocr-detection-vlpt'
        self.test_image = 'data/test/images/ocr_detection.jpg'
        self.test_image_vlpt = 'data/test/images/ocr_detection_vlpt.jpg'
        self.task = Tasks.ocr_detection

    def pipeline_inference(self, pipeline: Pipeline, input_location: str):
        result = pipeline(input_location)
        print('ocr detection results: ')
        print(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_model_from_modelhub(self):
        ocr_detection = pipeline(Tasks.ocr_detection, model=self.model_id)
        self.pipeline_inference(ocr_detection, self.test_image)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_with_vlpt_with_model_from_modelhub(self):
        ocr_detection = pipeline(Tasks.ocr_detection, model=self.model_id_vlpt)
        self.pipeline_inference(ocr_detection, self.test_image_vlpt)

    @unittest.skipUnless(test_level() >= 2, 'skip test in current test level')
    def test_run_modelhub_default_model(self):
        ocr_detection = pipeline(Tasks.ocr_detection)
        self.pipeline_inference(ocr_detection, self.test_image)

    @unittest.skip('demo compatibility test is only enabled on a needed-basis')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
