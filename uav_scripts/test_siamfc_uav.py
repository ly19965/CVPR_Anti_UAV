from modelscope.utils.cv.image_utils import show_video_tracking_result
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model='damo/cv_alex_video-single-object-tracking_siamfc-uav')
video_path = "./data/test/videos/uav.avi"
init_bbox = [302, 317, 390, 384] # the initial object bounding box in the first frame [x1, y1, x2, y2]
result = video_single_object_tracking((video_path, init_bbox))
show_video_tracking_result(video_path, result[OutputKeys.BOXES], "./tracking_result.avi")
print("result is : ", result[OutputKeys.BOXES])
