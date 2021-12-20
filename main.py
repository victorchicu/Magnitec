import os
import cv2

from imageai.Detection import VideoObjectDetection


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def detect_object():
    camera = cv2.VideoCapture(0)
    execution_path = os.getcwd()
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, "ptm/resnet50_coco_best_v2.1.0.h5"))
    detector.loadModel()
    video_path = detector.detectObjectsFromVideo(
        camera_input=camera,
        output_file_path=os.path.join(execution_path, "camera_detected_video"),
        frames_per_second=20,
        log_progress=True,
        minimum_percentage_probability=30
    )
    print(video_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detect_object()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
