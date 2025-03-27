import os

import cv2
import torch
from yolov10.ultralytics import YOLOv10
# Load model directly
from transformers import AutoModel

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results
# defining function for creating a writer (for mp4 videos)
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'x264')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

if __name__ == "__main__":
    torch.set_num_threads(1)
    cuda = '0'
    video_path = '/data1/wjx/S003/input/video/divided1.mp4'
    output_filename = video_path.replace('input','output') #'/data1/wjx/S003/output/divided1.mp4'
    cap = cv2.VideoCapture(video_path)
    writer = create_video_writer(cap, output_filename)
    if torch.cuda.is_available():
        model = YOLOv10.from_pretrained('jameslahm/yolov10x')
        # model = AutoModel.from_pretrained("jameslahm/yolov10x")
        # model = YOLO("yolov10x.pt").to("cuda")
    else:
        model = YOLOv10.from_pretrained('jameslahm/yolov10x')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    while True:
        success, img = cap.read()
        if not success:
            break
        result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
        writer.write(result_img)
        cv2.imshow("Image", result_img)

        cv2.waitKey(1)
    cap.release()