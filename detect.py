import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from easy_deep_sort import EasyDeepSort
import torch


MAX_FRAMES = 1000
TARGET_CLS = 0 # human
THRESH_DETECTION = 0.25
MATCHING_THRESH_METRIC = 0.5
ENC_MODEL_PATH = "/home/market/yolov8_deepsort_pose/networks/networks/mars-small128.pb"
easy_ds = EasyDeepSort(
    enc_model_path=ENC_MODEL_PATH,
    metric="cosine",
    matching_threshold=MATCHING_THRESH_METRIC
)

video_path = "test.mp4"
output_video = "result.mp4"



video_cap = cv2.VideoCapture(video_path)

ret, frame = video_cap.read()
cap_out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MP4V'), video_cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

print(frame.shape)
model = YOLO("yolov8l-pose.pt")
# model = YOLO("yolov8n.pt")

count = 0
while 1:
    
    if count%1==0:
        preds = model(frame)
        for p in preds:
            detections = []
            for r in p.boxes.data.tolist():
                print(r)
                x1, y1, x2, y2, score, cls_id = r
                cls_id = int(cls_id)
                if cls_id==TARGET_CLS:
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    if score > THRESH_DETECTION:
                        detections.append([x1, y1, x2-x1, y2-y1, score])

            easy_ds.run(frame, detections)
            annotator = Annotator(
                frame,
                line_width=2,
                font_size=2,
                font='Arial.ttf',
                pil=False,  # Classify tasks default to pil=True
                example=p.names)
            
            # Plot Pose results
            if p.keypoints is not None:
                # print(p.keypoints.data)
                for k in reversed(p.keypoints.data):
                    annotator.kpts(k, frame.shape, radius=5, kpt_line=True)
                    
            
            for r in p.boxes.data.tolist():
                print(r)
                x1, y1, x2, y2, score, cls_id = r
                cls_id = int(cls_id)
                if cls_id==TARGET_CLS:
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    color = (77,23,255)
                    p1, p2 = (x1, y1), (x2, y2)
                    # cv2.rectangle(frame, p1, p2, color, 2)
                    
                    label = "{score:0.3f}".format(score=score)
                    w, h = cv2.getTextSize(label, 0, fontScale=0.5, thickness=3)[0]  # text width, height
                    outside = p1[1] - h >= 5
                    p2 = p1[0] + w, p1[1] - h - 5 if outside else p1[1] + h + 5
                    cv2.rectangle(frame, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(frame,
                                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                0,
                                0.5,
                                color,
                                thickness=1,
                                lineType=cv2.LINE_AA)
                    # annotator.box_label(d.xyxy.squeeze(), name, color=colors(c, True))


            frame = easy_ds.draw_tracks(frame)
            cv2.imshow("kpts", annotator.result())
            cap_out.write(frame)
            cv2.waitKey(1)
            
            # print(r.boxes)
            # print(p.keypoints.xy)
        
    ret, frame = video_cap.read()
    if not ret or count>MAX_FRAMES:
        print(ret)
        break

    count += 1

cap_out.release()
cap_out.release()
cv2.destroyAllWindows()

