
import random
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools.generate_detections import create_box_encoder
import numpy as np
import cv2 

class EasyDeepSort:
    """
    Parameters
    ----------
    model_path: path to 
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
        
    """
        
    def __init__(self, enc_model_path, metric, matching_threshold, budget=None) -> None:
        
        metric = nn_matching.NearestNeighborDistanceMetric(
            metric, matching_threshold, budget)
        self.tracker = Tracker(metric)
        self.encoder = create_box_encoder(enc_model_path, batch_size=1) # we are predicting one frame
        self.results = []
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]



    def prepare(self, detections):
        bboxes = np.asarray([d[:-1] for d in detections])
        scores = [d[-1] for d in detections]
        return bboxes, scores

    def run(self, img, detections):
        
        bboxes, scores = self.prepare(detections)
        features = self.encoder(img, bboxes)

        dets = self.create_detections(bboxes=bboxes, scores=scores, features=features)
        
        # Update tracker.
        self.tracker.predict()
        self.tracker.update(dets)


        # Store results.
        self.results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr() # cocnvert to xyxy (yolov8 format)
            self.results.append([track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    def create_detections(self, bboxes, scores, features):
        detection_list = []
        for i, bbox in enumerate(bboxes):
            detection_list.append(Detection(bbox, scores[i], features[i]))
        return detection_list
    
    def draw_tracks(self, img):
        for res in self.results:
            track_id, x1, y1, x2, y2 = res
            # we use % (rem) in case there is not enough colors
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (self.colors[track_id % len(self.colors)]), 2)
        return img
    




