import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class MultiStepPredictor:
    def __init__(self, history_size=10, search_expansion=1.5, 
                prediction_interval=5, max_predictions=6, max_time_to_keep=10):
        self.track_history = {}
        self.history_size = history_size
        self.search_expansion = search_expansion
        self.prediction_interval = prediction_interval
        self.max_predictions = max_predictions
        self.max_time_to_keep = max_time_to_keep

    def predict_search_areas(self, active_tracks, lost_objects, fps):
        predictions = {}
        active_ids = {t['id'] for t in active_tracks}

        max_frames_to_keep = int(self.max_time_to_keep * fps)
        
        for track_id in list(self.track_history.keys()):
            if track_id in lost_objects and lost_objects[track_id] > max_frames_to_keep:
                del self.track_history[track_id]
                continue
            if track_id not in active_ids and track_id in lost_objects:
                frames_lost = lost_objects[track_id]
                
                if len(self.track_history[track_id]) >= 2:
                    last_bbox = self.track_history[track_id][-1]
                    w = last_bbox[2] - last_bbox[0]
                    h = last_bbox[3] - last_bbox[1]
                    
                    # Генерация предсказаний с учетом интервала
                    predictions[track_id] = []
                    for step in range(1, self.max_predictions + 1):
                        pred_frame = frames_lost + step * self.prediction_interval
                        uncertainty = 1.0 + (step * 0.2)
                        
                        # Расчет смещения
                        dx, dy = self._calc_movement_vector(track_id)
                        pred_dx = dx * pred_frame * uncertainty
                        pred_dy = dy * pred_frame * uncertainty
                        
                        # Расчет области поиска
                        expanded_w = w * self.search_expansion * uncertainty
                        expanded_h = h * self.search_expansion * uncertainty
                        
                        bbox = (
                            int(last_bbox[0] + pred_dx - expanded_w/2),
                            int(last_bbox[1] + pred_dy - expanded_h/2),
                            int(last_bbox[2] + pred_dx + expanded_w/2),
                            int(last_bbox[3] + pred_dy + expanded_h/2)
                        )
                        
                        predictions[track_id].append((pred_frame, bbox))
        
        return predictions

    def _calc_movement_vector(self, track_id):
        """Вычисляет средний вектор движения с учетом ускорения"""
        history = list(self.track_history[track_id])
        if len(history) < 2:
            return 0, 0
        
        # Рассчитываем смещения между всеми парами кадров
        displacements = []
        for i in range(1, len(history)):
            prev_center = np.array([(history[i-1][0] + history[i-1][2])/2, 
                                   (history[i-1][1] + history[i-1][3])/2])
            curr_center = np.array([(history[i][0] + history[i][2])/2, 
                                   (history[i][1] + history[i][3])/2])
            displacements.append(curr_center - prev_center)
        
        # Усредняем с учетом последних смещений
        weights = np.linspace(0.5, 1.5, len(displacements))  # Больший вес новым данным
        avg_dx = np.average([d[0] for d in displacements], weights=weights)
        avg_dy = np.average([d[1] for d in displacements], weights=weights)
        
        return avg_dx, avg_dy

