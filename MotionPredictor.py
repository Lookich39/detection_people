import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch

class MotionPredictor:
    def __init__(self, history_size=5, search_expansion=5):
        self.track_history = {}  # {id: deque([(x1,y1,x2,y2), ...])}
        self.history_size = history_size
        self.search_expansion = search_expansion  # Коэффициент расширения области поиска
        
    def predict_search_areas(self, active_tracks):
        """Возвращает прогнозируемые области поиска для пропавших объектов"""
        predictions = {}
        
        # Находим пропавшие треки
        active_ids = {t['id'] for t in active_tracks}
        missing_ids = set(self.track_history.keys()) - active_ids
        
        for track_id in missing_ids:
            if len(self.track_history[track_id]) >= 2:
                # Линейная экстраполяция
                dx, dy = self._calc_movement_vector(track_id)
                last_bbox = self.track_history[track_id][-1]
                
                # Расширенная область поиска
                w = last_bbox[2] - last_bbox[0]
                h = last_bbox[3] - last_bbox[1]
                
                predicted_x1 = last_bbox[0] + dx - w*(self.search_expansion-1)/2
                predicted_y1 = last_bbox[1] + dy - h*(self.search_expansion-1)/2
                predicted_x2 = last_bbox[2] + dx + w*(self.search_expansion-1)/2
                predicted_y2 = last_bbox[3] + dy + h*(self.search_expansion-1)/2
                
                predictions[track_id] = (
                    int(predicted_x1), int(predicted_y1),
                    int(predicted_x2), int(predicted_y2)
                )
        
        return predictions
    
    def _calc_movement_vector(self, track_id):
        """Вычисляет средний вектор движения"""
        history = list(self.track_history[track_id])
        dx_total, dy_total = 0, 0
        
        for i in range(1, len(history)):
            prev_center = ((history[i-1][0] + history[i-1][2])/2, (history[i-1][1] + history[i-1][3])/2)
            curr_center = ((history[i][0] + history[i][2])/2, (history[i][1] + history[i][3])/2)
            dx_total += curr_center[0] - prev_center[0]
            dy_total += curr_center[1] - prev_center[1]
        
        avg_dx = dx_total / (len(history)-1)
        avg_dy = dy_total / (len(history)-1)
        return avg_dx, avg_dy

def process_video_with_search_areas(model, video_path):
    cap = cv2.VideoCapture(video_path)
    predictor = MotionPredictor(history_size=5, search_expansion=1.5)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Получаем текущие треки
        results = model.track(
            frame,
            classes=[0],  # Только класс 0 (люди)
            conf=0.5,
            iou=0.4,
            persist=True,
            tracker="botsort.yaml",
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        current_tracks = []
        
        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                    bbox = tuple(map(int, box[:4]))
                    current_tracks.append({'id': int(track_id), 'bbox': bbox})
                
                    # Обновляем историю
                    if int(track_id) not in predictor.track_history:
                        predictor.track_history[int(track_id)] = deque(maxlen=predictor.history_size)
                    predictor.track_history[int(track_id)].append(bbox)
        
        # 2. Получаем области поиска для пропавших объектов
        search_areas = predictor.predict_search_areas(current_tracks)
        
        # 3. Визуализация
        for track in current_tracks:
            x1, y1, x2, y2 = track['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)  # Зеленый - текущие
            
        for track_id, area in search_areas.items():
            # Красный прямоугольник - область поиска
            cv2.rectangle(frame, (area[0], area[1]), (area[2], area[3]), (0,0,255), 1)
            
            # Желтая точка - прогнозируемый центр
            pred_center = ((area[0]+area[2])//2, (area[1]+area[3])//2)
            cv2.circle(frame, pred_center, 5, (0,255,255), -1)
            
            cv2.putText(frame, f"Search {track_id}", (area[0], area[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        cv2.imshow("Motion Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Запуск
model = YOLO('yolov8n.pt')
process_video_with_search_areas(model, "in_video.mp4")