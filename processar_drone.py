import cv2
import json
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
simulation_data = {}

video_path = "drone_video.mp4"
cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output_drone.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.25, verbose=False)
    
    r = results[0]

    annotated_frame = r.plot()
   
    out.write(annotated_frame)
    
    if r.boxes is not None and r.boxes.id is not None:
        coords = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        
        frame_data = []
        for i in range(len(ids)):
            frame_data.append({
                "vehicle_id": int(ids[i]),
                "bounding_box_xyxy": [float(c) for c in coords[i]]
            })
            
        simulation_data[f"frame_{frame_idx}"] = frame_data
        
    print(f"Processado frame {frame_idx}")
    frame_idx += 1

cap.release()
out.release()

with open('drone_trajectories.json', 'w') as json_file:
    json.dump(simulation_data, json_file, indent=4)

print("Processing complete. Video saved as 'output_drone.avi' and data as 'drone_trajectories.json'.")