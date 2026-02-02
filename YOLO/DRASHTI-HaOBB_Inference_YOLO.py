from ultralytics import YOLO
import supervision as sv
import cv2

# Load model
model = YOLO(".pt") # Give the path of weight file on which the inference is required

# Create annotator with black color and thicker lines
obb_annotator = sv.OrientedBoxAnnotator(
    color=sv.Color.from_hex("#000000"),  # Black
    thickness=3
)

# Read and write video
video_path = "video.MP4" # Give the path of video file on which the inferencing is required
cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "inferenced_video.mp4", # Give the path of output video file
    cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply prediction with specified arguments
    results = model.predict(
        source=frame,
        conf=0.25,
        imgsz=1024,
        device='cuda:0',
        agnostic_nms=True,
        verbose=False
    )[0]

    obb_detections = sv.Detections.from_ultralytics(results)
    frame = obb_annotator.annotate(scene=frame, detections=obb_detections)
    out.write(frame)

cap.release()
out.release()
print("✅ Video processing complete.")
