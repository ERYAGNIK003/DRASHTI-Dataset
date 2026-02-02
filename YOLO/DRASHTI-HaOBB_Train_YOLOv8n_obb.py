from ultralytics import YOLO
import torch

import os
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8n-obb.yaml')  #Select the model yaml file of your choice like yolov8s-obb.yaml, yolov8m-obb.yaml, yolov8l-obb.yaml, yolov8x-obb.yaml
model.train(
    data='DRASHTI-HaOBB.yaml', # Give the path of the DRASTI dataset yaml file.
    epochs=30,
    imgsz=1024,
    batch=64,
    workers=16,
    device=device,
    project='/train_project_folder' # Give the path of Folder where the training files are stored.
)

model.val(
    data='DRASHTI-HaOBB.yaml', # Give the path of the DRASTI dataset yaml file.
    project='/train_project_folder' #Give the path of Folder where the validating files are stored.
)
