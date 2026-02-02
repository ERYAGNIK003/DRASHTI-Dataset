from ultralytics import YOLO
import torch
import yaml

import os
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

model = YOLO('.pt') # Give the path of weight file on which the testing is required

# Run validation on test set
metrics = model.val(
    data='DRASHTI-HaOBB.yaml', # Give the path of the DRASTI dataset yaml file.
    imgsz=1280,
    project='test_project_folder', # Give the path of Folder where the testing files are stored.
    split='test',
    agnostic_nms = True,
    plots = True,
    verbose = True
)

# Load class names from YAML
with open('DRASHTI-HaOBB.yaml', 'r') as f:
    data_yaml = yaml.safe_load(f)

names = data_yaml['names']

# Access and print metrics
print(f"mAP50-95 (OBB): {metrics.box.map:.3f}")
print(f"mAP50 (OBB):    {metrics.box.map50:.3f}")
print(f"mAP75 (OBB):    {metrics.box.map75:.3f}")
for i, ap in enumerate(metrics.box.maps):
    print(f"Class: {names[i]:<15} AP: {ap:.3f}")

confusion_matrix_csv = metrics.confusion_matrix.to_csv()

# Define output path for the CSV
output_csv_path = 'confusion_matrix.csv' # Give the path of Folder where the csv files are stored.

# Write to file
with open(output_csv_path, 'w') as f:
    f.write(confusion_matrix_csv)

print(f"Confusion matrix saved to: {output_csv_path}")
