# Benchmarking YOLO11 Models on DRASHTI-HaOBB Dataset using Stepwell HPC (High Performance Computing) facility

This repository provides the complete workflow for benchmarking YOLO11 models on the **DRASHTI-HaOBB** dataset using **HPC clusters with the Slurm job scheduler**. It includes environment setup, data preparation, model training, testing, and benchmarking steps.

---

## 📦 Environment Setup (HPC Modules)

Before starting, ensure the appropriate modules are loaded, and the Python virtual environment is created:

```bash
module load Python/3.10.15
module load cuda/latest       # Typically uses torch-2.7.1+cu126
module list                   # Verify Python and CUDA are loaded

python3.10 -m venv yolo_env
source ~/yolo_env/bin/activate
pip install --upgrade pip
pip install ultralytics       # Version used: 8.3.159

mkdir NewFolder
cd ~/NewFolder
```

---

## 📁 Step 1: Dataset Preparation

1. **Download** and unzip the `DRASHTI-HaOBB` dataset and `DRASHTI-HaOBB.yaml` inside `NewFolder`.

2. Update the dataset path in the `DRASHTI-HaOBB.yaml` file.

3. **Annotation Conversion for YOLO OBB**

   - Navigate to the converter.py script:
     ```
     yolo_env/lib/python3.10/site-packages/ultralytics/data/converter.py
     ```

   - Edit function `convert_dota_to_yolo_obb()`:
     - Around **line 458**, update `class_mapping`:
       ```python
       class_mapping = {
           "Auto3WCargo": 0,
           "AutoRicksaw": 1,
           "Bus": 2,
           "Container": 3,
           "Mixer": 4,
           "MotorCycle": 5,
           "PickUp": 6,
           "SUV": 7,
           "Sedan": 8,
           "Tanker": 9,
           "Tipper": 10,
           "Trailer": 11,
           "Truck": 12,
           "Van": 13,
       }
       ```

     - Around **line 499**, where it says `for split in {"train", "val"}:`, update:
       ```python
       for split in {"train", "val", "test"}:
       ```

     - Around **line 508**, in place of `.png` change:
       ```python
       image_ext = ".jpg"
       ```

4. Run the conversion:
- This will create the necessary YOLO OBB format annotations in the `DRASHTI-HaOBB/labels/` directory with names `train, val, test`.
```python
from ultralytics.data.converter import convert_dota_to_yolo_obb
convert_dota_to_yolo_obb("../DRASHTI-HaOBB") # Adjust the path to your DRASTI dataset
```

5. The final structure becomes:
```
DRASHTI-HaOBB/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── train_original/
    ├── val/
    ├── val_original/
    ├── test/
    └── test_original/
```

---

## 🏋️ Step 2: Training the YOLO11 OBB Model

1. Download the training script `DRASHTI-HaOBB_Train_YOLO11n_obb.py` and place it in your working directory.

2. Configure:
   - Path to `DRASHTI-HaOBB.yaml`
   - Output directory for results (`project` path in script)
   - Choose desired model YAML file (e.g., `yolo11n-obb.yaml`)

3. Prepare the training bash script `DRASHTI-HaOBB_Train_YOLO11n_obb.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=DRASHTI-HaOBB_Train_YOLO11n_obb
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --partition=gpu

module load Python/3.10.15

# Activate your virtual environment
source ~/yolo_env/bin/activate

# Debug GPU availability before training
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Run the training
python DRASHTI-HaOBB_Train_YOLO11n_obb.py
```

4. Launch training:
```bash
sbatch DRASHTI-HaOBB_Train_YOLO11n_obb.sh
```

5. Monitor progress:
- Check the job status:
```bash
squeue
```

6. Training logs:
   - Check the Training files like `result.csv` by navigating to the `project` directory specified in the script.
   - Logs are saved in `logs/` directory:
     - `output.log` for standard output.
     - `error.log` for errors.
   - Model weights and metrics are saved in the `project` directory specified in the training script.
   - Make a plot of the training result like mAP, loss, from `results.csv` file in sheets or excel.

---

## ✅ Testing

1. Download and configure `DRASHTI-HaOBB_Test_YOLO11n_obb.py`:
   - Set path to `DRASHTI-HaOBB.yaml`
   - Set path to trained model and output `project` folder

2. Create a bash file `DRASHTI-HaOBB_Test_YOLO11n_obb.sh` similar to the training script and execute:
```bash
sbatch DRASHTI-HaOBB_Test_YOLO11n_obb.sh
```

3. Testing outputs:
   - Metrics (mAP, Precision, Recall)
   - Confusion matrix
   - Saved in the `project` folder

---

## ⚡ Benchmarking (parameters like mAP50-95,  Parameters(in M), GFLOPs, Model Size, Inference, FPS, Input Size)

1. Download `DRASHTI-HaOBB_Benchmark_YOLO.py`.

2. Update:
   - Paths to trained model `.pt` files
   - Path to `DRASHTI-HaOBB.yaml`

3. Internally, the script uses a temporary copy of the `DRASHTI-HaOBB.yaml` file and points to the test set for evaluation.

4. Run:
```bash
python DRASHTI-HaOBB_Benchmark_YOLO.py
```

---

## 🎥 Inference on Video

1. Download `DRASHTI-HaOBB_Inference_YOLO.py`.

2. Configure the script:
   - Model path
   - Input video path
   - Output video path for annotated video results

3. Run:
```bash
python DRASHTI-HaOBB_Inference_YOLO.py
```

---

## 📂 Directory Summary

```
NewFolder/
├── DRASHTI-HaOBB/
│   ├── images/
│   └── labels/
├── DRASHTI-HaOBB.yaml
├── DRASHTI-HaOBB_Train_YOLO11n_obb.py
├── DRASHTI-HaOBB_Test_YOLO11n_obb.py
├── DRASHTI-HaOBB_Benchmark_YOLO.py
├── DRASHTI-HaOBB_Inference_YOLO.py
├── DRASHTI-HaOBB_Train_YOLO11n_obb.sh
├── DRASHTI-HaOBB_Test_YOLO11n_obb.sh
├── logs/
└── runs/
```

---

## 📌 Notes

- Ensure `ultralytics` version is consistent across training, testing, and benchmarking.
- Adapt SLURM job scripts according to your HPC scheduler and resource availability.
- Always activate the virtual environment before running scripts.

---

## 📜 Citation

If you use this pipeline or dataset in your research, please cite:

- The DRASHTI-HaOBB Dataset paper
```bibtex
```
- Ultralytics YOLOv11 framework

---

## 🔗 Acknowledgements

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- DRASTI Dataset Authors
- Stepwell HPC Support Team
