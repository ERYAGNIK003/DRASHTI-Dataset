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
