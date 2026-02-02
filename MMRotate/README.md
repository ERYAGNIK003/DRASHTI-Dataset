
# Benchmarking MMRotate Framework on DRASHTI-HaOBB Dataset using NVIDIA GPUs

This repository provides the complete workflow to train and test MMRotate models using the **DRASHTI-HaOBB** dataset on **NVIDIA
Quadro RTX 6000/8000 GPU**. It includes environment setup, dataset configuration, model training, and evaluation steps.

---

## 📦 Environment Setup

Create and activate the virtual environment:

```bash
python3.8 -m venv mmrotate_env
source mmrotate_env/bin/activate
pip install --upgrade pip
```

Install PyTorch and essential dependencies (for CUDA 12.4):

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -U openmim
mim install mmcv
mim install mmdet\<3.0.0
```

Clone and install MMRotate:

```bash
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

---

## 📁 Dataset Preparation

1. Download and unzip the **DRASHTI-HaOBB dataset** into a folder of your preference.
2. **Modify the structure** of the dataset as follows.
   Change the structure from this
   ```
    DRASHTI-HaOBB/
    ├── images/
    │   ├── train/
    │   └── val/
    |   └── test/
    └── labels/
        ├── train_original/
        └── val_original/
        └── test_original/
    ```
   to
   ```
    DRASHTI-HaOBB/
    ├── train/
    │   ├── images/
    │   └── labels/
    |── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    ```
---

## ⚙️ Configuration Steps

### 1. Edit Dataset Path in `dotav1.py`

File: `mmrotate/configs/_base_/datasets/dotav1.py`

- Change the value of `data_root` to your local DRASTI dataset path.
- Modify the following parameters accordingly:
```python
train=dict(
    ann_file='train/labels/',
    img_prefix='train/images/',
    ...
)
val=dict(
    ann_file='val/labels/',
    img_prefix='val/images/',
    ...
)
test=dict(
    ann_file='test/labels/',
    img_prefix='test/images/',
    ...
)
```

### 2. Update Classes in `dota.py`

File: `mmrotate/mmrotate/datasets/dota.py`

- Replace the `CLASSES` tuple with the following:
```python
CLASSES = (
    'Auto3WCargo', 'AutoRicksaw', 'Bus', 'Container', 'Mixer',
    'MotorCycle', 'PickUp', 'SUV', 'Sedan', 'Tanker',
    'Tipper', 'Trailer', 'Truck', 'Van'
)
```

- Also change any file endings from `.png` to `.jpg` in this file to match the DRASTI image format.
- Ensure `.jpg` format is used for all image files in the DRASTI dataset.

### 3. Enable TensorBoard Logging

File: `mmrotate/configs/_base_/default_runtime.py`

- Uncomment the following line to enable TensorBoard logging:
```python
dict(type='TensorboardLoggerHook')
```

### 4. Modify Model Config for Number of Classes

File: `mmrotate/configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py`

- Set the number of classes to 14:
```python
model = dict(
    bbox_head=dict(
        num_classes=14
    )
)
```
- Modify batch size, learning rate, or other hyperparameters in the config file if needed.
---

## 🏋️ Training

To begin training, use the following command inside the MMRotate directory:

```bash
python tools/train.py configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py
```

This will start the training process, and logs will be saved to the `work_dirs` directory automatically created.
- Checkpoints will be saved in `work_dirs/<experiment_name>/`.

To monitor training progress, you can use TensorBoard:

```bash
load_ext tensorboard
tensorboard --logdir work_dirs/s2anet_r50_fpn_1x_dota_le135
```

---

## ✅ Testing and 📊 Visualizing Results

After training, use the following command to test the trained model:

```bash
python tools/test.py configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py work_dirs/s2anet_r50_fpn_1x_dota_le135/latest.pth --work-dir inference_s2anet --show-dir inference_s2anet/imgs --eval mAP
```

Results will include the mean Average Precision (mAP) and class-wise metrics. This will also generate visualizations in the `inference_s2anet/imgs` directory.


## Get Flops and Params
To get the FLOPs and parameters of the model, use the following command:

```bash
python tools/analysis_tools/get_flops.py s2anet_r50_fpn_1x_dota_le135.py 
```

## 📈 Benchmarking

To FPS, Inference time of the model, you can use the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=1 tools/analysis_tools/benchmark.py s2anet_r50_fpn_1x_dota_le135.py work_dirs/s2anet_r50_fpn_1x_dota_le135/latest.pth  --max-iter 7000 --log-interval 100  --launcher pytorch
```

## Model Size
 Download the file `getmodelsize.py` and run the following command to get the model size:
```bash
python getmodelsize.py
```

---

## 🔗 Acknowledgements

- [DRASTI Dataset Authors](https://github.com/YourDatasetLinkIfAny)
- [MMRotate](https://github.com/open-mmlab/mmrotate)

## Citation
If you use this code or dataset in your research, please cite the following:

<!-- ```bibtex
@article{dasti2023,
  title={DRASTI: A Large-Scale Dataset for Object Detection in Indian Traffic Scenes},
  author={Your Name and Co-authors},
  journal={Journal of Computer Vision},
  year={2023},
  volume={XX},
  pages={XX-XX}
}
``` -->


