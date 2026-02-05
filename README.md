# DRASHTI-HaOBB Dataset
**D**rone nadi**R**-view **A**nnotated image**S** of ve**H**icles datase**T** for **I**ndia (**DRASHTI**) - **H**eading-**a**ngle **O**riented **B**ounding **B**ox (**HaOBB**).

**Download** the dataset from the Zenodo platform🔗: **https://zenodo.org/records/18278989**

The DRASHTI-HaOBB dataset is organised into six ZIP files, containing images and corresponding labels separately for the training, validation, and test splits. After downloading, the dataset must be reorganised into the format described below. Each image directory contains JPEG images, and the corresponding label directory contains annotations in a text file.

**Dataset structure**
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
**For training and testing various models from different OOD ( Oriented Object Detection) frameworks (YOLO and MMRotate) on the DRASHTI-HaOBB dataset, follow the instructions given in the respective directories.**
