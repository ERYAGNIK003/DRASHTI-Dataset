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

## 📜 Citation

If you use this pipeline or dataset in your research, please cite:

- The DRASHTI-HaOBB Dataset paper
```
@ARTICLE{11421776,
  author={Bhavsar, Yagnik M. and Zaveri, Mazad S. and Raval, Mehul S. and Patel, Krishna R. and Zaveri, Shaheriar B.},
  journal={IEEE Data Descriptions}, 
  title={Descriptor: Drone Nadir-view Annotated Images of Vehicles Detection Dataset for India with Heading-angle Oriented Bounding Box (DRASHTI-HaOBB)}, 
  year={2026},
  volume={},
  number={},
  pages={1-12},
  keywords={Annotations;Road traffic;Roads;Drones;Object detection;Road accidents;Autonomous aerial vehicles;Videos;Driver behavior;Safety;Oriented bounding box (OBB);Oriented object detection (OOD);UAV-based data collection;Nadir-view;Vehicle detection dataset},
  doi={10.1109/IEEEDATA.2026.3670752}}
```
- DRASHTI-HaOBB Dataset 
```
@dataset{bhavsar_2026_18278989,
  author= {Bhavsar, Yagnik and Zaveri, Mazad and Raval, Mehul and Zaveri, Shaheriar and Ahmedabad University},
  title= {DRASHTI-HaOBB: Drone nadiR-view Annotated imageS of veHicles dataseT for India - Heading-angle Oriented Bounding Box: Indian Vehicle Oriented- Object-Detection Dataset with 1.3 Million Samples},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.18278989},
  url          = {https://doi.org/10.5281/zenodo.18278989},
}
```
