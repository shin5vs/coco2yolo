# COCO to YOLOv8 Segmentation Converter

This script converts datasets from the **COCO format** (used for object detection and instance segmentation) to the **YOLOv8 format**, with support for both **bounding boxes** and **segmentation polygons**.

## Features

- Converts COCO-style JSON annotations to YOLO-style `.txt` files.
- Supports both **bounding box** and **segmentation** annotations.
- Automatically organizes images and labels into `train`, `valid`, and `test` directories.
- Optionally **copies images** into a new directory structure or just references them.
- Generates a `data.yaml` file compatible with **Ultralytics YOLOv8**.

## Directory Structure

| Path                         | Description                            |
|------------------------------|----------------------------------------|
| `dataset_root/`              | Root output directory                  |
| ├── `images/`                | Image directories by split             |
| │   ├── `train/`             | Training images                        |
| │   ├── `valid/`             | Validation images                      |
| │   └── `test/`              | Test images                            |
| ├── `labels/`                | YOLO annotation directories by split   |
| │   ├── `train/`             | Training labels                        |
| │   ├── `valid/`             | Validation labels                      |
| │   └── `test/`              | Test labels                            |
| ├── `train.txt`              | List of training image paths           |
| ├── `val.txt`                | List of validation image paths         |
| ├── `test.txt`               | List of test image paths               |
| └── `data.yaml`              | YOLOv8 configuration file              |
