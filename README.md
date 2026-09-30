# Brain Tumor Detection with YOLO11

Object detection of brain tumors in MRI images using [Ultralytics YOLO11](https://docs.ultralytics.com/). Two model sizes (YOLO11s and YOLO11m) were fine-tuned from pretrained weights and compared on a four-class dataset: **Glioma, Meningioma, Pituitary tumor, and No Tumor**.

## Results

Evaluated on the validation split (1,745 images, 1,784 labeled instances), images resized to 512 px.

| Model | Parameters | Precision | Recall | mAP@50 | mAP@50-95 | Inference (per image) |
|---|---|---|---|---|---|---|
| **YOLO11s** | 9.4M | 0.886 | 0.864 | **0.908** | 0.553 | 4.6 ms |
| YOLO11m | 20.0M | 0.881 | 0.852 | 0.899 | 0.553 | 11.3 ms |

The smaller model matches the larger one on mAP@50-95 and is slightly ahead on mAP@50 while running about 2.5x faster, so YOLO11s is the better accuracy/latency trade-off on this dataset.

### Per-class results (YOLO11s)

| Class | Images | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Glioma | 541 | 0.792 | 0.702 | 0.783 | 0.406 |
| Meningioma | 516 | 0.934 | 0.951 | 0.982 | 0.732 |
| No Tumor | 279 | 0.962 | 0.968 | 0.982 | 0.605 |
| Pituitary | 409 | 0.857 | 0.834 | 0.883 | 0.471 |

Meningioma and No Tumor are detected almost perfectly at IoU 0.5. **Glioma is the hardest class** (lowest recall and localization quality), which is the natural target for future improvement.

## Approach

1. **Data audit**: count images per class from the YOLO-format label files
2. **Training**: fine-tune `yolo11s.pt` and `yolo11m.pt` (COCO-pretrained) for up to 150 epochs, image size 512, batch size 8, early-stopping patience 20
3. **Validation**: per-class precision, recall, mAP@50 and mAP@50-95 with Ultralytics' validator
4. **Inference**: run the best checkpoint on the test images and save annotated predictions

**Dataset:** a Roboflow-exported, YOLO11-format brain tumor dataset (`Brain Tumor.v3i.yolov11`) with 7,218 training and 1,745 validation images. The dataset is not redistributed in this repo.

**Environment used:** Ultralytics 8.3.175, PyTorch 2.5.1 (CUDA 12.1), Python 3.12, NVIDIA RTX 4050 Laptop GPU (6 GB).

## Repository structure

```
Brain-Tumor-Detection/
├── tumor_detection.ipynb      # data audit, training, validation, inference
├── prediction_result/         # YOLO11s run outputs
├── prediction_results/predict # sample predictions on test images
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/ishaan175pathak/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
pip install -r requirements.txt
```

Download the dataset in YOLO11 format (folders `train/`, `valid/`, `test/` plus `data.yaml`) into the project root, then run `tumor_detection.ipynb`. The core commands are:

```bash
# train
yolo task=detect mode=train model=yolo11s.pt data="Brain Tumor.v3i.yolov11/data.yaml" \
     epochs=150 imgsz=512 batch=8 device=0 patience=20

# validate
yolo task=detect mode=val model=path/to/best.pt data="Brain Tumor.v3i.yolov11/data.yaml"

# predict
yolo task=detect mode=predict model=path/to/best.pt source="Brain Tumor.v3i.yolov11/test/images"
```

## Limitations and next steps

- Metrics are reported on the validation split; a held-out test-set evaluation would strengthen the results
- Glioma performance lags the other classes; class-balanced sampling, stronger augmentation, or higher input resolution are natural next experiments
- Research and educational project only. It is **not** a medical device and must not be used for clinical decisions

## Tech stack

Python · PyTorch · Ultralytics YOLO11 · Pandas · NumPy · Matplotlib · Jupyter

## Author

**Ishaan Pathak** · [GitHub](https://github.com/ishaan175pathak)
