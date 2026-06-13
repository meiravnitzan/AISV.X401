# Traffic Sign Classification with Targeted Generative Augmentation

## Problem
This project builds a complete deep learning system for traffic sign classification on GTSRB. The main experiment tests whether targeted generative augmentation can improve a weak class without meaningfully hurting overall performance.

## Dataset
The project uses the German Traffic Sign Recognition Benchmark (GTSRB), a 43-class traffic sign dataset. The official training and test zip files are unpacked automatically by `dataset.py`.

## Model
The classifier is a pretrained ResNet-18 with a new final layer for 43 classes. Two modes are supported:
- `baseline`: original real training data only.
- `gen22`: real data plus synthetic class-22 images.

## Training
### Baseline
```bash
python train.py --config config.yaml --mode baseline
```

### Evaluate baseline
```bash
python evaluate.py --config config.yaml --checkpoint models/best_baseline.pt --tag baseline
```

### Generative augmentation
Generate synthetic class-22 images into `data/synth_class22/` using the Stable Diffusion img2img notebook workflow.

### Train augmented model
```bash
python train.py --config config.yaml --mode gen22
```

### Evaluate augmented model
```bash
python evaluate.py --config config.yaml --checkpoint models/best_gen22.pt --tag gen22
```

## Results
Final measured results:
- Baseline accuracy: `0.9941409342834521`
- Baseline macro-F1: `0.9920009159817401`
- Gen22 accuracy: `0.9930324623911322`
- Gen22 macro-F1: `0.9893305423529042`

Class 22 comparison:
- Baseline: precision `1.0`, recall `0.9`, F1 `0.9473684210526315`
- Gen22: precision `1.0`, recall `0.9833333333333333`, F1 `0.9915966386554622`

Interpretation: targeted generative augmentation slightly lowered global metrics, but significantly improved the previously weak class-22 recall and F1.

## Demo
Run single-image inference with:
```bash
python demo.py --config config.yaml --checkpoint models/best_baseline.pt --image path/to/image.ppm
```

## Repository Structure
```text
project/
├── data/
├── notebooks/
├── models/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── inference.py
├── demo.py
├── config.yaml
├── requirements.txt
└── README.md
```
