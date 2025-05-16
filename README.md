
![GitHUB](https://github.com/user-attachments/assets/26a90352-eb52-4f8b-8d2a-ca7406a4978d)


# SUMGraph (ObjectVisA-120)
SUMGraph model given ObjectVisA-120 dataset

*The object-based nature of human visual attention is well-known in cognitive science but has only played a minor role in computational visual attention models so far. This is mainly due to a lack of suitable datasets and evaluation metrics for object-based attention. To address these limitations, we present ObjectVisA-120 – a novel 120-participant dataset of spatial navigation in virtual reality specifcally geared to object-based attention evaluations. ObjectVisA-120 not only features accurate gaze data and a complete state-space representation of objects in the virtual environment, but it also offers variable scenario complexities and rich annotations, including panoptic segmentation, depth information, and vehicle keypoints. We further propose object-based similarity (oSIM) as a novel metric to evaluate the performance of object-based visual attention models, a previously unexplored performance characteristic. Our evaluations show that explicitly optimising for object-based attention not only improves oSIM performance but also leads to an improved model performance on common metrics. In addition, we present SUMGraph, a Mamba U-Net-based model, which explicitly encodes scene objects in a graph representation, leading to further performance improvements over several state-of-the-art visual attention prediction methods.*

## Installation
Ensure you have Python >= 3.10 installed on your system. Then, install the required libraries and dependencies.

## Requirements
Install PyTorch and other necessary libraries:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Pre-trained Weights
Download the SUMGraph model from the provided cloud link and move it to the specified directory:
```bash
Download SUMGraph model: sumgraph_model.pth
Change the loading filepath in XXX
```

## Inference

To generate saliency maps, use the `inference.py` script. Here are the steps and commands:

```bash
python inference.py
```

## Training
Run the training process:

1. **Pretrained Encoder Weights**: Download from [VMamba GitHub]

```bash
python train.py
```
## Validation
Run the validation process:

1. **Pretrained Encoder Weights**:

```bash
python validation.py
```


