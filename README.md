# Letters Classification

A computer vision system that detects and classifies letters in images using few-shot learning with Vision Transformer (
ViT) backbone and prototypical networks.

## 🚀 Features

- **Few-shot learning**: Train on minimal examples per class using prototypical networks
- **Vision Transformer backbone**: Leverages pre-trained ViT for robust feature extraction
- **Letter detection**: Automatic bounding box detection for letters in images
- **REST API**: FastAPI-based web service for real-time inference
- **Flexible architecture**: Easy to extend to new letter classes or alphabets

## 📁 Project Structure

```
letters/
├── src/
│   ├── classifier/
│   │   ├── classifier.py          # ViT feature extractor and prototype computation
│   │   └── dataset.py             # Few-shot dataset wrapper and data transforms
│   ├── detector/
│   │   └── detector.py            # Letter detection model
│   ├── train.py                   # Training script for few-shot learning
│   ├── test.py                    # Inference script for batch processing
│   ├── inference_api.py           # FastAPI web service
│   └── extract_letters.py         # Prepare individual letters
└── data/
    └── letters.png                # Image with letters for train/test
```

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, but recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd letters

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Structure

Organize your dataset in ImageFolder format:

```
data/dataset/
├── A/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── B/
│   ├── img1.jpg
│   └── ...
└── ...
```

## 🎯 Usage

### Extract letters

Extract letters from one large image to create dataset:

```bash
python3 src/extract_letters.py \
    --image_path data/letters.png \
    --save_dir data/query
```

**Parameters:**

- `--image_path`: Path to the image with letters
- `--save_dir`: Output images directory path

### Training

Before train create `dataset` directory as shown in the [Dataset Structure](#-dataset-structure).  
Train a few-shot classification model:

```bash
python3 src/train.py \
    --dataset-path data/dataset \
    --episodes 5 \
    --n-way 4 \
    --k-shot 3 \
    --q-query 5 \
    --lr 1e-4 \
    --pretrained \
    --output-dir checkpoints
```

**Parameters:**

- `--dataset-path`: Path to training dataset
- `--episodes`: Number of training episodes
- `--n-way`: Number of classes per episode
- `--k-shot`: Support examples per class
- `--q-query`: Query examples per class
- `--lr`: Learning rate
- `--pretrained`: Use pre-trained ViT backbone
- `--output-dir`: Directory for saving models and logs

### Batch Inference

Run inference on a folder of images:

```bash
python3 test.py \
    --model_path checkpoints/fewshot_protonet_20250715-203358.pth \
    --support_data data/dataset \
    --query_folder data/query \
    --output_file data/predictions.json \
    --n_way 4 \
    --k_shot 3 \
    --confidence_threshold 0.5
```

**Parameters:**

- `--model_path`: Path to trained model
- `--support_data`: Path to support set dataset
- `--query_folder`: Folder containing images for inference (extracted from main image)
- `--output_file`: JSON file for saving predictions
- `--confidence_threshold`: Minimum confidence for predictions

### Web API

Start the FastAPI server:

```bash
python3 src/inference_api.py
```

The API will be available at `http://127.0.0.1:8000`

#### API Endpoints

**Health Check**

```
GET /health
```

**Predict Letters**

```
POST /predict
Content-Type: multipart/form-data
Body: image file
```

**Example Response:**

```json
{
  "letter_counts": {
    "pi": 103,
    "lambda": 142,
    "mu": 162,
    "sigma": 293    
  },
  "total_detections": 700,
  "processing_time_ms": 40705.17
}
```

#### Using curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/letters.png"
```

## 🏗️ Architecture

### Few-Shot Learning Pipeline

1. **Feature Extraction**: Vision Transformer extracts rich features from images
2. **Prototype Computation**: Calculate class prototypes from support set examples
3. **Distance-based Classification**: Classify queries based on proximity to prototypes
4. **Episodic Training**: Train on randomly sampled episodes to improve generalization

### Detection + Classification

1. **Letter Detection**: Detect letter bounding boxes in input image
2. **Crop Extraction**: Extract individual letter crops from detections
3. **Few-shot Classification**: Classify each crop using the trained model
4. **Aggregation**: Count and return letter frequency statistics

## 📈 Monitoring

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir checkpoints/logs
```

View metrics:

- Training loss per episode
- Classification accuracy
- Learning curves

## ⚙️ Configuration

Edit the `Config` class in `src/inference_api.py` to adjust:

- Model checkpoint path
- Support dataset path
- Number of shots and classes
- File size limits
- Allowed image formats

```python
class Config:
    MODEL_PATH = "checkpoints/fewshot_protonet_20250715-203358.pth"
    SUPPORT_PATH = "data/dataset"
    K_SHOT = 3
    N_WAY = 4
    MAX_FILE_SIZE = 14 * 1024 * 1024  # 14MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
```