import io
from enum import nonmember
from typing import Dict
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from torchvision.datasets import ImageFolder
from pydantic import BaseModel, Field

from classifier.classifier import compute_prototypes
from classifier.dataset import FewShotDataset, create_transform
from detector.detector import Detector
from test import load_model, create_support_set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
class Config:
    MODEL_PATH = "checkpoints/fewshot_protonet_20250715-203358.pth"
    SUPPORT_PATH = "data/dataset"
    K_SHOT = 3
    N_WAY = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_FILE_SIZE = 14 * 1024 * 1024  # 14MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# --- Response Models ---
class PredictionResponse(BaseModel):
    letter_counts: Dict[str, int] = Field(..., description="Count of detected letters")
    total_detections: int = Field(..., description="Total number of detections")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional error details")


# --- Global Variables ---
model: torch.nn.Module | None = None
prototypes: torch.Tensor | None = None
support_class_names: list[str] | None = None
detector: Detector | None = None
transform: transforms.Compose | None = None

def load_image_from_bytes(img_bytes: bytes) -> np.ndarray:
    """Load and convert image from bytes to numpy array."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid image format: {str(e)}"
        )


def classify_crop(crop: np.ndarray) -> str:
    """Classify a single letter crop."""
    try:
        pil_crop = Image.fromarray(crop)
        input_tensor = transform(pil_crop).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            query_feat = model(input_tensor)
            dists = torch.cdist(query_feat, prototypes)
            probs = torch.softmax(-dists, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return support_class_names[pred]
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return "unknown"


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )

def initialize_model() -> tuple[torch.nn.Module, torch.Tensor, list[str]]:
    """Initialize model, prototypes, and class names."""
    try:
        # Validate paths
        if not Path(Config.MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found: {Config.MODEL_PATH}")
        if not Path(Config.SUPPORT_PATH).exists():
            raise FileNotFoundError(f"Support dataset not found: {Config.SUPPORT_PATH}")

        logger.info(f"Loading model from {Config.MODEL_PATH}")
        model = load_model(Config.MODEL_PATH)

        logger.info(f"Loading support dataset from {Config.SUPPORT_PATH}")
        support_transform = create_transform()
        support_dataset = ImageFolder(Config.SUPPORT_PATH, transform=support_transform)
        fs_dataset = FewShotDataset(support_dataset)

        # Create support set
        support_images, support_labels, class_names = create_support_set(
            fs_dataset, n_way=Config.N_WAY, k_shot=Config.K_SHOT
        )

        support_images = support_images.to(Config.DEVICE)
        support_labels = support_labels.to(Config.DEVICE)

        # Compute prototypes
        with torch.no_grad():
            support_features = model(support_images)
            prototypes = compute_prototypes(
                support_features,
                support_labels,
                list(range(len(class_names)))
            )

        logger.info(f"Model initialized successfully. Classes: {class_names}")
        return model, prototypes, class_names

    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global model, prototypes, support_class_names, detector, transform

    try:
        # Startup
        logger.info("Starting up application...")
        model, prototypes, support_class_names = initialize_model()
        detector = Detector()
        transform = create_transform()
        logger.info("Application startup complete")

        yield

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down application...")
        # Add any cleanup code here if needed

# --- Init app ---
app = FastAPI(
    title="Letter Detection API",
    description="API for detecting and classifying letters in images",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "device": str(Config.DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict letters in an uploaded image.

    Args:
        file: Uploaded image file

    Returns:
        PredictionResponse with letter counts and processing time
    """
    import time
    start_time = time.time()

    try:
        # Validate file
        validate_file(file)

        # Check file size
        img_bytes = await file.read()
        file_size = len(img_bytes)

        if file_size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE / (1024 * 1024):.1f}MB"
            )

        # Load image
        img_np = load_image_from_bytes(img_bytes)

        # Detect letters
        bboxes = detector.predict(img_np)
        scaled_bboxes = detector.scale_bboxes(img_np, bboxes)

        letter_counts: Dict[str, int] = {}

        # Process each detected bounding box
        for bbox in scaled_bboxes:
            try:
                x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

                # Validate bounding box
                if x < 0 or y < 0 or x + w > img_np.shape[1] or y + h > img_np.shape[0]:
                    logger.warning(f"Invalid bounding box: {bbox}")
                    continue

                crop = img_np[y:y + h, x:x + w]

                # Skip empty crops
                if crop.size == 0:
                    continue

                class_name = classify_crop(crop)
                letter_counts[class_name] = letter_counts.get(class_name, 0) + 1

            except Exception as e:
                logger.error(f"Error processing bounding box {bbox}: {str(e)}")
                continue

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            letter_counts=letter_counts,
            total_detections=sum(letter_counts.values()),
            processing_time_ms=round(processing_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
