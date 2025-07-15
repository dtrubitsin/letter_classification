import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Import your existing classes and functions
from classifier.classifier import ViTFeatureExtractor, compute_prototypes
from classifier.dataset import FewShotDataset, InferenceDataset, create_transform, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str) -> ViTFeatureExtractor:
    """Loads a trained few-shot model from the given path.

    Args:
        model_path (str): Path to the model checkpoint.

    Returns:
        ViTFeatureExtractor: Loaded and ready model.
    """
    model = ViTFeatureExtractor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def collate_fn(batch: list) -> tuple[torch.Tensor, list]:
    """Custom collate function for batching image and path pairs.

    Args:
        batch: List of (image, path) pairs.

    Returns:
        Tuple of batched images and list of corresponding paths.
    """
    images = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    return images, paths


def create_support_set(fs_dataset: FewShotDataset, n_way: int = 4, k_shot: int = 1) -> tuple:
    """Create support set.

    Args:
        fs_dataset (FewShotDataset): Few-shot dataset wrapper.
        n_way (int): Number of classes per episode.
        k_shot (int): Number of support examples per class.

    Returns:
        tuple: (support_images, support_labels, class_names)
    """
    # Sample an episode to get support set
    support_set, _, selected_classes = fs_dataset.sample_episode(n_way, k_shot, q_query=0)

    # Create dataloader for support set
    support_loader = DataLoader(support_set, batch_size=len(support_set), shuffle=False)
    support_images, support_labels = next(iter(support_loader))

    # Convert original labels to episode labels (0, 1, 2, ...)
    episode_labels = torch.tensor([selected_classes.index(label.item()) for label in support_labels])

    # Get class names from the base dataset
    base_dataset = fs_dataset.base_dataset
    if hasattr(base_dataset, 'classes'):
        class_names = [base_dataset.classes[cls] for cls in selected_classes]
    else:
        class_names = [f"class_{cls}" for cls in selected_classes]

    return support_images, episode_labels, class_names


def run_inference(model, support_images, support_labels, class_names, query_images):
    """Run inference on query images using prototypical networks.

    Args:
        model: Trained ViTFeatureExtractor model.
        support_images: Support set images tensor.
        support_labels: Support set labels tensor.
        class_names: List of class names.
        query_images: Query images tensor.

    Returns:
        tuple: (predictions, max_confidences, all_confidences)
    """
    model.eval()

    with torch.no_grad():
        # Move to device
        support_images = support_images.to(DEVICE)
        support_labels = support_labels.to(DEVICE)
        query_images = query_images.to(DEVICE)

        # Extract features
        support_feat = model(support_images)
        query_feat = model(query_images)

        # Compute prototypes
        prototypes = compute_prototypes(support_feat, support_labels, list(range(len(class_names))))

        # Compute distances and predictions
        dists = torch.cdist(query_feat, prototypes)
        predictions = dists.argmin(dim=1)
        confidences = F.softmax(-dists, dim=1)

        # Get max confidence for each prediction
        max_confidences = confidences.max(dim=1)[0]

    return predictions.cpu().numpy(), max_confidences.cpu().numpy(), confidences.cpu().numpy()


def get_image_paths(folder_path: str) -> list[str]:
    """Get all image paths from a folder.

    Args:
        folder_path: Path to folder containing images.

    Returns:
        list: Sorted list of image paths.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = []

    folder_path = Path(folder_path)
    for ext in image_extensions:
        image_paths.extend(folder_path.glob(f"*{ext}"))
        image_paths.extend(folder_path.glob(f"*{ext.upper()}"))

    return sorted(image_paths)


def prepare_data(args: argparse.Namespace) -> tuple:
    print(f"Creating support set from {args.support_data}")

    # Create base dataset for support set
    base_dataset = ImageFolder(args.support_data, transform=create_transform())
    fs_dataset = FewShotDataset(base_dataset)

    print(f"📁 Loaded {len(base_dataset)} images from {len(base_dataset.classes)} classes.")
    print(f"Available classes: {base_dataset.classes}")

    # Create support set
    support_images, support_labels, class_names = create_support_set(
        fs_dataset, n_way=args.n_way, k_shot=args.k_shot
    )

    print(f"Selected {len(class_names)} classes for this episode: {class_names}")
    print(f"Support set contains {len(support_images)} images")

    # Get query images
    print(f"Loading query images from {args.query_folder}")
    query_paths = get_image_paths(args.query_folder)

    if not query_paths:
        raise FileNotFoundError("No images found in query folder!")

    print(f"Found {len(query_paths)} images for inference")

    # Create dataset and dataloader using existing InferenceDataset
    query_dataset = InferenceDataset(query_paths, transform=create_transform())
    query_loader = DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    return support_images, support_labels, class_names, query_loader


def main():
    """Few-Shot Image Classification with ViT test script"""
    parser = argparse.ArgumentParser(description="Run inference on images using trained few-shot classifier")
    parser.add_argument("--model_path", type=str, default="fewshot_protonet.pth", help="Path to trained model")
    parser.add_argument("--support_data", type=str, required=True,
                        help="Path to support dataset (same structure as training data)")
    parser.add_argument("--query_folder", type=str, required=True,
                        help="Path to folder containing images for inference")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="Output file for predictions")
    parser.add_argument("--n_way", type=int, default=4, help="Number of classes per episode")
    parser.add_argument("--k_shot", type=int, default=3, help="Number of support examples per class")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)

    support_images, support_labels, class_names, query_loader = prepare_data(args)

    # Run inference
    print("Running inference...")
    results = []

    for batch_images, batch_paths in query_loader:
        predictions, confidences, full_confidences = run_inference(
            model, support_images, support_labels, class_names, batch_images
        )

        for i, (pred, conf, full_conf, img_path) in enumerate(
                zip(predictions, confidences, full_confidences, batch_paths)
        ):
            predicted_class = class_names[pred]

            # Create confidence dict for all classes
            confidence_dict = {class_names[j]: float(full_conf[j]) for j in range(len(class_names))}

            result = {
                "image_path": str(img_path),
                "predicted_class": predicted_class,
                "confidence": float(conf),
                "all_confidences": confidence_dict,
                "high_confidence": float(conf) >= args.confidence_threshold,
            }

            results.append(result)

            # Print result
            status = "✅" if result["high_confidence"] else "⚠️"
            print(f"{status} {Path(img_path).name}: {predicted_class} ({conf:.3f})")

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_file}")

    # Print summary
    high_conf_count = sum(1 for r in results if r["high_confidence"])
    print(f"\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"High confidence predictions: {high_conf_count}")
    print(f"Low confidence predictions: {len(results) - high_conf_count}")

    # Print class distribution
    class_counts = {}
    for result in results:
        class_counts[result["predicted_class"]] = class_counts.get(result["predicted_class"], 0) + 1

    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    print(f"\nEpisode classes used: {class_names}")


if __name__ == "__main__":
    main()
