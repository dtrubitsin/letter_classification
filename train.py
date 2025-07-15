import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

from classifier.classifier import ViTFeatureExtractor, compute_prototypes
from classifier.dataset import FewShotDataset, create_transform, set_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_episode(
        model: ViTFeatureExtractor,
        support_loader: DataLoader,
        query_loader: DataLoader,
        classes: list[int],
        optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    """Runs a single training or evaluation episode.

    Args:
        model (ViTFeatureExtractor): Feature extractor model.
        support_loader (DataLoader): Loader for support set.
        query_loader (DataLoader): Loader for query set.
        classes (list[int]): List of class indices in the episode.
        optimizer (torch.optim.Optimizer): Optimizer for training.

    Returns:
        tuple[float, float]: Loss and accuracy for the episode.
    """
    model.eval()
    support_images, support_labels = next(iter(support_loader))
    query_images, query_labels = next(iter(query_loader))

    support_images = support_images.to(DEVICE)
    query_images = query_images.to(DEVICE)
    support_labels = torch.tensor([classes.index(lab) for lab in support_labels], device=DEVICE)
    query_labels = torch.tensor([classes.index(lab) for lab in query_labels], device=DEVICE)

    with torch.set_grad_enabled(True):
        support_feat = model(support_images)
        query_feat = model(query_images)
        prototypes = compute_prototypes(support_feat, support_labels, list(range(len(classes))))

        dists = torch.cdist(query_feat, prototypes)
        loss = F.cross_entropy(-dists, query_labels)
        acc = (dists.argmin(dim=1) == query_labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), acc.item()


def train_fewshot(
        dataset: FewShotDataset,
        n_way: int,
        k_shot: int,
        q_query: int,
        n_episodes: int,
        lr: float,
        pretrained: bool,
        model_path: str,
        log_dir: str
) -> None:
    """Trains a few-shot classification model using episodic training.

    Args:
        dataset (FewShotDataset): Few-shot dataset wrapper.
        n_way (int): Number of classes per episode.
        k_shot (int): Number of support samples per class.
        q_query (int): Number of query samples per class.
        n_episodes (int): Number of episodes to train.
        lr (float): Learning rate.
        pretrained (bool): Whether to use a pretrained ViT backbone.
        model_path (str): Path to save the trained model.
        log_dir (str): Directory for TensorBoard logs.
    """
    model = ViTFeatureExtractor(pretrained=pretrained).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"\n🚀 Starting training on {n_episodes} episodes...")

    for episode in range(1, n_episodes + 1):
        support_set, query_set, classes = dataset.sample_episode(n_way, k_shot, q_query)
        support_loader = DataLoader(support_set, batch_size=n_way * k_shot)
        query_loader = DataLoader(query_set, batch_size=n_way * q_query)

        loss, acc = run_episode(model, support_loader, query_loader, classes, optimizer)

        writer.add_scalar("Loss/train", loss, episode)
        writer.add_scalar("Accuracy/train", acc, episode)

        print(f"[{episode:02d}/{n_episodes}] Loss: {loss:.4f} | Acc: {acc * 100:.2f}%")

    writer.close()
    torch.save(model.state_dict(), model_path)
    print(f"\n✅  Training complete. Model saved to {model_path}")


def main():
    """Few-Shot Image Classification with ViT train script"""
    parser = argparse.ArgumentParser(description="Few-Shot Image Classification with ViT")
    parser.add_argument('--dataset-path', type=str, default='data/dataset', help='Path to dataset folder')
    parser.add_argument('--episodes', type=int, default=5, help='Number of training episodes')
    parser.add_argument('--n-way', type=int, default=4, help='Number of classes per episode')
    parser.add_argument('--k-shot', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--q-query', type=int, default=5, help='Number of query samples per class')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained ViT backbone')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Directory to save model and logs')

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(args.output_dir, f"fewshot_protonet_{timestamp}.pth")
    log_dir = os.path.join(args.output_dir, "logs", timestamp)

    dataset = ImageFolder(args.dataset_path, transform=create_transform())
    fs_dataset = FewShotDataset(dataset)

    print(f"📁 Loaded {len(dataset)} images from {len(dataset.classes)} classes.")
    print(f"📦 Using device: {DEVICE}")

    train_fewshot(
        dataset=fs_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        n_episodes=args.episodes,
        lr=args.lr,
        pretrained=args.pretrained,
        model_path=model_path,
        log_dir=log_dir
    )


if __name__ == '__main__':
    main()
