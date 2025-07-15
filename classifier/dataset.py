import random
from collections import defaultdict
from typing import cast, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


class FewShotDataset:
    """Wrapper over a base dataset to sample few-shot learning episodes."""

    def __init__(self, base_dataset: Dataset) -> None:
        """
        Args:
            base_dataset (Dataset): PyTorch dataset where each item returns (image, label).
        """
        self.base_dataset = base_dataset
        self.class_to_indices = self._build_index()

    def _build_index(self) -> dict[int, list[int]]:
        """Builds a mapping from class labels to sample indices.

        Returns:
            dict[int, list[int]]: Dictionary mapping class labels to list of indices.
        """
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(cast(Iterable, self.base_dataset)):
            class_to_indices[int(label)].append(idx)
        return class_to_indices

    def sample_episode(
            self, n_way: int = 4, k_shot: int = 1, q_query: int = 5
    ) -> tuple[Subset, Subset, list[int]]:
        """Samples an episode from the dataset.

        Args:
            n_way (int): Number of classes in the episode.
            k_shot (int): Number of support samples per class.
            q_query (int): Number of query samples per class.

        Returns:
            tuple[Subset, Subset, list[int]]: Support set, query set, and list of selected class labels.
        """
        if n_way > len(self.class_to_indices):
            raise ValueError(f"Requested {n_way} classes, but only {len(self.class_to_indices)} available.")

        selected_classes = random.sample(list(self.class_to_indices.keys()), n_way)
        support_indices, query_indices = [], []

        for cls in selected_classes:
            indices = self.class_to_indices[cls]
            if len(indices) < k_shot + q_query:
                raise ValueError(f"Not enough samples for class {cls}: required {k_shot + q_query}, got {len(indices)}")
            selected = random.sample(indices, k_shot + q_query)
            support_indices.extend(selected[:k_shot])
            query_indices.extend(selected[k_shot:])

        support_set = Subset(self.base_dataset, support_indices)
        query_set = Subset(self.base_dataset, query_indices)

        return support_set, query_set, selected_classes


class InferenceDataset(Dataset):
    """Dataset for running inference on a list of image files."""

    def __init__(self, image_paths: list[str], transform: transforms.Compose) -> None:
        """
        Args:
            image_paths (list[str]): List of image file paths.
            transform (transforms.Compose): Transform to apply (e.g., torchvision transforms).
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Args:
            idx (int): Index of the image.

        Returns:
            tuple[torch.Tensor, str]: Transformed image and its file path.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return image, image_path
