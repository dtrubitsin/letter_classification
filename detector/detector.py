import os
from dataclasses import dataclass

import cv2
import numpy as np

# --- CONFIG ---
BBOX_SCALE = 1.1
SAVE_DIR = "data/dataset"


@dataclass
class BBox:
    """Represents a bounding box."""
    x: int
    y: int
    width: int
    height: int


class Detector:
    """Simple detector using connected components."""

    @staticmethod
    def sort_components_bookwise(stats: np.ndarray) -> list[np.ndarray]:
        """
        Sorts connected components in a book-wise (top-to-bottom, left-to-right) fashion.

        Args:
            stats (np.ndarray): Array of shape [N, 5] with component statistics.

        Returns:
            List[np.ndarray]: Sorted list of stats arrays (components).
        """
        median_h = np.median(stats[:, cv2.CC_STAT_HEIGHT])
        sorting_window = median_h * 0.5

        sorted_indices = stats[:, cv2.CC_STAT_TOP].argsort()
        stats = stats[sorted_indices]

        rows = []
        used = np.zeros(len(stats), dtype=bool)

        for i, stat in enumerate(stats):
            if used[i]:
                continue
            y = stat[cv2.CC_STAT_TOP]
            in_row = np.abs(stats[:, cv2.CC_STAT_TOP] - y) < sorting_window
            in_row = in_row & (~used)
            used |= in_row
            row = stats[in_row]
            rows.append(sorted(row, key=lambda x: x[cv2.CC_STAT_LEFT]))

        return [item for row in rows for item in row]

    def predict(self, img: np.ndarray) -> list[BBox]:
        """
        Detects connected components as bounding boxes in an image.

        Args:
            img (np.ndarray): Input image (H, W, C) or (H, W).

        Returns:
            List[BBox]: List of detected bounding boxes.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # Skip the first component (background)
        stats = stats[1:]

        # Sort detections
        sorted_stats = self.sort_components_bookwise(stats)

        # Create bounding boxes
        bboxes = [
            BBox(
                x=int(s[cv2.CC_STAT_LEFT]),
                y=int(s[cv2.CC_STAT_TOP]),
                width=int(s[cv2.CC_STAT_WIDTH]),
                height=int(s[cv2.CC_STAT_HEIGHT])
            )
            for s in sorted_stats
        ]
        return bboxes

    @staticmethod
    def visualize_bboxes(img: np.ndarray, bboxes: list[BBox]) -> np.ndarray:
        """
        Draws bounding boxes on the image.

        Args:
            img (np.ndarray): Original image.
            bboxes (List[BBox]): Bounding boxes to draw.

        Returns:
            np.ndarray: Image with boxes drawn.
        """
        img_out = img.copy()
        for bbox in bboxes:
            pt1 = (bbox.x, bbox.y)
            pt2 = (bbox.x + bbox.width, bbox.y + bbox.height)
            cv2.rectangle(img_out, pt1, pt2, (0, 255, 0), 4)
        return img_out

    def save_letters(self, img: np.ndarray) -> None:
        """
        Extracts and saves cropped letter regions to disk.

        Args:
            img (np.ndarray): Original image with letters.
        """
        os.makedirs(SAVE_DIR, exist_ok=True)

        bboxes = self.predict(img)
        scaled_bboxes = self.scale_bboxes(img, bboxes)

        for idx, bbox in enumerate(scaled_bboxes):
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            crop = img[y:y + h, x:x + w]
            save_path = os.path.join(SAVE_DIR, f"{idx}.jpg")
            cv2.imwrite(save_path, crop)

    @staticmethod
    def scale_bboxes(img: np.ndarray, bboxes: list[BBox]) -> list[BBox]:
        """
        Scales bounding boxes uniformly with clipping to image bounds.

        Args:
            img (np.ndarray): Original image.
            bboxes (List[BBox]): List of bounding boxes.

        Returns:
            List[BBox]: Scaled bounding boxes.
        """
        h, w = img.shape[:2]
        scaled = []

        for bbox in bboxes:
            new_w = bbox.width * BBOX_SCALE
            new_h = bbox.height * BBOX_SCALE
            dx = (new_w - bbox.width) / 2
            dy = (new_h - bbox.height) / 2

            new_x = int(np.clip(bbox.x - dx, 0, w - 1))
            new_y = int(np.clip(bbox.y - dy, 0, h - 1))
            new_w = int(np.clip(new_w, 1, w - new_x))
            new_h = int(np.clip(new_h, 1, h - new_y))

            scaled.append(BBox(x=new_x, y=new_y, width=new_w, height=new_h))

        return scaled
