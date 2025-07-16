import argparse

import cv2

from detector.detector import Detector


def main() -> None:
    """Letters extraction from an image."""
    parser = argparse.ArgumentParser(description='Extract letters from an image')
    parser.add_argument('--image_path', type=str, default='data/letters.png', help='Path to the image')
    parser.add_argument('--save_dir', type=str, default='data/query', help='Path to save the extracted letters')

    args = parser.parse_args()

    detector = Detector()

    img = cv2.imread(args.image_path)

    detector.save_letters(img, args.save_dir)


if __name__ == '__main__':
    main()
