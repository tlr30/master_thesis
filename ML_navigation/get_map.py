"""
Create map from drawing 

This script provides functionality to convert a hand-drawn or digital image 
into a binary black-and-white format and downscale it for use in navigation 
or robotics applications. It includes:

- Conversion to black and white using a threshold
- Downscaling while preserving essential black pixel information
- Saving both the processed image and its corresponding NumPy array
- Command-line interface for easy parameter control

Dependencies: OpenCV (cv2), NumPy, argparse, pathlib, Streamlit (optional), tempfile

Author: Tim Riekeles
Date: 2025-08-05
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
import streamlit as st
import tempfile


def convert_to_black_white(input_path: str, threshold: int = 128) -> np.ndarray:
    """
    Convert an image to black and white using a specified threshold.

    Args:
        input_path (str): Path to the input image.
        threshold (int): Threshold value for binary conversion. Defaults to 128.

    Returns:
        np.ndarray: The black-and-white image as a NumPy array.

    Raises:
        FileNotFoundError: If the input image cannot be loaded.
    """
    ## load the image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Input image not found at {input_path}")

    ## convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## apply threshold to convert to black and white
    _, bw_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)

    return bw_image


def downscale_preserve_black(bw_image: np.ndarray, scale_percent: float) -> np.ndarray:
    """
    Downscale a binary image while preserving black pixels.

    Args:
        bw_image (np.ndarray): Binary image (0 or 255) to downscale.
        scale_percent (float): Percentage to scale the image.

    Returns:
        np.ndarray: Downscaled binary image.
    """
    ## ensure input is binary (0 or 255)
    if not np.array_equal(np.unique(bw_image), [0, 255]):
        raise ValueError("Image must be binary (0 or 255) before resizing.")

    ## calculate new dimensions based on the percentage
    width = int(bw_image.shape[1] * scale_percent / 100)
    height = int(bw_image.shape[0] * scale_percent / 100)
    new_size = (width, height)

    ## resize using INTER_AREA to preserve black pixels (0)
    downscaled_image = cv2.resize(bw_image, new_size, interpolation=cv2.INTER_AREA)

    ## ensure the final image is strictly black & white
    # _, downscaled_bw = cv2.threshold(downscaled_image, 128, 255, cv2.THRESH_BINARY)
    _, downscaled_bw = cv2.threshold(downscaled_image, 250, 255, cv2.THRESH_BINARY)
    # _, downscaled_bw = cv2.threshold(downscaled_image, 245, 255, cv2.THRESH_BINARY)

    return downscaled_bw, bw_image.shape


def main():
    ## parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Create a navigation map based on a drawing with desired dimensions."
    )
    parser.add_argument(
        "--dimension_percentage",
        "-dp",
        type=float,
        default=5.0,
        help="Percentage to scale the image. Defaults to 5%%.",
    )
    parser.add_argument(
        "--drawing",
        type=str,
        default="warehouse",
        help="Drawing of the desired map. Defaults to 'warehouse.jpg'.",
    )
    args = parser.parse_args()

    ## define file paths
    base_dir = Path("ML_navigation/map_creation")
    input_image_path = base_dir / "original_files" / f"{args.drawing}.jpg"
    downscaled_output_path = base_dir / f"d_bw_{args.drawing}.jpg"

    ## convert to black and white
    try:
        bw_image = convert_to_black_white(str(input_image_path))
    except Exception as e:
        print(f"Error converting image to black and white: {e}")
        return

    ## downscale the black and white image while preserving black pixels
    try:
        downscaled_bw, original_shape = downscale_preserve_black(bw_image, args.dimension_percentage)
    except Exception as e:
        print(f"Error downscaling image: {e}")
        return

    ## save the final black-and-white image
    cv2.imwrite(str(downscaled_output_path), downscaled_bw)

    ## convert to NumPy array (0 for white, 1 for black)
    bw_array = 1 - (downscaled_bw / 255)

    ## print original and converted dimensions
    message = (
        "✅ The image has been successfully resized.\n"
        f"- Original dimensions: {original_shape[1]} (width) x {original_shape[0]} (height)\n"
        f"- Scaling factor: {args.dimension_percentage}%\n"
        f"- New dimensions: {bw_array.shape[1]} (width) x {bw_array.shape[0]} (height)"
    )
    print(message)

    ## save the NumPy array
    np.save(base_dir / "bw_array.npy", bw_array)


if __name__ == "__main__":
    main()