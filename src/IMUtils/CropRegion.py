import cv2
import numpy as np
import sys
import os

def crop_white_board(image:np.ndarray|str, output_path=None, display_result=False):
    # Read the image
    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        raise ValueError("Could not load the image")

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to segment the white board from the background
    # Since the board is lighter, we invert the threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours of the white board
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image")

    # Find the largest contour (assuming the board is the largest light area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # More robust correction: Check the edges of the bounding rectangle for the board
    # Use the grayscale image to check pixel intensity near the edges
    padding_factor = 0.05  # 5% of width/height as a dynamic padding
    dynamic_padding_x = int(w * padding_factor)
    dynamic_padding_y = int(h * padding_factor)

    # Ensure the dynamic padding doesn’t exceed image boundaries
    x = max(0, x - dynamic_padding_x)
    y = max(0, y - dynamic_padding_y)
    w = min(image.shape[1] - x, w + 2 * dynamic_padding_x)
    h = min(image.shape[0] - y, h + 2 * dynamic_padding_y)

    # Additional check: Ensure we capture the full board by verifying edge intensity
    # Check if the edges of the cropped region still contain light pixels (board)
    cropped_gray = gray[y:y + h, x:x + w]
    _, edge_thresh = cv2.threshold(cropped_gray, 200, 255, cv2.THRESH_BINARY)
    edge_pixels = cv2.countNonZero(edge_thresh)

    # If there are still light pixels near the edges, expand further (optional safeguard)
    if edge_pixels > 0:
        additional_padding = 10  # Add a small fixed padding if light pixels are detected
        x = max(0, x - additional_padding)
        y = max(0, y - additional_padding)
        w = min(image.shape[1] - x, w + 2 * additional_padding)
        h = min(image.shape[0] - y, h + 2 * additional_padding)

    # Crop the image with the adjusted bounding rectangle
    cropped_image = image[y:y + h, x:x + w]

    if display_result:
        # Optionally, display the original and cropped images
        cv2.imshow('Original Image', image)
        cv2.imshow('Cropped White Board', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the cropped image temporarily
    if not output_path is None:
        cv2.imwrite(output_path, cropped_image)

    return cropped_image, (x, y, w, h)


def correct_perspective(cropped_image, original_bbox=None):
    # Convert cropped image to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection for more robust edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Apply morphological closing to fill gaps (e.g., from grid lines or stains)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours of the board in the cropped image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the cropped image")

    # Find the largest contour (assuming the board is the largest light area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour for the full board extent
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Approximate the contour to a polygon (e.g., a quadrilateral for the board)
    arc_length = cv2.arcLength(largest_contour, True)
    epsilon = 0.03 * arc_length  # Adjusted epsilon for better corner detection
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) < 4:
        raise ValueError("Could not detect enough corners for a quadrilateral shape for the board")

    # Ensure we have exactly 4 points (take the first 4 if more are detected)
    corners = approx[:4].reshape(-1, 2)

    # Sort corners to ensure correct ordering (top-left, top-right, bottom-left, bottom-right)
    # Calculate the centroid to help with ordering
    centroid = np.mean(corners, axis=0)

    # Sort by angle relative to centroid to ensure correct ordering
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    ordered_corners = corners[sorted_indices]

    # Reorder to ensure top-left, top-right, bottom-right, bottom-left
    # Find the topmost and bottommost points (by y-coordinate)
    top_points = ordered_corners[ordered_corners[:, 1].argsort()[:2]]
    bottom_points = ordered_corners[ordered_corners[:, 1].argsort()[2:]]

    # Sort top and bottom points by x-coordinate
    top_left = top_points[top_points[:, 0].argmin()]
    top_right = top_points[top_points[:, 0].argmax()]
    bottom_left = bottom_points[bottom_points[:, 0].argmin()]
    bottom_right = bottom_points[bottom_points[:, 0].argmax()]

    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    # Get the dimensions of the cropped image to match the size closely
    height, width = cropped_image.shape[:2]

    # Define the desired output rectangle (perfect rectangle with the same size as the cropped image, or slightly larger)
    # Add a small padding to ensure we capture the full board
    padding = 10  # Small padding to avoid cropping edges
    desired_width = width + 2 * padding  # Slightly larger than cropped width
    desired_height = height + 2 * padding  # Slightly larger than cropped height
    dst_points = np.array([
        [0, 0],  # Top-left
        [desired_width - 1, 0],  # Top-right
        [desired_width - 1, desired_height - 1],  # Bottom-right
        [0, desired_height - 1]  # Bottom-left
    ], dtype=np.float32)

    # Compute the perspective transform
    perspective_transform = cv2.getPerspectiveTransform(ordered_corners, dst_points)

    # Apply the perspective transform to correct the board
    corrected_image = cv2.warpPerspective(cropped_image, perspective_transform, (desired_width, desired_height))

    return corrected_image

def process_board(image_path):
    # Step 1: Crop the white board region
    cropped_image, bbox = crop_white_board(image_path, 'cropped_temp.jpg')

    # Step 2: Apply perspective correction to the cropped image
    corrected_image = correct_perspective(cropped_image)

    # Optionally, display the original, cropped, and corrected images
    original_image = cv2.imread(image_path)
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Cropped White Board', cropped_image)
    cv2.imshow('Corrected White Board', corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the corrected image
    cv2.imwrite('corrected_board.jpg', corrected_image)

    # Clean up temporary file
    # if os.path.exists('cropped_temp.jpg'):
    #    os.remove('cropped_temp.jpg')

    return corrected_image


# Example usage with command-line argument or user input
if __name__ == "__main__":
    try:
        # Check if filename is provided as a command-line argument
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            # If no argument, prompt user for input
            image_path = input("Please enter the path to your image file: ")

        process_board(image_path)
    except Exception as e:
        print(f"Error: {e}")
