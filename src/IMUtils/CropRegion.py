import cv2
import numpy as np

def crop_white_board(image:np.ndarray | str):
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

    return cropped_image, (x, y, w, h)


def correct_perspective(cropped_image: np.ndarray, return_matrix: bool = False):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 100, 200)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the cropped image")
    largest = max(contours, key=cv2.contourArea)
    eps = 0.03 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    if len(approx) < 4:
        raise ValueError("Could not detect enough corners")
    corners = approx[:4].reshape(-1, 2).astype(np.float32)

    # Order corners TL, TR, BR, BL robustly
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:,1]-c[1], corners[:,0]-c[0])
    ordered = corners[np.argsort(angles)]
    top = ordered[ordered[:,1].argsort()[:2]]
    bottom = ordered[ordered[:,1].argsort()[2:]]
    tl = top[top[:,0].argmin()]; tr = top[top[:,0].argmax()]
    bl = bottom[bottom[:,0].argmin()]; br = bottom[bottom[:,0].argmax()]
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    h, w = cropped_image.shape[:2]
    pad = 10
    dst = np.array([[0,0],
                    [w+2*pad-1, 0],
                    [w+2*pad-1, h+2*pad-1],
                    [0, h+2*pad-1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    out_w, out_h = int(w + 2*pad), int(h + 2*pad)
    corrected = cv2.warpPerspective(cropped_image, H, (out_w, out_h))
    return (corrected, H, (out_w, out_h)) if return_matrix else corrected

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
    import argparse

    # Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='This is a white board processor. Can create crops and skew corrections.')
    parser.add_argument('-img', dest='img', type=str, default='.',
                        help='Path to image to process.', required=True)
    parser.add_argument('-vi', action='store_true', default=False, dest='visualize',
                        help='Visualize results.')
    parser.add_argument('-v', action='count', default=0, dest='verbose',
                        help='Amount of verbosity (more \'v\'s means more verbose).')
    config, unparsed = parser.parse_known_args()

    process_board(config.img)
