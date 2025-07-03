import os
import cv2
import numpy as np
import subprocess
from flirimageextractor import FlirImageExtractor
from PIL import Image
from io import BytesIO
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

def getFlirThermalAndOptical(file_path: str):
    """
    Given a FLIR JPG image path, returns:
    - thermal_color: np.ndarray of inferno-colored thermal image
    - optical_image: PIL.Image of the embedded optical image
    """
    # --- Extract thermal image ---
    flir = FlirImageExtractor()
    flir.process_image(file_path, RGB=False)
    
    thermal_data = flir.get_thermal_np()
    thermal_gray = cv2.normalize(thermal_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thermal_color = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_INFERNO)

    # --- Extract embedded optical image ---
    result = subprocess.run(
        ["exiftool", "-b", "-EmbeddedImage", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    optical_image = Image.open(BytesIO(result.stdout))

    return thermal_color, optical_image

def process_hand_segmentation(image_input, predictor, desired_size=(320, 240), min_area=1000):
    image, base_name = load_and_prepare_image(image_input, desired_size)
    image = enhance_contrast(image)
    hand_masks = segment_hands_with_sam(image, predictor, desired_size, min_area, base_name)
    
    if hand_masks is None:
        return None

    segmented_hands = [feather_edges(image, mask, feather_amount=5) for mask in hand_masks]
    return tuple(segmented_hands)  # (left_hand, right_hand)

def load_and_prepare_image(image_input, desired_size):
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_ANYCOLOR)
        if image is None:
            raise ValueError(f"Failed to read image from path: {image_input}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = os.path.splitext(os.path.basename(image_input))[0]
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
        base_name = "image"
    else:
        raise ValueError("image_input must be a file path or numpy.ndarray")

    if image.shape[:2][::-1] != desired_size:
        image = cv2.resize(image, desired_size, interpolation=cv2.INTER_AREA)

    return image, base_name

# === Enhance contrast ===
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

# === Improved feathering ===
def feather_edges(image, mask, feather_amount=5):
    mask = mask.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    edge = mask_dilated - mask
    edge_blur = cv2.GaussianBlur(edge, (0, 0), feather_amount)
    soft_mask = np.clip(mask + edge_blur, 0, 1)
    alpha = np.stack([soft_mask] * 3, axis=-1)
    blended = (image * alpha).astype(np.uint8)
    return blended

def segment_hands_with_sam(image, predictor, desired_size, min_area, base_name="image"):
    predictor.set_image(image)
    h, w = desired_size[1], desired_size[0]

    input_point = np.array([
        [int(w * 0.25), int(h * 0.6)],
        [int(w * 0.75), int(h * 0.6)],
        [int(w * 0.25), int(h * 0.4)],
        [int(w * 0.75), int(h * 0.4)],
        [int(w * 0.5), int(h * 0.85)]
    ])
    input_label = np.array([1, 1, 1, 1, 0])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = masks[0].astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask)

    hands = []
    for label in range(1, num_labels):
        hand_mask = (labels == label).astype(np.uint8)
        area = cv2.countNonZero(hand_mask)
        if area < min_area:
            continue
        M = cv2.moments(hand_mask)
        if M["m00"] == 0:
            continue
        center_x = int(M["m10"] / M["m00"])
        hands.append((center_x, hand_mask))

    # Handle only 1 hand case
    if len(hands) == 1:
        hands = try_split_one_hand(hands[0][1], w, min_area)
        if hands is None:
            print(f"[{base_name}] Failed to split 1 hand into 2.")
            return None

    if len(hands) != 2:
        print(f"[{base_name}] Skipping: did not find exactly 2 hands.")
        return None

    # Sort left-to-right
    hands.sort(key=lambda h: h[0])
    return [hand[1] for hand in hands]

def try_split_one_hand(big_mask, width, min_area):
    mid_x = width // 2
    left_mask = np.zeros_like(big_mask)
    right_mask = np.zeros_like(big_mask)
    left_mask[:, :mid_x] = big_mask[:, :mid_x]
    right_mask[:, mid_x:] = big_mask[:, mid_x:]

    if cv2.countNonZero(left_mask) > min_area and cv2.countNonZero(right_mask) > min_area:
        return [(int(width * 0.25), left_mask), (int(width * 0.75), right_mask)]
    return None

def load_and_preprocess(image_input, desired_size):
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise ValueError(f"Cannot load image: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError("image_input must be a path or numpy array")
    if img.shape[:2][::-1] != desired_size:
        img = cv2.resize(img, desired_size, interpolation=cv2.INTER_AREA)
    return img

def filter_and_score_masks(masks, image, landmarker, num_hands, thresholds):
    width_upper, width_lower, area_lower, area_higher = thresholds
    h, w = image.shape[:2]
    area_img = h * w
    results = []
    for idx, m in enumerate(masks):
        mask = m["segmentation"].astype(np.uint8)
        area = cv2.countNonZero(mask)
        area_ratio = area / area_img
        if not (area_lower <= area_ratio <= area_higher):
            continue

        feathered = feather_edges(image, mask)
        count = count_hand_landmarks(landmarker, feathered)
        if count != (42 if num_hands == 2 else 21):
            continue

        dist = average_min_distance_to_corners(mask)
        results.append((idx, mask, dist))
    return results

def select_masks(results, thresholds, image, landmarker_two, two_separate):
    if two_separate:
        scored = sorted(results, key=lambda x: x[2], reverse=True)[:2]
    else:
        scored = [max(results, key=lambda x: x[2])]
    masks = [res[1] for res in scored]

    if len(masks) == 2:
        combined = cv2.add(feather_edges(image, masks[0]), feather_edges(image, masks[1]))
        count = count_hand_landmarks(landmarker_two, combined)
        return [combined] if count == 42 else []
    return [feather_edges(image, masks[0])]

def segment_optical_hands(image_input, mask_generator, landmarker_one, landmarker_two, desired_size=(320,240)):
    img = load_and_preprocess(image_input, desired_size)
    masks = mask_generator.generate(img)

    results = filter_and_score_masks(masks, img, landmarker_two, 2, (0.9, 0.2, 0.06, 0.82))
    two_sep = False
    if not results:
        results = filter_and_score_masks(masks, img, landmarker_one, 1, (0.95, 0.15, 0.03, 0.9))
        two_sep = True

    if not results:
        return []  # No valid hands found

    out_masks = select_masks(results, None, img, landmarker_two, two_sep)
    return out_masks  # List of segmented images (RGB np.ndarrays)

def count_hand_landmarks(landmarker, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = landmarker.detect(mp_image)
    return sum(len(hand) for hand in result.hand_landmarks) if result.hand_landmarks else 0

def average_min_distance_to_corners(mask):
    h, w = mask.shape

    corners = np.array([
        [0, 0],              # top-left
        [0, w - 1],          # top-right
        [h // 2, 0],         # middle-left
        [h // 2, w - 1],     # middle-right
        [0, w // 2],         # top-middle
        [0, w // 2]          # count center twice
    ])

    ys, xs = np.nonzero(mask)
    points = np.stack([ys, xs], axis=1)

    if len(points) == 0:
        return np.inf  # empty mask

    # Base distance score
    min_dists = [np.min(np.linalg.norm(points - corner, axis=1)) for corner in corners]
    score = np.sum(min_dists)

    # === Center Proximity Score ===
    mask_center = np.mean(points, axis=0)  # (y, x)
    center = np.array([h // 2, w // 2])
    avg_center_dist = np.linalg.norm(mask_center - center)

    # Max possible distance = image diagonal (for normalization)
    max_dist = np.linalg.norm([h / 2, w / 2])
    center_score = (1 + avg_center_dist / max_dist) * score / 2
    score += center_score

    # === Punish if mask touches any edge ===
    if np.any(mask[0, :]):            # top edge
        score *= 0.5
    if np.any(mask[0:h//2, 0]):            # left edge
        score *= 0.5
    if np.any(mask[0:h//2, w - 1]):        # right edge
        score *= 0.5

    return abs(score)

def extract_hand_from_thermal(thermal_image, optical_segmented_image):
    """Extract hand regions from thermal image using optical segmentation mask."""
    # Create a binary mask where the optical image is not black
    # (any channel > 0 implies it's a hand pixel)
    hand_mask = np.any(optical_segmented_image != 0, axis=-1).astype(np.uint8)  # shape: (H, W)

    # Apply mask to thermal image
    masked_thermal = cv2.bitwise_and(thermal_image, thermal_image, mask=hand_mask)

    return masked_thermal

def detect_and_draw_landmarks(image, landmarker):
    """
    Detect hand landmarks in an image and draw them on the image.

    Args:
        image (np.array): BGR image.
        landmarker: Initialized MediaPipe HandLandmarker.

    Returns:
        image_with_landmarks (np.array): RGB image with landmarks drawn.
        landmarks_list (list of tuples): List of (x, y) pixel coordinates for all detected landmarks.
        total_landmarks (int): Total number of landmarks detected.
    """
    # Convert BGR to RGB for MediaPipe processing
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image=image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = landmarker.detect(mp_image)

    landmarks_list = []
    image_with_landmarks = rgb_image.copy()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x_px = int(landmark.x * image.shape[1])
                y_px = int(landmark.y * image.shape[0])
                landmarks_list.append((x_px, y_px))
                cv2.circle(image_with_landmarks, (x_px, y_px), 3, (0, 255, 0), -1)

    # Convert back to BGR before returning to keep consistent with OpenCV usage
    image_with_landmarks = cv2.cvtColor(image_with_landmarks, cv2.COLOR_RGB2BGR)
    return image_with_landmarks, landmarks_list, len(landmarks_list)

def func_landmark(left_img=None, right_img=None, combined_img=None, 
                  has_left=False, has_right=False, has_both=False, landmarker=None):
    """
    Process images to detect hand landmarks according to provided flags.

    Returns landmarks list if valid, else None.
    """

    if has_left and has_right:
        left_marked, left_landmarks, left_count = detect_and_draw_landmarks(left_img, landmarker)
        right_marked, right_landmarks, right_count = detect_and_draw_landmarks(right_img, landmarker)

        combined = cv2.add(left_img, right_img)
        combined_marked, combined_landmarks, combined_count = detect_and_draw_landmarks(combined, landmarker)

        if left_count == 21 and right_count == 21 and combined_count == 42:
            return combined_landmarks
        elif combined_count != 42:
            print("Error: Combined landmarks count not equal 42")
            return None
        elif left_count == 42:
            return left_landmarks
        elif right_count == 42:
            return right_landmarks
        else:
            print(f"Skipped: incomplete landmarks (left: {left_count}, right: {right_count})")
            return None

    elif has_left and left_img is not None:
        marked, landmarks, count = detect_and_draw_landmarks(left_img, landmarker)
        if count in [21, 42]:
            return landmarks
        print(f"Skipped: invalid left-only landmarks count: {count}")
        return None

    elif has_right and right_img is not None:
        marked, landmarks, count = detect_and_draw_landmarks(right_img, landmarker)
        if count == 42:
            return landmarks
        print(f"Skipped: invalid right-only landmarks count: {count}")
        return None

    elif has_both and combined_img is not None:
        marked, landmarks, count = detect_and_draw_landmarks(combined_img, landmarker)
        if count == 42:
            return landmarks
        print(f"Skipped: invalid both-only landmarks count: {count}")
        return None

    return None

def resize_scale(optical_image, scale_factor):
    """Resize optical image based on scale factor."""
    h, w = optical_image.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    return cv2.resize(optical_image, (new_w, new_h))

def center_pad_image(smaller_img, target_shape):
    """Center pad smaller image to match target shape."""
    small_h, small_w = smaller_img.shape[:2]
    target_h, target_w = target_shape[:2]
    
    top = (target_h - small_h) // 2
    bottom = target_h - small_h - top
    left = (target_w - small_w) // 2
    right = target_w - small_w - left

    padded_img = cv2.copyMakeBorder(smaller_img, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT, value=0)
    return padded_img

def translate_image(image, x_shift=0, y_shift=0):
    """Translate image by x and y pixels. Positive y_shift moves down, negative moves up."""
    height, width = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated = cv2.warpAffine(image, M, (width, height), borderValue=(0,0,0))
    return translated

def register_thermal_to_optical(thermal_image, thermal_segment_left, thermal_segment_right, 
                               optical_segmented, one_landmarker, two_landmarker, scale=1.41):
    """
    Register thermal image to optical image and return landmarks, shifted thermal, and masked thermal.
    
    Args:
        thermal_image (np.array): Original thermal image
        thermal_segment_left (np.array): Left hand segmented thermal image
        thermal_segment_right (np.array): Right hand segmented thermal image
        optical_segmented (np.array): Segmented optical image
        landmarker: MediaPipe HandLandmarker instance
        scale (float): Scale factor for optical image (default: 1.41)
    
    Returns:
        tuple: (optical_landmarks, shifted_thermal, masked_thermal) or (None, None, None) if landmarks not found
    """
    
    # Apply scaling to optical image
    optical_scaled = resize_scale(optical_segmented, scale)
    
    # Resize thermal images to standard size if needed
    if thermal_image.shape[:2] != (240, 320):
        thermal_image = cv2.resize(thermal_image, (320, 240), interpolation=cv2.INTER_AREA)
        thermal_segment_left = cv2.resize(thermal_segment_left, (320, 240), interpolation=cv2.INTER_AREA)
        thermal_segment_right = cv2.resize(thermal_segment_right, (320, 240), interpolation=cv2.INTER_AREA)
    
    # Apply padding to thermal images to match optical scaled shape
    thermal_padded = center_pad_image(thermal_image, optical_scaled.shape)
    thermal_segment_padded_left = center_pad_image(thermal_segment_left, optical_scaled.shape)
    thermal_segment_padded_right = center_pad_image(thermal_segment_right, optical_scaled.shape)
    
    # Resize everything back to standard size
    thermal_padded = cv2.resize(thermal_padded, (320, 240), interpolation=cv2.INTER_AREA)
    thermal_segment_padded_left = cv2.resize(thermal_segment_padded_left, (320, 240), interpolation=cv2.INTER_AREA)
    thermal_segment_padded_right = cv2.resize(thermal_segment_padded_right, (320, 240), interpolation=cv2.INTER_AREA)
    optical_scaled = cv2.resize(optical_scaled, (320, 240), interpolation=cv2.INTER_AREA)
    
    # Detect landmarks
    thermal_landmarks_left = func_landmark(thermal_segment_padded_left, "", "", True, False, False, one_landmarker)
    thermal_landmarks_right = func_landmark("", thermal_segment_padded_right, "", False, True, False, one_landmarker)
    optical_landmarks = func_landmark("", "", optical_scaled, False, False, True, two_landmarker)
    
    # Check if we have required landmarks
    if ((thermal_landmarks_left is None and thermal_landmarks_right is None) or optical_landmarks is None):
        return None, None, None
    
    # Determine which optical points to use based on hand position
    optical_point_left = 9
    optical_point_right = 30
    if (optical_landmarks[9][0] > (optical_scaled.shape[1] // 2)):
        optical_point_left = optical_point_left + 21
        optical_point_right = 9
    
    # Calculate shift based on available thermal landmarks
    if thermal_landmarks_left is None:
        # No left landmarks, use the right
        x_shift = optical_landmarks[optical_point_right][0] - thermal_landmarks_right[9][0]
        y_shift = optical_landmarks[optical_point_right][1] - thermal_landmarks_right[9][1]
    else:
        # Use left landmarks
        x_shift = optical_landmarks[optical_point_left][0] - thermal_landmarks_left[9][0]
        y_shift = optical_landmarks[optical_point_left][1] - thermal_landmarks_left[9][1]
    
    # Apply translation to thermal image
    shifted_thermal = translate_image(thermal_padded, y_shift=y_shift, x_shift=x_shift)
    
    # Apply mask to get final result
    masked_thermal = extract_hand_from_thermal(shifted_thermal, optical_scaled)
    
    return optical_landmarks, shifted_thermal, masked_thermal

def create_hand_landmarker(model_path, num_hands=2):
    """
    Create and return a MediaPipe HandLandmarker instance.
    
    Args:
        model_path (str): Path to the hand_landmarker.task model file
        num_hands (int): Maximum number of hands to detect (default: 2)
    
    Returns:
        vision.HandLandmarker: Initialized landmarker instance
    """
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=num_hands
    )

    return vision.HandLandmarker.create_from_options(options)

def reorder_landmarks_if_needed(landmarks):
    """
    Ensures that landmarks of the left hand (with smaller x) come before the right hand.
    Assumes:
    - landmarks is a (42, 2) or (42, 3) array.
    - landmarks[0:21] are one hand, landmarks[21:42] are the other.
    """

    # Check if we have exactly 42 landmarks (2 hands)
    if landmarks.shape[0] != 42:
        return landmarks  # No change needed

    left_hand = landmarks[:21]
    right_hand = landmarks[21:]

    # Compare x-coordinates (column 0) of landmark 0 and 21
    if landmarks[0][0] > landmarks[21][0]:  # Left hand is actually second in the array
        landmarks = np.concatenate([right_hand, left_hand], axis=0)

    return landmarks

def remap_landmarks(landmarks):
    """
    Reorders landmarks according to a custom label mapping and removes irrelevant points.

    Args:
        landmarks (np.ndarray): Original array of shape (42, N) where N=2 or 3 (x, y, [z])

    Returns:
        np.ndarray: Remapped landmark array of shape (32, N)
    """
    # Irrelevant landmarks to remove
    irrelevant_indices = {4, 8, 12, 16, 20, 25, 29, 33, 37, 41}

    # Mapping: new_index -> old_index
    mapping = [
        21, 0, 22, 1, 23, 2, 24, 3,
        26, 5, 27, 6, 28, 7, 30, 9,
        31, 10, 32, 11, 34, 13, 35, 14,
        36, 15, 38, 17, 39, 18, 40, 19
    ]

    # Apply mapping and ignore irrelevant indices
    filtered_landmarks = np.array([landmarks[i] for i in mapping if i not in irrelevant_indices])

    return filtered_landmarks

def get_hand_centers(landmarks):
    centers = []
    centers.append(intersect(landmarks[1], landmarks[17], landmarks[12], landmarks[0]))
    centers.append(intersect(landmarks[22], landmarks[38], landmarks[33], landmarks[21]))
    return centers

def intersect(p1, p2, p3, p4):
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]

    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return None
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return int(x), int(y)

# Transform mask image into binary mask for second channel
# In actual use, using 1/8 of the max value in the image or lower is working well
def load_segmented_image_as_mask(path, threshold=128):
    # Load as grayscale (0–255)
    seg_img = Image.open(path).convert("L")
    seg_array = np.array(seg_img)

    # Convert to binary mask: hand = 1, background = 0
    binary_mask = (seg_array > threshold).astype(np.uint8)
    return binary_mask  # Shape: (H, W), values 0 or 1

def prepare_4_channel_image(thermal_image, segmentation_mask, hand_centers=None):
    if thermal_image.ndim == 3 and thermal_image.shape[2] == 3:
        pil_gray = Image.fromarray(thermal_image).convert("L")
        thermal_image = np.array(pil_gray).astype(np.float32)
    elif thermal_image.ndim == 2:
        thermal_image = thermal_image.astype(np.float32)
    else:
        raise ValueError("Expected RGB or 2D image")

    # Normalize
    thermal = (thermal_image - thermal_image.min()) / (thermal_image.max() - thermal_image.min() + 1e-6)
    thermal = torch.tensor(thermal, dtype=torch.float32)
    mask = torch.tensor(segmentation_mask, dtype=torch.float32)
    masked_thermal = thermal * mask
    H, W = thermal.shape

    if hand_centers is not None and len(hand_centers) > 0:
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        distances = [torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2) for cx, cy in hand_centers]
        distance_map = torch.min(torch.stack(distances, dim=0), dim=0).values
        distance_map = distance_map / (distance_map.max() + 1e-6)
    else:
        distance_map = torch.nn.functional.avg_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=11, stride=1, padding=5).squeeze()

    image_4ch = torch.stack([thermal, mask, masked_thermal, distance_map], dim=0)
    return image_4ch

def preprocess_flir_image(
    image_path,
    output_dir="outputs",
    scale=1.41,
):
    """
    Complete preprocessing pipeline for a FLIR image.
    
    Returns:
        image_4ch (torch.Tensor): 4-channel image tensor (4, H, W)
        optical_landmarks (list of (x, y)): Detected optical hand landmarks
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract thermal and optical images
    thermal, optical = getFlirThermalAndOptical(image_path)
    thermal_path = os.path.join(output_dir, "thermal.jpg")
    try:
        cv2.imwrite(thermal_path, thermal)
    except Exception as e:
        raise RuntimeError(f"Failed to write thermal image to {thermal_path}: {e}")
    optical_path = os.path.join(output_dir, "optical.jpg")
    cv2.imwrite(thermal_path, thermal)
    try:
        optical.save(optical_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save optical image to {optical_path}: {e}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # === Load SAM ===
    sam = sam_model_registry["vit_h"](
        checkpoint=os.path.join(MODEL_DIR, "sam_vit_h_4b8939.pth")
    ).to(device)
    sam_automatic = SamAutomaticMaskGenerator(sam)
    sam_predictor = SamPredictor(sam)

    # === Load MediaPipe model ===
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options_two_hand = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=
                                os.path.join(MODEL_DIR, "hand_landmarker.task")),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )

    landmarker_two_hand = vision.HandLandmarker.create_from_options(options_two_hand)

    options_one_hand = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=
                                os.path.join(MODEL_DIR, "hand_landmarker.task")),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    landmarker_one_hand = vision.HandLandmarker.create_from_options(options_one_hand)

    # Step 2: Segment thermal hands
    thermal_seg_result = process_hand_segmentation(thermal_path, sam_predictor)
    if thermal_seg_result is None:
        raise RuntimeError("Thermal hand segmentation failed.")
    thermal_left, thermal_right = thermal_seg_result

    # Step 3: Segment optical hands
    optical_segmented_list = segment_optical_hands(optical_path, sam_automatic, landmarker_one_hand, landmarker_two_hand)
    if not optical_segmented_list:
        raise RuntimeError("Optical hand segmentation failed.")
    optical_segmented = optical_segmented_list[0]

    # Step 4: Register and align
    optical_landmarks, shifted_thermal, masked_thermal = register_thermal_to_optical(
        thermal,
        thermal_left,
        thermal_right,
        optical_segmented,
        landmarker_one_hand,
        landmarker_two_hand,
        scale=scale
    )

    if optical_landmarks is None:
        raise RuntimeError("Registration failed or landmarks not found.")

    # Step 5: Get hand centers
    hand_centers = get_hand_centers(optical_landmarks)
    if not hand_centers or any(c is None for c in hand_centers):
        raise RuntimeError("Failed to compute hand centers.")
    
    # Step 6: Convert mask to binary
    temp_mask_path = os.path.join(output_dir, "masked_thermal.jpg")
    try:
        cv2.imwrite(temp_mask_path, masked_thermal)
    except Exception as e:
        raise RuntimeError(f"Failed to write masked thermal image to {temp_mask_path}: {e}")
    segmentation_mask = load_segmented_image_as_mask(temp_mask_path)
    
    # Step 7: Prepare 4-channel tensor
    image_4ch = prepare_4_channel_image(
        thermal_image=shifted_thermal,
        segmentation_mask=segmentation_mask,
        hand_centers=hand_centers
    )
    optical_landmarks = np.array(optical_landmarks)
    landmarks = reorder_landmarks_if_needed(optical_landmarks)
    landmarks = remap_landmarks(landmarks)

    for path in [thermal_path, optical_path, temp_mask_path]:
        try:
            os.remove(path)
        except OSError:
            pass

    return image_4ch, landmarks, shifted_thermal
