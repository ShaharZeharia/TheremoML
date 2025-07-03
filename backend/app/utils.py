import cv2
import numpy as np
import io
from firebase_admin import storage

def annotate_inflammation_predictions(thermal_image, landmarks, probabilities, threshold=0.5):
    """
    Draws large hollow circles around inflamed joints on the thermal image and gives report.

    Args:
        thermal_image (np.ndarray): Original thermal image (H, W), dtype=np.uint8 or float32
        landmarks (list or np.ndarray): List of (x, y) coordinates for 32 joints
        probabilities (list or np.ndarray): List of 32 probabilities
        threshold (float): Inflammation threshold (default: 0.5)

    Returns:
        np.ndarray: Annotated RGB image
    """
    # Convert to 3-channel RGB if needed
    # if len(thermal_image.shape) == 2:
    #     # image_rgb = cv2.cvtColor(thermal_image, cv2.COLOR_GRAY2BGR)
    # else:
    #     image_rgb = thermal_image.copy()

    # Draw hollow red circles for inflamed joints and report back on any inflammations found
    inflammation_report = []
    for i, prob in enumerate(probabilities):
        if prob > threshold:
            x, y = map(int, landmarks[i])
            cv2.circle(thermal_image, (x, y), radius=15, color=(0, 0, 255), thickness=2)  # Hollow red circle
            inflammation_report.append({
                "joint_index": i,
                "position": {"x": x, "y": y},
                "probability": round(prob * 100, 2)  # percentage
            })

    if inflammation_report:
        parts = [f"Joint {rep['joint_index']}" for rep in inflammation_report]
        summary_text = "Inflammation detected at joints: " + ", ".join(parts)
    else:
        summary_text = "All joints appear normal."

    # Ensure output is RGB for correct display outside OpenCV
    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
    return summary_text, thermal_image

def upload_image_to_firebase(uid, image, dest_filename):
    """
    Uploads a PIL image or numpy image directly to Firebase Storage without saving locally.
    
    Args:
        image (PIL.Image.Image or np.ndarray): The image to upload.
        dest_filename (str): The filename to use in Firebase Storage.
    
    Returns:
        str: The public URL of the uploaded image.
    """
    # Convert to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        from PIL import Image
        image = Image.fromarray(image)

    # Save image to an in-memory buffer
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    # Upload to Firebase
    bucket = storage.bucket() 
    blob = bucket.blob(f"{uid}/processed_images/{dest_filename}")
    blob.upload_from_file(image_bytes, content_type='image/jpeg')
    
    # Make it public (optional)
    blob.make_public()
    print(f"Uploaded to Firebase. Public URL: {blob.public_url}")
    return blob.public_url
