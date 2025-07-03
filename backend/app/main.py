import firebase_admin.storage
from flask import Flask, request, jsonify
import requests
import threading
import logging
from datetime import datetime
import os
from PIL import Image
import firebase_admin
from firebase_admin import credentials
from dotenv import load_dotenv

from preprocess import preprocess_flir_image
from models import predict_joint_inflammation
from utils import annotate_inflammation_predictions, upload_image_to_firebase

load_dotenv()
cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not cred_path or not os.path.exists(cred_path):
    raise ValueError(f"Missing or invalid GOOGLE_APPLICATION_CREDENTIALS environment variable:{cred_path}")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'thermoml.firebasestorage.app'
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

@app.route('/api', methods=['POST'])
def analyze_image():
    try:
        data = request.get_json()
        required_fields = ['imageUrl', 'callbackUrl', 'reportId', 'uid', 'requestToken']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        image_url = data['imageUrl']
        callback_url = data['callbackUrl']
        report_id = data['reportId']
        uid = data['uid']
        request_token = data['requestToken']
    
        logger.info(f"Received analysis request for reportId: {report_id}")

        thread = threading.Thread(
            target=process_image_async,
            args=(image_url, callback_url, report_id, uid, request_token)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "status": "accepted",
            "message": "Analysis started",
            "reportId": report_id,
            "estimatedTime": "2-5 minutes"
        }), 202

    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        return jsonify({"error": "Failed to process request"}), 500

def process_image_async(image_url, callback_url, report_id, uid, request_token):
    try:
        logger.info(f"Starting async processing for report {report_id}")

        # --- Download image ---
        response = requests.get(image_url)
        response.raise_for_status()

        img_path = f"temp_{report_id}.jpg"
        with open(img_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Saved image to {img_path}")

        # --- Preprocess FLIR image ---
        image_4ch, landmarks, shifted_thermal = preprocess_flir_image(img_path)
        logger.info("Preprocessing completed")

        # --- Run prediction ---
        predictions = predict_joint_inflammation(image_4ch, landmarks, MODEL_DIR)
        logger.info("Prediction completed")

        diagnosis_text, image = annotate_inflammation_predictions(shifted_thermal, landmarks, predictions)

        processed_image_url = upload_image_to_firebase(uid, image, img_path)

        result = {
            "diagnosis": diagnosis_text,
            "processedImageUrl": processed_image_url,
            "metadata": {},
        }

        send_callback(callback_url, report_id, uid, result, request_token=request_token)

        # Clean up temp file
        os.remove(img_path)

    except Exception as e:
        logger.error(f"Async processing failed for report {report_id}: {e}")
        send_callback(callback_url, report_id, uid, None, error=str(e))

def send_callback(callback_url, report_id, uid, result=None, error=None, request_token=None):
    try:
        payload = {
            "reportId": report_id,
            "uid": uid,
            "requestToken": request_token,
            "callbackSecret": os.environ.get("CALLBACK_SECRET"),
        }
        if error:
            payload["error"] = error
        else:
            payload.update({
                "diagnosis": result["diagnosis"],
                "processedImageUrl": result["processedImageUrl"],
                "metadata": result.get("metadata"),
            })

        response = requests.post(callback_url, json=payload, timeout=30)
        if response.status_code == 200:
            logger.info(f"Callback sent successfully for report {report_id}")
        else:
            logger.error(f"Callback failed: {response.status_code} {response.text}")

    except Exception as e:
        logger.error(f"Failed to send callback: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
