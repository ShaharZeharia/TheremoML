import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import firebase from "../../firebase-config";
import { getFunctions, httpsCallable } from "firebase/functions";
import { getAuth } from "firebase/auth";
import "./UploadImg.css";

function UploadImg() {
  const [file, setFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [previewURL, setPreviewURL] = useState(null);
  const navigate = useNavigate();
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    const auth = firebase.auth();
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (!user) {
        navigate("/login", { replace: true });
      }
    });

    return () => unsubscribe();
  }, [navigate]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (
      selectedFile &&
      (selectedFile.type === "image/jpeg" || selectedFile.type === "image/jpg")
    ) {
      if (selectedFile.size > 8 * 1024 * 1024) {
        setUploadMessage("File size must be less than 8MB.");
        return;
      }
      setFile(selectedFile);
      setPreviewURL(URL.createObjectURL(selectedFile));
      setUploadMessage("");
    } else {
      setFile(null);
      setPreviewURL(null);
      setUploadMessage("Please select a valid JPG file.");
    }
  };

  const handleUpload = async () => {
    const auth = firebase.auth();
    const user = auth.currentUser;

    if (!file) {
      setUploadMessage("Please select a file first.");
      return;
    }

    if (file.size > 8 * 1024 * 1024) {
      setUploadMessage("File size must be less than 8MB.");
      return;
    }

    setIsUploading(true);

    if (!user) {
      setUploadMessage("User is not authenticated. Please log in again.");
      setIsUploading(false);
      return;
    }

    const storageRef = firebase
      .storage()
      .ref(`uploads/${user.uid}/${file.name}`);

    const functions = getFunctions(firebase.app(), "us-central1");
    const initiateAnalysis = httpsCallable(functions, "initiateAnalysis");

    try {
      // Upload image to Firebase Storage
      await storageRef.put(file);
      const downloadURL = await storageRef.getDownloadURL();

      console.log("Sending to initiateAnalysis:", {
        imageUrl: downloadURL,
        //uid: user.uid,
      });

      console.log("currentUser in UploadImg:", firebase.auth().currentUser);

      const result = await initiateAnalysis({
        imageUrl: downloadURL,
      });

      setUploadMessage("Upload successful! Analysis initiated...");

      // Navigate to report page with both imageUrl and reportId
      navigate("/report", {
        state: {
          imageUrl: downloadURL,
          reportId: result.data.reportId,
        },
      });
    } catch (error) {
      console.error("Upload or analysis initiation error:", error);
      setUploadMessage(
        "Upload failed or couldn't initiate analysis. Please try again."
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-page">
      <p className="upload-slogan">Scan. Detect. Protect.</p>
      <h2>Upload File to Firebase</h2>
      <input type="file" onChange={handleFileChange} />
      {previewURL && (
        <div className="preview-container">
          <p>Preview:</p>
          <img src={previewURL} alt="Preview" className="preview-image" />
        </div>
      )}
      <button
        className={`upload-button ${isUploading ? "loading" : ""}`}
        onClick={handleUpload}
        disabled={isUploading}
      >
        {isUploading && <div className="spinner-inline" />}
        {isUploading ? "Uploading..." : "Upload"}
      </button>
      {uploadMessage && <p>{uploadMessage}</p>}
    </div>
  );
}

export default UploadImg;
