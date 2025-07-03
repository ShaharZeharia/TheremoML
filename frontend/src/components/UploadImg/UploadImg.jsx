import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import firebase from "../../firebase-config";
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
    if (!file) {
      setUploadMessage("Please select a file first.");
      return;
    }

    if (file.size > 8 * 1024 * 1024) {
      setUploadMessage("File size must be less than 8MB.");
      return;
    }

    setIsUploading(true);

    try {
      // Ensure user is authenticated
      const auth = firebase.auth();
      const user = auth.currentUser;

      if (!user) {
        setUploadMessage("Please log in first.");
        setIsUploading(false);
        return;
      }

      console.log("User is authenticated:", user.uid);

      const storageRef = firebase
        .storage()
        .ref(`uploads/${user.uid}/${file.name}`);

      // Upload image to Firebase Storage
      await storageRef.put(file);
      const downloadURL = await storageRef.getDownloadURL();

      console.log("Image uploaded successfully. URL:", downloadURL);

      // Get the user's ID token for authentication
      const idToken = await user.getIdToken();

      // Make POST request to your Cloud Function
      const functionUrl = `https://us-central1-thermoml.cloudfunctions.net/initiateAnalysis`;

      const response = await fetch(functionUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${idToken}`,
        },
        body: JSON.stringify({
          imageUrl: downloadURL,
          uid: user.uid,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `HTTP error! status: ${response.status}`
        );
      }

      const result = await response.json();

      console.log("Analysis initiated successfully:", result);

      setUploadMessage("Upload successful! Analysis initiated...");

      // Navigate to report page
      navigate("/report", {
        state: {
          imageUrl: downloadURL,
          reportId: result.reportId,
        },
      });
    } catch (error) {
      console.error("Upload or analysis initiation error:", error);

      if (
        error.message.includes("401") ||
        error.message.includes("unauthenticated")
      ) {
        setUploadMessage("Authentication failed. Please log in again.");
        navigate("/login");
      } else {
        setUploadMessage(
          `Error: ${error.message || "Upload failed. Please try again."}`
        );
      }
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
