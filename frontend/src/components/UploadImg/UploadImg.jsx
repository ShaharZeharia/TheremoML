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

    const storageRef = firebase
      .storage()
      .ref(`uploads/${user.uid}/${file.name}`);

    try {
      await storageRef.put(file);
      const downloadURL = await storageRef.getDownloadURL();
      setUploadMessage("Upload successful!");
      navigate("/report", {
        state: { imageUrl: downloadURL },
      });
    } catch (error) {
      console.error("Upload error:", error);
      setUploadMessage("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  // const handleUpload = async () => {
  //   if (!file) {
  //     setUploadMessage("Please select a file first.");
  //     return;
  //   }

  //   if (file.size > 8 * 1024 * 1024) {
  //     setUploadMessage("File size must be less than 8MB.");
  //     return;
  //   }

  //   setIsUploading(true);

  //   setTimeout(() => {
  //     setIsUploading(false);
  //     setUploadMessage("Simulated upload complete!");
  //   }, 20000);
  // };

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
