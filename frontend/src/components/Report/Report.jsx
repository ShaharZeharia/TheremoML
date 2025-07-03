import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import firebase from "../../firebase-config";
import "./Report.css";

function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { imageUrl, reportId } = location.state || {};
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState(
    "Initializing analysis..."
  );

  useEffect(() => {
    if (!imageUrl || !reportId) {
      navigate("/home");
      return;
    }

    const auth = firebase.auth();
    const user = auth.currentUser;

    if (!user) {
      navigate("/login");
      return;
    }

    // Set up real-time listener for report status
    const reportRef = firebase
      .database()
      .ref(`users/${user.uid}/reports/${reportId}`);

    const unsubscribe = reportRef.on(
      "value",
      (snapshot) => {
        const data = snapshot.val();

        if (data) {
          setReportData(data);

          switch (data.status) {
            case "processing":
              setStatusMessage("Analyzing image, please wait...");
              setLoading(true);
              break;
            case "completed":
              setStatusMessage("Analysis complete!");
              setLoading(false);
              break;
            case "failed":
              setError(data.error || "Analysis failed. Please try again.");
              setLoading(false);
              break;
            default:
              setStatusMessage("Processing...");
          }
        }
      },
      (error) => {
        console.error("Database error:", error);
        setError("Failed to track analysis progress.");
        setLoading(false);
      }
    );

    // Cleanup listener on unmount
    return () => {
      reportRef.off("value", unsubscribe);
    };
  }, [imageUrl, reportId, navigate]);

  if (loading) {
    return (
      <div className="fullscreen-loader">
        <div className="spinner" />
        <p>{statusMessage}</p>
        <div className="progress-info">
          <p>This may take a few minutes...</p>
          {reportData?.createdAt && (
            <p>
              Started: {new Date(reportData.createdAt).toLocaleTimeString()}
            </p>
          )}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="report-container error-container">
        <h2>Analysis Error</h2>
        <p className="error-message">{error}</p>
        <div className="error-actions">
          <button onClick={() => navigate("/home")} className="upload-new-button">
            Upload New Image
          </button>
        </div>
      </div>
    );
  }

  if (!reportData || reportData.status !== "completed") {
    return (
      <div className="report-container">
        <p>Report data not available.</p>
        <button onClick={() => navigate("/home")} className="back-button">
          Back to Upload
        </button>
      </div>
    );
  }

  return (
    <div className="report-container">
      <div className="report-header">
        <h2>Diagnosis Report</h2>
        <div className="report-meta">
          <p>
            Completed:{" "}
            {new Date(
              reportData.updatedAt || reportData.createdAt
            ).toLocaleString()}
          </p>
          <span className="status-badge completed">✓ Analysis Complete</span>
        </div>
      </div>

      <div className="report-content">
        <div className="image-section">
          <h3>Processed Image</h3>
          <img
            src={reportData.processedImageUrl || reportData.imageUrl}
            alt="Processed result"
            className="report-image"
          />
        </div>

        <div className="diagnosis-section">
          <h3>Diagnosis</h3>
          <div className="diagnosis-text">
            {reportData.diagnosis || "No diagnosis available"}
          </div>
        </div>
      </div>

      <div className="report-actions">
        <button
          onClick={() => navigate("/home")}
          className="upload-new-button"
        >
          Analyze Another Image
        </button>
        <button onClick={() => window.print()} className="print-button">
          Print Report
        </button>
      </div>
    </div>
  );
}

export default ReportPage;
