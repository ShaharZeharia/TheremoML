import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import firebase from "../../firebase-config";
import "./Report.css";

function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { imageUrl } = location.state || {};
  const [processedImage, setProcessedImage] = useState(null);
  const [diagnosis, setDiagnosis] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!imageUrl) {
      navigate("/upload");
      return;
    }

    const fetchReport = async () => {
      //     CLOUD FUNCTIONS
      //     try {
      //       const analyzeAndSaveReport = firebase
      //         .app()
      //         .functions("your-region") // e.g., "us-central1"
      //         .httpsCallable("analyzeAndSaveReport");

      //       const result = await analyzeAndSaveReport({ imageUrl });

      //       setProcessedImage(result.data.processedImageUrl || imageUrl);
      //       setDiagnosis(result.data.diagnosis);
      //     } catch (err) {
      //       console.error(err);
      //       setError("There was an error processing the image.");
      //     } finally {
      //       setLoading(false);
      //     }
      //   };
      //    STRAIGHT TO HUGGING FACE/SERVER
      //   try {
      //     const response = await fetch("https://your-server.com/api/analyze", {
      //       method: "POST",
      //       headers: { "Content-Type": "application/json" },
      //       body: JSON.stringify({ imageUrl }),
      //     });

      //     if (!response.ok) throw new Error("Failed to fetch diagnosis.");

      //     const data = await response.json();
      //     setProcessedImage(data.processedImageUrl);
      //     setDiagnosis(data.diagnosis);
      //   } catch (err) {
      //     console.error(err);
      //     setError("There was an error processing the image.");
      //   } finally {
      //     setLoading(false);
      //   }
      // };
      try {
        // Simulate network delay
        await new Promise((res) => setTimeout(res, 10000));

        // Simulate response from server
        const mockData = {
          processedImageUrl: imageUrl, // You can keep it the same for testing
          diagnosis: "This is a mock diagnosis result for testing purposes.",
        };

        setProcessedImage(mockData.processedImageUrl);
        setDiagnosis(mockData.diagnosis);
      } catch (err) {
        console.error(err);
        setError("There was an error processing the image.");
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [imageUrl, navigate]);

  if (loading) {
    return (
      <div className="fullscreen-loader">
        <div className="spinner" />
        <p>Analyzing image, please wait...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="report-container">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="report-container">
      <h2>Diagnosis Report</h2>
      <img
        src={processedImage}
        alt="Processed result"
        className="report-image"
      />
      <div className="diagnosis-text">{diagnosis}</div>
    </div>
  );
}

export default ReportPage;
