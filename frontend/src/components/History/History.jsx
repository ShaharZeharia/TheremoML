import React, { useEffect, useState, useRef, useCallback } from "react";
import firebase from "../../firebase-config";
import { useNavigate } from "react-router-dom";
import "./History.css";

function History() {
  const [reports, setReports] = useState([]);
  const [lastTimestamp, setLastTimestamp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const loaderRef = useRef(null);
  const observerRef = useRef(null);
  const PAGE_SIZE = 5;
  const navigate = useNavigate();

  const fetchReports = useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);

    const auth = firebase.auth();
    const user = auth.currentUser;
    if (!user) return;

    try {
      const dbRef = firebase.database().ref(`users/${user.uid}/reports`);
      let query = dbRef.orderByChild("createdAt").limitToLast(PAGE_SIZE + 1);

      if (lastTimestamp) {
        query = query.endAt(lastTimestamp - 1);
      }

      const snapshot = await query.once("value");
      const data = [];

      snapshot.forEach((childSnapshot) => {
        const val = childSnapshot.val();
        if (val && val.createdAt) {
          data.push({
            id: childSnapshot.key,
            ...val,
          });
        }
      });

      data.sort((a, b) => b.createdAt - a.createdAt);
      const newReports = data.slice(0, PAGE_SIZE);

      // setReports((prev) => [...prev, ...newReports]);
      setReports((prev) => {
        const existingIds = new Set(prev.map((r) => r.id));
        const uniqueNew = newReports.filter((r) => !existingIds.has(r.id));
        return [...prev, ...uniqueNew];
      });

      setLastTimestamp(
        newReports.length ? newReports[newReports.length - 1].createdAt : null
      );
      setHasMore(data.length > PAGE_SIZE);
    } catch (error) {
      console.error("Error fetching reports:", error);
    } finally {
      setLoading(false);
    }
  }, [loading, hasMore, lastTimestamp]);

  // Initial load effect
  useEffect(() => {
    fetchReports(true);
  }, [fetchReports]);

  // Function for loading more reports (called by intersection observer)
  const loadMoreReports = useCallback(() => {
    if (!loading && hasMore) {
      fetchReports(false, lastTimestamp);
    }
  }, [fetchReports, loading, hasMore, lastTimestamp]);

  // Observer for infinite scroll
  useEffect(() => {
    const loaderElement = loaderRef.current;

    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    if (!loaderElement || !hasMore) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMoreReports();
        }
      },
      { threshold: 0.1 }
    );

    observerRef.current.observe(loaderElement);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, loadMoreReports]);

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="history-container">
      <div className="history-header">
        <h2>Your Report History</h2>
        <p className="history-subtitle">
          Track your medical analysis reports over time
        </p>
      </div>

      {reports.length === 0 && !loading && (
        <div className="empty-state">
          <div className="empty-icon">📋</div>
          <h3>No Reports Yet</h3>
          <p>
            Your medical reports will appear here once you start using the
            diagnostic tool.
          </p>
        </div>
      )}

      <div className="reports-grid">
        {reports
          .filter((report) => report.status !== "failed")
          .map((report) => (
            <div
              key={report.id}
              className="report-card"
              onClick={() =>
                navigate("/report", {
                  state: {
                    imageUrl:
                      report.processedImageUrl || report.originalImageUrl,
                    reportId: report.id,
                  },
                })
              }
            >
              <div className="report-image-container">
                <img
                  src={report.processedImageUrl || report.originalImageUrl}
                  alt="Medical Report"
                />
                <div className="report-date">
                  {formatDate(report.createdAt)}
                </div>
              </div>
              <div className="report-content">
                <div className="diagnosis-text">
                  <h4>Diagnosis</h4>
                  <p>{report.diagnosis}</p>
                </div>
              </div>
            </div>
          ))}
      </div>

      <div ref={loaderRef} className="loader-container">
        {loading && (
          <div className="loader">
            <div className="spinner"></div>
            <p>Loading more reports...</p>
          </div>
        )}
        {!hasMore && reports.length > 0 && (
          <div className="end-message">
            <p>✨ You've reached the end of your reports</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default History;
