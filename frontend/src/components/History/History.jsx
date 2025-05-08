import React, { useEffect, useState, useRef, useCallback } from "react";
import firebase from "../../firebase-config";
import "./History.css";

function History() {
  const [reports, setReports] = useState([]);
  const [lastTimestamp, setLastTimestamp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const loaderRef = useRef(null);
  const observerRef = useRef(null);
  const PAGE_SIZE = 5;

  const fetchReports = useCallback(async () => {
    if (loading || !hasMore) return;

    const auth = firebase.auth();
    const user = auth.currentUser;
    if (!user) return;

    setLoading(true);
    const dbRef = firebase.database().ref(`users/${user.uid}/reports`);
    let query = dbRef.orderByChild("timestamp").limitToLast(PAGE_SIZE + 1);

    if (lastTimestamp) {
      query = query.endAt(lastTimestamp - 1);
    }

    query.once("value", (snapshot) => {
      const data = [];
      snapshot.forEach((childSnapshot) => {
        data.push({
          id: childSnapshot.key,
          ...childSnapshot.val(),
        });
      });

      // Sort and update state
      data.reverse(); // newest to oldest
      const newReports = data.slice(0, PAGE_SIZE);
      setReports((prev) => [...prev, ...newReports]);

      if (newReports.length > 0) {
        setLastTimestamp(newReports[newReports.length - 1].timestamp);
      }

      // If we received fewer than PAGE_SIZE + 1, there's no more data
      if (data.length < PAGE_SIZE + 1) {
        setHasMore(false);
      }

      setLoading(false);
    });
  }, [lastTimestamp, loading, hasMore]);

  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  // Observer for infinite scroll
  useEffect(() => {
    const loaderElement = loaderRef.current;

    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    if (!loaderElement || !hasMore) return;

    observerRef.current = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        fetchReports();
      }
    });

    observerRef.current.observe(loaderElement);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [fetchReports, hasMore]);

  return (
    <div className="history-container">
      <h2>Your Report History</h2>
      {reports.map((report) => (
        <div key={report.id} className="report-card">
          <img src={report.imageUrl} alt="Report" />
          <p>{report.diagnosis}</p>
        </div>
      ))}
      <div ref={loaderRef}>
        {loading && <p>Loading more...</p>}
        {!hasMore && <p>No more reports to load.</p>}
      </div>
    </div>
  );
}

export default History;
