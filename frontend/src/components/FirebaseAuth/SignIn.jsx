import React, { useState, useEffect } from "react";
import firebase from "../../firebase-config";
import { useNavigate } from "react-router-dom";
import "./FirebaseAuthUI.css";

function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const auth = firebase.auth();
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (user) {
        navigate("/home", { replace: true });
      }
    });
    return () => unsubscribe();
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const auth = firebase.auth();
    try {
      await auth.signInWithEmailAndPassword(email, password);
      navigate("/home");
    } catch (err) {
      setError("Failed to sign in. Please check your credentials.");
      console.error(err);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-box">
        <img
          src="/thermal-imaging.png"
          alt="ThermoML Logo"
          className="auth-icon"
        />
        <h2>Sign In</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            required
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            required
            onChange={(e) => setPassword(e.target.value)}
          />
          <button type="submit">Sign In</button>
          {error && <p className="error-message">{error}</p>}
        </form>
      </div>
    </div>
  );
}

export default SignIn;
