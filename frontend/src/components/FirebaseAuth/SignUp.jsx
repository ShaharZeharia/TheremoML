import React, { useState, useEffect } from "react";
import firebase from "../../firebase-config";
import { useNavigate } from "react-router-dom";
import "./FirebaseAuthUI.css";

function SignUp() {
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
      await auth.createUserWithEmailAndPassword(email, password);
      navigate("/home");
    } catch (err) {
      let message;
      switch (err.code) {
        case "auth/email-already-in-use":
          message = "This email is already in use. Try signing in instead.";
          break;
        case "auth/invalid-email":
          message = "Please enter a valid email address.";
          break;
        case "auth/weak-password":
          message = "Password should be at least 6 characters.";
          break;
        case "auth/operation-not-allowed":
          message = "Email/password sign-up is currently disabled.";
          break;
        case "auth/network-request-failed":
          message = "Network error. Please check your connection.";
          break;
        default:
          message =
            "Failed to create account, unknown error. Please try again.";
          break;
      }
      setError(message);
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
        <h2>Create an Account</h2>
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
          <button type="submit">Sign Up</button>
          {error && <p className="error-message">{error}</p>}
        </form>
      </div>
    </div>
  );
}

export default SignUp;
