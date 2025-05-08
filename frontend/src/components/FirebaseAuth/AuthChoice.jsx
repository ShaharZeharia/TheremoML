import React from "react";
import { useNavigate } from "react-router-dom";
import "./AuthChoice.css";

function AuthChoice() {
  const navigate = useNavigate();

  return (
    <div className="auth-choice-container">
      <h2>Continue to your ThermoML experience</h2>
      <p>Please choose an option:</p>
      <div className="auth-buttons">
        <button onClick={() => navigate("/login/signin")}>Sign In</button>
        <button onClick={() => navigate("/login/signup")}>
          Create an Account
        </button>
      </div>
    </div>
  );
}

export default AuthChoice;
