import React, { useEffect } from "react";
import "firebase/compat/auth";
import "firebaseui/dist/firebaseui.css";
import * as firebaseui from "firebaseui";
import firebase from "../../firebase-config";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import "./FirebaseAuthUI.css";

function FirebaseAuthUI() {
  const navigate = useNavigate();

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        // If the user is already logged in, redirect them to the home page
        navigate("/home", { replace: true });
      } else {
        // If not logged in, start the FirebaseUI authentication flow
        const ui =
          firebaseui.auth.AuthUI.getInstance() ||
          new firebaseui.auth.AuthUI(firebase.auth());

        ui.start("#firebaseui-auth-container", {
          signInOptions: [firebase.auth.EmailAuthProvider.PROVIDER_ID],
          signInSuccessUrl: "/home",
        });
      }
    });

    // Clean up the listener when the component is unmounted
    return () => unsubscribe();
  }, [navigate]);

  return (
    <div className="auth-page">
      <div className="auth-box">
        <img
          src="/thermal-imaging.png"
          alt="ThermoML Logo"
          className="auth-icon"
        />
        <div id="firebaseui-auth-container" />
      </div>
    </div>
  );
}

export default FirebaseAuthUI;
