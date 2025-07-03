import React, { useEffect, useState } from "react";
import {
  Link,
  Routes,
  Route,
  BrowserRouter as Router,
  useLocation,
  useNavigate,
} from "react-router-dom";
import firebase from "./firebase-config";
import "./App.css";
import SignIn from "./components/FirebaseAuth/SignIn.jsx";
import SignUp from "./components/FirebaseAuth/SignUp.jsx";
import UploadImg from "./components/UploadImg/UploadImg.jsx";
import NotFound from "./components/NotFound.jsx";
import AuthChoice from "./components/FirebaseAuth/AuthChoice.jsx";
import Home from "./components/Home/Home.jsx";
import Report from "./components/Report/Report.jsx";
import History from "./components/History/History.jsx";

function Navbar({ user, onLogout }) {
  const location = useLocation();
  const navigate = useNavigate();
  const showNavbar = !location.pathname.startsWith("/login");

  if (!showNavbar) return null;

  return (
    <nav className="navbar">
      <Link to="/" className="logo-container">
        <img src="/logo-text.svg" alt="ThermoML Logo" className="nav-logo" />
      </Link>
      <ul className="nav-links">
        <li>
          <Link to="/">Home</Link>
        </li>
        <li>
          <Link to="/home">Upload</Link>
        </li>
        <li>
          <Link to="/history">History</Link>
        </li>
      </ul>

      {user ? (
        <button className="logout-button" onClick={onLogout}>
          Logout
        </button>
      ) : (
        <button className="logout-button" onClick={() => navigate("/login")}>
          Sign In
        </button>
      )}
    </nav>
  );
}

function AppContent() {
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const auth = firebase.auth();
    const unsubscribe = auth.onAuthStateChanged((u) => setUser(u));
    return () => unsubscribe();
  }, []);

  const handleLogout = () => {
    const auth = firebase.auth();
    auth.signOut().then(() => {
      setUser(null);
      navigate("/login", { replace: true });
    });
  };

  return (
    <>
      <Navbar user={user} onLogout={handleLogout} />
      <Routes>
        <Route path="*" element={<NotFound />} />
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<AuthChoice />} />
        <Route path="/login/signin" element={<SignIn />} />
        <Route path="/login/signup" element={<SignUp />} />
        <Route path="/home" element={<UploadImg />} />
        <Route path="/report" element={<Report />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
