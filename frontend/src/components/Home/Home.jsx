import React from "react";
import "./Home.css";

function Home() {
  return (
    <div className="home-page">
      <header className="home-header">
        <h1>Welcome to ThermoML</h1>
        <p>AI-driven thermal image analysis for early inflammation detection</p>
      </header>

      <section className="home-section">
        <h2>Our Mission</h2>
        <p>
          ThermoML is a medical AI system developed to detect inflammations and
          hand-related pathologies using machine learning on thermal images. Our
          goal is to provide a reliable, non-invasive tool that supports doctors
          in early diagnosis and patient care.
        </p>
      </section>

      <section className="home-section">
        <h2>Research Background</h2>
        <p>
          This project is developed in collaboration with the{" "}
          <strong>TH-SRG01</strong> study, led by Dr. Yair Barzilay and Dr.
          Lilach Gavish. The study focuses on thermal imaging of surgeons'
          hands, a population known to be susceptible to inflammation due to the
          physical demands of surgery.
        </p>
        <p>
          By analyzing thermal anomalies and pain distribution patterns,
          ThermoML aims to provide a practical diagnostic tool that helps
          doctors and patients detect inflammation more effectively. The dataset
          includes thermal images of both surgeons and non-surgeon doctors to
          improve model accuracy and reliability.
        </p>
      </section>
    </div>
  );
}

export default Home;
