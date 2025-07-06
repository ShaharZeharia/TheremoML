# ThermoML

  


ThermoML is an AI-powered system for detecting hand inflammations through thermal imaging analysis. Developed in collaboration with medical researchers **Dr. Yair Barzilay** and **Dr. Lilach Gavish** as part of study **TH-SRG01**, this tool provides non-invasive diagnostic capabilities for healthcare professionals.
To get more information about the project head to [Documents](https://github.com/ShaharZeharia/ThermoML-Documents).

## ✨ Features

-   **AI-powered inflammation detection** using ResNet50 and computer vision
-   **Web-based interface** for easy accessibility
-   **Firebase backend** with cloud functions
-   **Python inference system** with local and Hugging Face Space deployment
-   **Real-time thermal image processing** with SAM segmentation

  

## 🚀 Getting Started

  

**For detailed instructions, head to our [User Guide](https://github.com/ShaharZeharia/ThermoML-Documents/blob/main/User%20Guide.pdf)**

  

Or follow the quick setup guide below:

  

### 1. Clone the Repository

  

```bash

git  clone  https://github.com/ShaharZeharia/TheremoML.git

cd  TheremoML

  

```

  

### 2. Backend Setup

  

#### Requirements

  

- Python 3.12.0

- Git

- ngrok (optional, for public URL)

  

#### Download Required Models

  

1.  **SAM Model** - [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

- Checkpoint: `vit_h`2

  

2.  **HandLandMarker Model** - [Download Latest Checkpoint](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

  

3.  **Our Built ResNet50 Models** - [Google Drive Link](https://drive.google.com/file/d/1xsXC-JeMoBzNDr6dGPhHJhUylfm3ic7U/view?usp=sharing)

  

#### Environment Configuration

  

Create a `.env` file in the Python app directory:

  

```env

GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json

MODELS_PATH=./models/

PORT=8000

DEBUG=True

CALLBACK_SECRET=your_secure_callback_secret

  

```

  

#### Run the Python Backend

  

```bash

cd  app

pip  install  -r  requirements.txt

python  main.py

  

```

  

#### (Optional) Expose with ngrok

  

```bash

ngrok  http  8000

  

```

  

### 3. Firebase Cloud Functions (Local)

  

#### Environment Setup

  

Create a `.env` file inside `/functions`:

  

```env

HUGGINGFACE_URL=https://your-ngrok-url.ngrok.app/api

HUGGINGFACE_TOKEN=your_huggingface_token

# Leave blank if running locally

CALLBACK_KEY=your_secure_callback_secret

# Same as the one in the python .env file

  

```

  

#### Install and Run

  

```bash

npm  install  -g  firebase-tools

firebase  login

firebase  use  --add

cd  functions

npm  install

firebase  emulators:start  --only  functions

  

```

  

### 4. Frontend Setup

  

The frontend React app is located in the root folder.

  

#### Install Dependencies

  

```bash

npm  install

  

```

  

#### Run Locally

  

```bash

npm  run  start

  

```

  

This will launch the app at `http://localhost:3000`.

  

## 🌐 Deployment

  

### Frontend Hosting

  

1. Build the React app:

```bash

npm run build

```

2. Deploy the `build/` folder to your hosting service.

  

### Live Demo

  

🔗 **Website**: [https://thermoml.web.app](https://thermoml.web.app)

  

>  **Note**: The frontend and functions are currently deployed. The backend logic is not deployed. If you wish to upload a FLIR image through the website for processing, please contact us at **ThermoML@outlook.com** to run our server backend online.

  

## 📧 Contact

  

For backend processing requests or questions, reach out to us at: **ThermoML@outlook.com**

  

## 📸 Examples

The user's images were blurred for the privacy of the participants.

<div align="left">

 <img src="https://i.ibb.co/99tGYswN/example4.jpg" alt="Upload Example" width="600"/>

  <img src="https://i.ibb.co/bZrgW3C/example3.jpg" alt="Home Page" width="600"/>

  <br/><br/>

  <img src="https://i.ibb.co/XTN8xth/example2.jpg" alt="History Example" width="600"/>

  <br/><br/>

  <img src="https://i.ibb.co/99tGYswN/example4.jpg" alt="Upload Example" width="600"/>

</div>


## 🛠️ Tech Stack

  

-  **Frontend**: React

-  **Backend**: Python (Flask)

-  **Cloud Functions**: Firebase

-  **AI Models**: ResNet50, SAM (Segment Anything Model), Google Handlandmarker

-  **Deployment**: Firebase Hosting, Hugging Face Spaces
  

## Acknowledgments

  

This project was developed as part of our final BSc Computer Science project.

We thank Dr. [Name] and Dr. [Name] for providing access to the medical dataset under an approved study.

The software (ThermoML) is developed independently by Max Feldman and Shahar Zeharia and is licensed under the MIT License.
