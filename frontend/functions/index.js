const functions = require("firebase-functions");
const admin = require("firebase-admin");
const fetch = require("node-fetch");

admin.initializeApp();

exports.analyzeAndSaveReport = functions
  .region("europe-west1") // Change if deploying elsewhere
  .https.onCall(async (data, context) => {
    const uid = context.auth?.uid;
    const imageUrl = data.imageUrl;

    if (!uid || !imageUrl) {
      throw new functions.https.HttpsError(
        "unauthenticated",
        "User must be authenticated and imageUrl must be provided."
      );
    }

    try {
      // Example Hugging Face call — customize with your API endpoint
      const response = await fetch("https://your-huggingface-model/api", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer YOUR_HUGGING_FACE_TOKEN`,
        },
        body: JSON.stringify({ imageUrl }),
      });

      if (!response.ok) {
        throw new Error("Failed to get diagnosis from Hugging Face");
      }

      const result = await response.json();

      const diagnosis = result.diagnosis || "No diagnosis available";
      const processedImageUrl = result.processedImageUrl || imageUrl;

      // Save to Firebase Realtime Database
      const newRef = admin.database().ref(`users/${uid}/reports`).push(); // generates unique report ID

      await newRef.set({
        imageUrl,
        processedImageUrl,
        diagnosis,
        createdAt: admin.database.ServerValue.TIMESTAMP,
      });

      return { diagnosis, processedImageUrl };
    } catch (error) {
      console.error("Error in analyzeAndSaveReport:", error);
      throw new functions.https.HttpsError(
        "internal",
        "Something went wrong while analyzing the image."
      );
    }
  });
