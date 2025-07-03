const functions = require("firebase-functions");
const admin = require("firebase-admin");
// const fetch = require("node-fetch");
const crypto = require("crypto");
const { defineSecret } = require("firebase-functions/params");
const { onInit } = require("firebase-functions/v2/core");
const { onRequest } = require("firebase-functions/v2/https");
// const {onSchedule} = require("firebase-functions/v2/scheduler");
const https = require("https");

const huggingFaceUrlSecret = defineSecret("hugging_face_url");
const huggingFaceTokenSecret = defineSecret("hugging_face_token");
const callbackKey = defineSecret("callback_secret");

// Configuration
let huggingFaceUrl;
let huggingFaceToken;
let callbackSecret;
// let timeoutMinutes;
onInit(() => {
  huggingFaceUrl = huggingFaceUrlSecret.value();
  huggingFaceToken = huggingFaceTokenSecret.value();
  callbackSecret = callbackKey.value();
  // timeoutMinutes = 10; // Timeout for analysis
});

admin.initializeApp();

/**
 * Sends a push notification to a specific user using their FCM token.
 *
 * This function retrieves the user's FCM token from the Firebase Realtime
 * Database and sends a notification indicating that an analysis is complete.
 * The notification includes a title, body message, and additional data
 *  such as the report ID and type.
 *
 * @async
 * @function
 * @param {string} url - The unique user ID whose notification token is stored.
 * @param {string} options - The ID of the report that completed analysis.
 * @param {string} data - The body text of the notification to be sent.
 * @return {Promise<void>} A promise that resolves when the notification is
 * sent or an error is caught.
 *
 * @example
 * await sendNotificationToUser("user123", "report456",
 * "Your hand inflammation report is ready.");
 */
function makeHttpsRequest(url, options, data) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);

    const requestOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || 443,
      path: parsedUrl.pathname + parsedUrl.search,
      method: options.method || "GET",
      headers: options.headers || {},
    };

    const req = https.request(requestOptions, (res) => {
      let responseData = "";

      res.on("data", (chunk) => {
        responseData += chunk;
      });

      res.on("end", () => {
        resolve({
          status: res.statusCode,
          data: responseData,
        });
      });
    });

    req.on("error", (error) => {
      reject(error);
    });

    if (data) {
      req.write(data);
    }

    req.end();
  });
}

// Initiate analysis and return immediately
exports.initiateAnalysis = onRequest(
  {
    secrets: [huggingFaceUrlSecret, huggingFaceTokenSecret, callbackKey],
    cors: true,
  },
  async (req, res) => {
    // Set CORS headers
    res.set("Access-Control-Allow-Origin", "*");
    res.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

    // Handle preflight OPTIONS request
    if (req.method === "OPTIONS") {
      res.status(200).send();
      return;
    }

    // Only allow POST requests
    if (req.method !== "POST") {
      res.status(405).json({ error: "Method not allowed" });
      return;
    }

    try {
      // Get the ID token from the Authorization header
      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith("Bearer ")) {
        res.status(401).json({ error: "No authorization token provided" });
        return;
      }

      const idToken = authHeader.split("Bearer ")[1];

      // Verify the ID token
      const decodedToken = await admin.auth().verifyIdToken(idToken);
      const uid = decodedToken.uid;

      console.log("Authenticated user:", uid);

      // Get data from request body
      const { imageUrl } = req.body;

      console.log("Image URL received:", imageUrl);

      // Validate imageUrl
      if (!imageUrl) {
        res.status(400).json({ error: "imageUrl is required" });
        return;
      }

      try {
        new URL(imageUrl);
      } catch (e) {
        res.status(400).json({ error: "imageUrl must be a valid URL" });
        return;
      }

      // Create new report entry
      const newRef = admin.database().ref(`users/${uid}/reports`).push();
      const reportId = newRef.key;
      const requestToken = crypto.randomBytes(32).toString("hex");

      await newRef.set({
        imageUrl,
        status: "processing",
        requestToken,
        createdAt: admin.database.ServerValue.TIMESTAMP,
        timeoutAt: Date.now() + 1000 * 60 * 30, // 30 minutes timeout
      });

      // Prepare callback URL
      const callbackUrl = `https://us-central1-${process.env.GCLOUD_PROJECT}.cloudfunctions.net/processAnalysisResult`;

      // Validate environment variables
      if (!huggingFaceUrl) {
        console.error("HUGGING_FACE_URL environment variable is not set");
        await newRef.update({
          status: "failed",
          error: "Configuration error: Missing Hugging Face URL",
          updatedAt: admin.database.ServerValue.TIMESTAMP,
        });
        res.status(500).json({ error: "Server configuration error" });
        return;
      }

      // Validate URL format
      try {
        new URL(huggingFaceUrl);
      } catch (e) {
        console.error("Invalid HUGGING_FACE_URL format:", huggingFaceUrl);
        await newRef.update({
          status: "failed",
          error: "Configuration error: Invalid Hugging Face URL format",
          updatedAt: admin.database.ServerValue.TIMESTAMP,
        });
        res.status(500).json({ error: "Server configuration error" });
        return;
      }

      const requestPayload = JSON.stringify({
        imageUrl,
        callbackUrl,
        reportId,
        uid,
        requestToken,
        callbackSecret: callbackSecret,
      });

      console.log("Making request to:", huggingFaceUrl);
      console.log("Request payload:", requestPayload);

      // Make request using native HTTPS module
      makeHttpsRequest(
        huggingFaceUrl,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${huggingFaceToken}`,
            "Content-Length": Buffer.byteLength(requestPayload),
          },
        },
        requestPayload
      )
        .then((response) => {
          console.log(
            "Request to Hugging Face successful, status:",
            response.status
          );
          console.log("Response from Hugging Face:", response.data);
        })
        .catch(async (error) => {
          console.error("Failed to send request to Hugging Face:", error);
          console.error("Error details:", error.message, error.stack);
          await newRef.update({
            status: "failed",
            error: "Failed to initiate analysis with external service",
            updatedAt: admin.database.ServerValue.TIMESTAMP,
          });
        });

      // Return success response
      res.status(200).json({
        reportId,
        status: "processing",
        message: "Analysis initiated. You will be notified when complete.",
        estimatedCompletionTime: 30 * 60, // 30 minutes in seconds
      });
    } catch (error) {
      console.error("Error in initiateAnalysis:", error);

      if (
        error.code === "auth/id-token-expired" ||
        error.code === "auth/argument-error"
      ) {
        res.status(401).json({ error: "Invalid or expired token" });
      } else {
        res.status(500).json({ error: "Internal server error" });
      }
    }
  }
);

// Function 2: Process the result when Hugging Face calls back
exports.processAnalysisResult = onRequest(
  {
    secrets: [callbackKey],
  },
  async (req, res) => {
    // Helper that logs the failure, updates the report,
    // notifies the user, and schedules deletion
    const failAndNotify = async ({ uid, reportId, reason }) => {
      try {
        const reportRef = admin
          .database()
          .ref(`users/${uid}/reports/${reportId}`);
        await reportRef.update({
          status: "failed",
          error: reason,
          updatedAt: admin.database.ServerValue.TIMESTAMP,
        });

        await sendNotificationToUser(
          uid,
          reportId,
          "Analysis failed: " + reason
        );

        // Schedule deletion in 1 minute
        setTimeout(async () => {
          try {
            await fetch(
              `https://us-central1-${process.env.GCLOUD_PROJECT}.cloudfunctions.net/deleteReport`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  uid,
                  reportId,
                  callbackSecret: callbackKey.value(),
                }),
              }
            );
            console.log(`Scheduled deletion for report ${reportId}`);
          } catch (deleteErr) {
            console.error("Failed to delete report after timeout:", deleteErr);
          }
        }, 60 * 1000);
      } catch (innerErr) {
        console.error("Error while handling failure:", innerErr);
      }
    };

    // 1. Handle non-POST
    if (req.method !== "POST") {
      const { uid, reportId } = req.body || {};
      if (uid && reportId) {
        await failAndNotify({
          uid,
          reportId,
          reason: "Invalid request method",
        });
      }
      return res.status(405).send("Method Not Allowed");
    }

    try {
      const {
        reportId,
        uid,
        diagnosis,
        processedImageUrl,
        error,
        requestToken,
        callbackSecret,
      } = req.body;

      // 2. Callback secret check
      if (callbackSecret !== callbackKey.value()) {
        if (uid && reportId) {
          await failAndNotify({
            uid,
            reportId,
            reason: "Unauthorized: Invalid callback secret",
          });
        }
        return res.status(401).send("Unauthorized");
      }

      // 3. Required fields check
      if (!reportId || !uid || !requestToken) {
        if (uid && reportId) {
          await failAndNotify({
            uid,
            reportId,
            reason:
              "Report failed during processing try a different " +
              "image, or try again",
          });
        }
        return res
          .status(400)
          .send(
            "Report failed during processing, try a different " +
              "image, or try again"
          );
      }

      const reportRef = admin
        .database()
        .ref(`users/${uid}/reports/${reportId}`);
      const snapshot = await reportRef.once("value");

      // 4. Report exists check
      if (!snapshot.exists()) {
        return res.status(404).send("Report not found");
      }

      const reportData = snapshot.val();

      // 5. Request token match check
      if (reportData.requestToken !== requestToken) {
        await failAndNotify({
          uid,
          reportId,
          reason: "Unauthorized: Request token mismatch",
        });
        return res.status(401).send("Unauthorized");
      }

      // 6. Processing state check
      if (reportData.status !== "processing") {
        await failAndNotify({
          uid,
          reportId,
          reason: "Report is not in processing state",
        });
        return res.status(400).send("Invalid report status");
      }

      // 7. Final result validity check
      const isResultInvalid = !diagnosis || !processedImageUrl;
      if (error || isResultInvalid) {
        const failReason = error || "Incomplete result data";
        await failAndNotify({ uid, reportId, reason: failReason });
        return res
          .status(200)
          .send("Failure handled and scheduled for deletion.");
      }

      // 8. Success path
      await reportRef.update({
        status: "completed",
        diagnosis,
        processedImageUrl,
        updatedAt: admin.database.ServerValue.TIMESTAMP,
      });

      await sendNotificationToUser(uid, reportId, "Analysis complete!");
      return res.status(200).send("Success");
    } catch (err) {
      console.error("Unhandled error in processAnalysisResult:", err);
      const { uid, reportId } = req.body || {};
      if (uid && reportId) {
        await failAndNotify({ uid, reportId, reason: "Internal server error" });
      }
      return res.status(500).send("Internal Server Error");
    }
  }
);

exports.deleteReport = onRequest(
  {
    secrets: [callbackKey],
  },
  async (req, res) => {
    const { uid, reportId, callbackSecret } = req.body;

    if (callbackSecret !== callbackKey.value()) {
      return res.status(401).send("Unauthorized");
    }

    if (!uid || !reportId) {
      return res
        .status(400)
        .send(
          "Report failed during processing, try a different image," +
            "or try again"
        );
    }

    try {
      await admin.database().ref(`users/${uid}/reports/${reportId}`).remove();
      console.log(`Deleted report ${reportId} for user ${uid}`);
      res.status(200).send("Deleted");
    } catch (error) {
      console.error("Error deleting report:", error);
      res.status(500).send("Internal Server Error");
    }
  }
);

// Function 4: Get report status (with better error handling)
exports.getReportStatus = functions.https.onCall(async (data, context) => {
  const uid = data.uid;
  const reportId = data.reportId;

  if (!uid || typeof reportId !== "string") {
    throw new functions.https.HttpsError(
      "invalid-argument",
      "Missing or invalid reportId or user not authenticated."
    );
  }

  try {
    const snapshot = await admin
      .database()
      .ref(`users/${uid}/reports/${reportId}`)
      .once("value");

    if (!snapshot.exists()) {
      throw new functions.https.HttpsError("not-found", "Report not found.");
    }

    const reportData = snapshot.val();

    // Don't expose the requestToken to the client
    const { ...clientSafeData } = reportData;

    return clientSafeData;
  } catch (error) {
    console.error("Error in getReportStatus:", error);
    throw new functions.https.HttpsError(
      "internal",
      "Something went wrong while fetching the report."
    );
  }
});

// Function 5: Get all user reports with pagination
exports.getUserReports = functions.https.onCall(async (data, context) => {
  const uid = data.uid;
  const limit = Math.min(data.limit || 20, 100);
  const startAfter = data.startAfter; // For pagination

  if (!uid) {
    throw new functions.https.HttpsError(
      "unauthenticated",
      "User must be authenticated."
    );
  }

  try {
    let query = admin
      .database()
      .ref(`users/${uid}/reports`)
      .orderByChild("createdAt")
      .limitToLast(limit);

    if (startAfter) {
      query = query.endBefore(startAfter);
    }

    const snapshot = await query.once("value");
    const reports = [];

    snapshot.forEach((childSnapshot) => {
      const reportData = childSnapshot.val();
      // Don't expose the requestToken to the client
      const { ...clientSafeData } = reportData;
      reports.push({
        id: childSnapshot.key,
        ...clientSafeData,
      });
    });

    return {
      reports: reports.reverse(), // Most recent first
      hasMore: reports.length === limit,
    };
  } catch (error) {
    console.error("Error in getUserReports:", error);
    throw new functions.https.HttpsError(
      "internal",
      "Something went wrong while fetching reports."
    );
  }
});

/**
 * Sends a push notification to a specific user using their FCM token.
 *
 * This function retrieves the user's FCM token from the Firebase Realtime
 * Database and sends a notification indicating that an analysis is complete.
 * The notification includes a title, body message, and additional data
 *  such as the report ID and type.
 *
 * @async
 * @function
 * @param {string} uid - The unique user ID whose notification token is stored.
 * @param {string} reportId - The ID of the report that completed analysis.
 * @param {string} message - The body text of the notification to be sent.
 * @return {Promise<void>} A promise that resolves when the notification is
 * sent or an error is caught.
 *
 * @example
 * await sendNotificationToUser("user123", "report456",
 * "Your hand inflammation report is ready.");
 */
async function sendNotificationToUser(uid, reportId, message) {
  try {
    // Get user's FCM token from database
    const tokenSnapshot = await admin
      .database()
      .ref(`users/${uid}/fcmToken`)
      .once("value");

    if (tokenSnapshot.exists()) {
      const fcmToken = tokenSnapshot.val();

      const notification = {
        title: "Analysis Complete",
        body: message,
        data: {
          reportId: reportId,
          type: "analysis_complete",
        },
      };

      await admin.messaging().sendToDevice(fcmToken, {
        notification: notification.title
          ? {
              title: notification.title,
              body: notification.body,
            }
          : undefined,
        data: notification.data,
      });

      console.log(`Notification sent to user ${uid}`);
    } else {
      console.log(`No FCM token found for user ${uid}`);
    }
  } catch (error) {
    console.error(`Failed to send notification to user ${uid}:`, error);
  }
}
