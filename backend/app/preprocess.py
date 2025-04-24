import cv2
import numpy as np
import os

def register_images(optical_path, thermal_path, output_path="aligned_thermal.jpg"):
    # Load images
    optical = cv2.imread(optical_path)
    thermal = cv2.imread(thermal_path)

    # Convert to grayscale
    gray_optical = cv2.cvtColor(optical, cv2.COLOR_BGR2GRAY)
    gray_thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_optical, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_thermal, None)

    # Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract point coordinates
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches[:50]]).reshape(-1,1,2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches[:50]]).reshape(-1,1,2)

    # Homography
    matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp thermal image
    aligned_thermal = cv2.warpPerspective(thermal, matrix, (optical.shape[1], optical.shape[0]))
    cv2.imwrite(output_path, aligned_thermal)

    return aligned_thermal




# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models 

# def register_images(optical_image, thermal_image):
#     # Step 1: Preprocessing - Resize thermal image to approximate optical image size
#     thermal_image_resized = cv2.resize(thermal_image, (optical_image.shape[1], optical_image.shape[0]))

#     # Step 2: Feature Detection - Use ORB
#     orb = cv2.ORB.create()
#     keypoints1, descriptors1 = orb.detectAndCompute(optical_image, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(thermal_image_resized, None)

#     # Step 3: Feature Matching - Brute Force Matcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(descriptors1, descriptors2)
#     matches = sorted(matches, key=lambda x: x.distance)

#     # Step 4: Compute Homography Matrix
#     src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#     matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

#     # Step 5: Warp Thermal Image
#     aligned_thermal = cv2.warpPerspective(thermal_image_resized, matrix,
#                                           (optical_image.shape[1], optical_image.shape[0]))

#     return aligned_thermal

# def build_unet(input_shape, num_classes):
#     inputs = layers.Input(input_shape)

#     # Encoder
#     conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

#     # Decoder
#     up1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(pool1)
#     up1 = layers.concatenate([up1, conv1])
#     conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
#     conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

#     outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv2)

#     return models.Model(inputs=[inputs], outputs=[outputs])

# # Define model
# input_shape = (256, 256, 4)  # Assuming 4-channel input
# num_classes = 3  # Joints, palm, outside world
# model = build_unet(input_shape, num_classes)

# # Compile
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train
# model.fit(train_dataset, validation_data=val_dataset, epochs=50, batch_size=8)
      
