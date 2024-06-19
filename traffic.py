import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import the video footage
cap = cv2.VideoCapture('./src/assets/data_1.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
# Step 2: Get the first frame
ret, first_frame = cap.read()
if not ret:
    exit()

# Step 3: Convert the first frame to grayscale
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.equalizeHist(first_frame_gray)
# first_frame_negative = 255 -first_frame_gray

#creating particle speed array
particle_speeds = []
# Create empty list to store particle paths
particle_paths = []

# Step 4: Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 5: Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equlized_frame = cv2.equalizeHist(gray_frame)
    # negative_frame = 255 - gray_frame
    # Step 6: Calculate absolute difference between current frame and first frame
    frame_diff = cv2.absdiff(equlized_frame, first_frame_gray)
    
    
    # Step 7: Threshold the difference image to obtain a binary image
    _, binary_frame = cv2.threshold(frame_diff, 125, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    binary_frame_af_morpho = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
    
    # Step 8: Track the paths of the moving particles
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calculate centroid of each particle
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            
            # Calculate speed of the particle (assuming constant frame rate)
            particle_speeds.append((cx, cy))
            # Store particle position
            particle_paths.append((cx, cy))
    
     # Draw paths of particles
    # for i in range(1, len(particle_paths)):
    #     if (0 <= particle_paths[i - 1][0] < frame.shape[1] and
    #             0 <= particle_paths[i - 1][1] < frame.shape[0] and
    #             0 <= particle_paths[i][0] < frame.shape[1] and
    #             0 <= particle_paths[i][1] < frame.shape[0]):
    #         cv2.line(frame, particle_paths[i - 1], particle_paths[i], (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Frame',binary_frame_af_morpho)
    
    delay = int(1000 / fps)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Calculate particle speeds
# particle_speeds = np.array(particle_speeds)
# distances = np.linalg.norm(particle_speeds[1:] - particle_speeds[:-1], axis=1)
# speeds = distances * fps  # Assuming constant frame rate

# # Plot speeds
# plt.plot(speeds)
# plt.xlabel('Frame')
# plt.ylabel('Speed')
# plt.title('Particle Speed Over Time')
# plt.show()

# Convert particle paths to numpy array
# particle_paths = np.array(particle_paths)

# Plot paths
# plt.figure()
# plt.plot(particle_paths[:, 0], particle_paths[:, 1], 'b-')
# plt.xlabel('X position')
# plt.ylabel('Y position')
# plt.title('Particle Paths')
# plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
# plt.show()
