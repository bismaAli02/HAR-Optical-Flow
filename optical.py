import cv2
import numpy as np

# Function to generate optical flow


def generate_optical_flow(prev_gray, curr_gray):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow
    # optical_flow, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None, **lk_params)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    optical_flow, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None, **lk_params)
    # Filter out points with invalid status

    if optical_flow is not None:
        good_new = optical_flow[status == 1]
        good_old = p0[status == 1]

    # Display the optical flow features (you can customize this part based on your needs)
    for i, (new, old) in enumerate(zip(optical_flow, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(curr_gray, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(curr_gray, (int(a), int(b)), 5, (0, 255, 0), -1)

    return optical_flow, good_old

# Function to extract upper and lower coordinates


def extract_coordinates(optical_flow):
    # Reshape the optical flow array to (N, 2)
    optical_flow_reshaped = optical_flow.reshape(-1, 2)

    # Extract x and y coordinates from optical flow vectors
    x_coords = optical_flow_reshaped[:, 0]
    y_coords = optical_flow_reshaped[:, 1]

    # Find the upper and lower coordinates
    upper_coord = (int(np.min(x_coords)), int(np.min(y_coords)))
    lower_coord = (int(np.max(x_coords)), int(np.max(y_coords)))

    print(upper_coord, lower_coord)
    return upper_coord, lower_coord


# Function to calculate distance


def calculate_distance(upper_coord, lower_coord):
    return abs(upper_coord[1] - lower_coord[1])

# Function to classify activity based on distance


def classify_activity(distance, threshold):
    if distance < threshold:
        return "Sitting"
    else:
        return "Standing"


def generate_frames():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Read the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    # Set an initial threshold (you may need to adjust this based on observations)
    threshold = 100

    while True:
        # Read the current frame
        ret, curr_frame = cap.read()
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Generate optical flow
        optical_flow, good_old = generate_optical_flow(prev_gray, curr_gray)

        # Extract upper and lower coordinates
        upper_coord, lower_coord = extract_coordinates(optical_flow)

        # Calculate distance
        distance = calculate_distance(upper_coord, lower_coord)

        # Classify activity based on distance
        activity_label = classify_activity(distance, threshold)

        # Display optical flow features
        # Your code to display optical flow features here

        # Display the label

        for i, (new, old) in enumerate(zip(optical_flow, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(curr_gray, (int(a), int(b)),
                     (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(curr_gray, (int(a), int(b)), 5, (0, 255, 0), -1)

        # Display dots at the upper and lower coordinates
        # Blue dot for upper coordinate

        # Display the frame
        # Apply background subtraction
        fgmask = fgbg.apply(curr_frame)

        # Optionally, you can perform morphological operations to improve the mask
        # For example, you can use cv2.morphologyEx() to perform closing
        # kernel = np.ones((5, 5), np.uint8)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        next = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Apply the binary mask to the output of dense optical flow
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = cv2.bitwise_and(bgr, bgr, mask=fgmask)

        cv2.putText(result, "Activity: {}".format(activity_label),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update the previous frame and grayscale image for the next iteration
        prev_gray = curr_gray.copy()

        _, buffer = cv2.imencode('.jpg', result)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
