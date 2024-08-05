import cv2
from deepface import DeepFace

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.
    
    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None]

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

def main():
    # Load filters
    teardrop = cv2.imread("filters/teardrop.png", cv2.IMREAD_UNCHANGED)
    rainbowsmile = cv2.imread("filters/rainbowsmile.png", cv2.IMREAD_UNCHANGED)
    fire = cv2.imread("filters/fire.png", cv2.IMREAD_UNCHANGED)
    exclamation = cv2.imread("filters/exclamation.png", cv2.IMREAD_UNCHANGED)
    angryeyebrows = cv2.imread("filters/angry-eyebrows.png", cv2.IMREAD_UNCHANGED)

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # IF WEBCAM DOES NOT WORK, TRY CHANGING THE INDEX FROM 0 TO 1, OR 1 TO 0
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if emotion == 'happy':
                resized_rainbowsmile_width = w
                resized_rainbowsmile_height = int((resized_rainbowsmile_width / rainbowsmile.shape[1]) * rainbowsmile.shape[0])
                rainbowsmile_resized = cv2.resize(rainbowsmile, (resized_rainbowsmile_width, resized_rainbowsmile_height))
                alpha_rainbowsmile = rainbowsmile_resized[:, :, 3] / 255.0

                # Adjust the position to 2/3 down from the top of the face
                rainbowsmile_position = (x + w // 2 - rainbowsmile_resized.shape[1] // 2, y + int(5 * h / 6) - rainbowsmile_resized.shape[0] // 2)
                overlay_image_alpha(frame, rainbowsmile_resized[:, :, :3], rainbowsmile_position, alpha_rainbowsmile)
            
            elif emotion == 'sad':
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=20, minSize=(5, 5))
                for (ex, ey, ew, eh) in eyes:
                    teardrop_resized = cv2.resize(teardrop, (ew, eh))
                    alpha_teardrop = teardrop_resized[:, :, 3] / 255.0
                    overlay_image_alpha(frame, teardrop_resized[:, :, :3], (x + ex, y + ey + int(eh / 1.5)), alpha_teardrop)
            
            elif emotion == 'angry':
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=20, minSize=(5, 5))
                if len(eyes) >= 2:
                    ex1, ey1, ew1, eh1 = eyes[0]
                    ex2, ey2, ew2, eh2 = eyes[1]

                    # Ensure the eyes are correctly detected
                    if ex1 < ex2:
                        left_eye, right_eye = (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2)
                    else:
                        left_eye, right_eye = (ex2, ey2, ew2, eh2), (ex1, ey1, ew1, eh1)

                    lx, ly, lw, lh = left_eye
                    rx, ry, rw, rh = right_eye

                    # Overlay fire on each eye
                    fire_resized1 = cv2.resize(fire, (lw, lh))
                    alpha_fire1 = fire_resized1[:, :, 3] / 255.0
                    fire_position1 = (x + lx, y + ly)
                    overlay_image_alpha(frame, fire_resized1[:, :, :3], fire_position1, alpha_fire1)

                    fire_resized2 = cv2.resize(fire, (rw, rh))
                    alpha_fire2 = fire_resized2[:, :, 3] / 255.0
                    fire_position2 = (x + rx, y + ry)
                    overlay_image_alpha(frame, fire_resized2[:, :, :3], fire_position2, alpha_fire2)

                    # Calculate the position for the angry eyebrows
                    eyebrow_width = (rx + rw // 2) - (lx + lw // 2)
                    if eyebrow_width > 0:
                        angryeyebrows_resized = cv2.resize(angryeyebrows, (eyebrow_width, lh))
                        alpha_angryeyebrows = angryeyebrows_resized[:, :, 3] / 255.0
                        angryeyebrows_position = (x + lx + lw // 2, y + ly - lh // 2)
                        overlay_image_alpha(frame, angryeyebrows_resized[:, :, :3], angryeyebrows_position, alpha_angryeyebrows)

            elif emotion == 'surprise':
                # Adjust the position of the exclamation mark
                exclamation_mark_resized = cv2.resize(exclamation, (w, h))
                alpha_exclamation_mark = exclamation_mark_resized[:, :, 3] / 255.0
                exclamation_mark_position = (x, y - h // 3)
                overlay_image_alpha(frame, exclamation_mark_resized[:, :, :3], exclamation_mark_position, alpha_exclamation_mark)

            elif emotion == 'fear':
                # Resize the teardrop image to the desired size
                teardrop_resized = cv2.resize(teardrop, (int(w * 0.1), int(h * 0.1)))
                alpha_teardrop = teardrop_resized[:, :, 3] / 255.0

                # Position the first teardrop on the top right of the forehead
                teardrop_position1 = (x + int(w * 0.75), y + int(h * 0.1))
                overlay_image_alpha(frame, teardrop_resized[:, :, :3], teardrop_position1, alpha_teardrop)

                # Position the second teardrop slightly below the first one
                teardrop_position2 = (x + int(w * 0.8), y + int(h * 0.2))
                overlay_image_alpha(frame, teardrop_resized[:, :, :3], teardrop_position2, alpha_teardrop)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(50) == 27:
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
