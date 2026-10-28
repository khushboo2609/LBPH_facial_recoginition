import cv2
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "trainer/lbph_model.yml"
LABELS_PATH = "trainer/labels.txt"

CASCADE_PATH = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

IMAGE_SIZE = (200, 200)

# Initial LBPH distance threshold
UNKNOWN_THRESHOLD = 60.0


# ==========================================
# LOAD LABELS
# ==========================================

label_map = {}

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        if "," in line:

            label, name = line.split(
                ",",
                1
            )

        elif ":" in line:

            label, name = line.split(
                ":",
                1
            )

        else:

            parts = line.split(
                None,
                1
            )

            label = parts[0]
            name = parts[1]

        label_map[int(label)] = name


# ==========================================
# LOAD LBPH MODEL
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    MODEL_PATH
)


# ==========================================
# LOAD HAAR CASCADE
# ==========================================

face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH
)


if face_cascade.empty():

    print("ERROR: Haar Cascade could not be loaded.")

    exit()


# ==========================================
# OPEN WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print("=" * 60)
print("             LBPH REAL-TIME RECOGNITION")
print("=" * 60)

print()
print("Press Q to quit.")
print(
    f"Unknown threshold: {UNKNOWN_THRESHOLD}"
)
print()


last_print_time = 0


# ==========================================
# REAL-TIME LOOP
# ==========================================

while True:

    ret, frame = camera.read()


    if not ret:

        print("ERROR: Could not read webcam frame.")

        break


    # --------------------------------------
    # Convert to grayscale
    # --------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )


    # --------------------------------------
    # Process each detected face
    # --------------------------------------

    for (x, y, w, h) in faces:

        face = gray[
            y:y+h,
            x:x+w
        ]


        # Resize exactly like training

        face = cv2.resize(
            face,
            IMAGE_SIZE
        )


        # ----------------------------------
        # LBPH prediction
        # ----------------------------------

        predicted_label, distance = recognizer.predict(
            face
        )


        predicted_name = label_map.get(
            predicted_label,
            "Unknown"
        )


        # ----------------------------------
        # UNKNOWN / MATCH
        # ----------------------------------

        if distance <= UNKNOWN_THRESHOLD:

            name = predicted_name
            status = "MATCH"

        else:

            name = "Unknown"
            status = "UNKNOWN"


        # ----------------------------------
        # Draw rectangle
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ----------------------------------
        # Display name
        # ----------------------------------

        cv2.putText(
            frame,
            f"{name}",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ----------------------------------
        # Display distance
        # ----------------------------------

        cv2.putText(
            frame,
            f"Distance: {distance:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # ----------------------------------
        # Terminal output
        # ----------------------------------

        current_time = time.time()

        if current_time - last_print_time >= 0.5:

            print(
                f"Face detected | "
                f"Name: {name} | "
                f"Distance: {distance:.2f} | "
                f"Status: {status}"
            )

            last_print_time = current_time


    # --------------------------------------
    # Show webcam
    # --------------------------------------

    cv2.imshow(
        "LBPH Face Recognition",
        frame
    )


    # --------------------------------------
    # Press Q to quit
    # --------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

print()
print("LBPH recognition stopped.")