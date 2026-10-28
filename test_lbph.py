import cv2
import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"

MODEL_PATH = os.path.join(
    TRAINER_PATH,
    "lbph_model.yml"
)

LABELS_PATH = os.path.join(
    TRAINER_PATH,
    "labels.txt"
)

IMAGE_SIZE = (200, 200)

TRAIN_IMAGES_PER_PERSON = 40


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

        # Support:
        # 0,Khushboo
        # 0:Khushboo
        # 0 Khushboo

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


print("=" * 60)
print("                    LBPH TESTING")
print("=" * 60)


# ==========================================
# TEST DATA
# ==========================================

y_true = []
y_pred = []

distances = []

total_test_images = 0


# ==========================================
# PROCESS EACH PERSON
# ==========================================

for person_name in sorted(
    os.listdir(DATASET_PATH)
):

    person_folder = os.path.join(
        DATASET_PATH,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue


    image_files = [
        f
        for f in os.listdir(person_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    # Numeric sorting

    try:

        image_files = sorted(
            image_files,
            key=lambda x: int(
                os.path.splitext(x)[0]
            )
        )

    except ValueError:

        image_files = sorted(
            image_files
        )


    # ======================================
    # TEST IMAGES = AFTER FIRST 40
    # ======================================

    test_files = image_files[
        TRAIN_IMAGES_PER_PERSON:
    ]


    print()
    print(f"Person: {person_name}")
    print(
        f"Testing images: {len(test_files)}"
    )


    # ======================================
    # TEST EACH IMAGE
    # ======================================

    for image_name in test_files:

        image_path = os.path.join(
            person_folder,
            image_name
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            print(
                f"WARNING: Could not read "
                f"{image_path}"
            )

            continue


        # Resize exactly like training

        image = cv2.resize(
            image,
            IMAGE_SIZE
        )


        # ==================================
        # LBPH PREDICTION
        # ==================================

        predicted_label, confidence = recognizer.predict(
            image
        )


        predicted_name = label_map.get(
            predicted_label,
            "Unknown"
        )


        # ==================================
        # STORE RESULTS
        # ==================================

        true_label = None

        for label, name in label_map.items():

            if name == person_name:

                true_label = label
                break


        y_true.append(
            true_label
        )

        y_pred.append(
            predicted_label
        )

        distances.append(
            confidence
        )

        total_test_images += 1


        print(
            f"{image_name} -> "
            f"{predicted_name} | "
            f"Distance: {confidence:.2f}"
        )


# ==========================================
# CALCULATE METRICS
# ==========================================

if total_test_images == 0:

    print()
    print(
        "ERROR: No test images found."
    )

    exit()


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ==========================================
# DISTANCE STATISTICS
# ==========================================

min_distance = min(
    distances
)

max_distance = max(
    distances
)

average_distance = np.mean(
    distances
)

median_distance = np.median(
    distances
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# ==========================================
# FINAL RESULTS
# ==========================================

print()
print("=" * 60)
print("                  LBPH RESULTS")
print("=" * 60)

print()

print(
    f"Total test images : "
    f"{total_test_images}"
)

print(
    f"Accuracy          : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision         : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall            : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score          : "
    f"{f1 * 100:.2f}%"
)

print()

print(
    f"Minimum distance  : "
    f"{min_distance:.2f}"
)

print(
    f"Maximum distance  : "
    f"{max_distance:.2f}"
)

print(
    f"Average distance  : "
    f"{average_distance:.2f}"
)

print(
    f"Median distance   : "
    f"{median_distance:.2f}"
)

print()

print("Confusion matrix:")

print(cm)

print()

print("=" * 60)
print("                 LBPH TESTING DONE")
print("=" * 60)