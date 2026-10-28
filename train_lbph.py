import cv2
import os
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"

IMAGE_SIZE = (200, 200)
TRAIN_IMAGES_PER_PERSON = 40

MODEL_PATH = os.path.join(
    TRAINER_PATH,
    "lbph_model.yml"
)

LABELS_PATH = os.path.join(
    TRAINER_PATH,
    "labels.txt"
)

# ==========================================
# CREATE TRAINER FOLDER
# ==========================================

os.makedirs(
    TRAINER_PATH,
    exist_ok=True
)

print("=" * 60)
print("                 LBPH TRAINING")
print("=" * 60)

faces = []
labels = []

label_map = {}
current_label = 0

# ==========================================
# LOAD DATASET
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

    if len(image_files) < TRAIN_IMAGES_PER_PERSON:

        print()
        print(
            f"WARNING: {person_name} has "
            f"only {len(image_files)} images."
        )

        continue

    label_map[current_label] = person_name

    print()
    print(f"Person: {person_name}")
    print(f"Total images: {len(image_files)}")
    print(
        f"Training images: "
        f"{TRAIN_IMAGES_PER_PERSON}"
    )
    print(
        f"Testing images: "
        f"{len(image_files) - TRAIN_IMAGES_PER_PERSON}"
    )

    # ======================================
    # FIRST 40 = TRAINING
    # ======================================

    train_files = image_files[
        :TRAIN_IMAGES_PER_PERSON
    ]

    for image_name in train_files:

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

        image = cv2.resize(
            image,
            IMAGE_SIZE
        )

        faces.append(image)
        labels.append(current_label)

    current_label += 1

# ==========================================
# CHECK TRAINING DATA
# ==========================================

if len(faces) == 0:

    print()
    print("ERROR: No training images found.")
    exit()

faces = np.array(faces)
labels = np.array(labels)

print()
print("=" * 60)
print("              TRAINING SUMMARY")
print("=" * 60)

print(
    f"Training images loaded: {len(faces)}"
)

print(
    f"Number of people: {len(label_map)}"
)

# ==========================================
# CREATE LBPH MODEL
# ==========================================

print()
print("Creating LBPH model...")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

# ==========================================
# TRAIN
# ==========================================

recognizer.train(
    faces,
    labels
)

print("Training completed!")

# ==========================================
# SAVE MODEL
# ==========================================

recognizer.write(
    MODEL_PATH
)

# ==========================================
# SAVE LABELS
# ==========================================

with open(
    LABELS_PATH,
    "w",
    encoding="utf-8"
) as file:

    for label, name in label_map.items():

        file.write(
            f"{label},{name}\n"
        )

# ==========================================
# FINAL OUTPUT
# ==========================================

print()
print("=" * 60)
print("               MODEL SAVED")
print("=" * 60)

print()
print(f"Model: {MODEL_PATH}")
print(f"Labels: {LABELS_PATH}")

print()
print("=" * 60)
print("             LBPH TRAINING DONE")
print("=" * 60)