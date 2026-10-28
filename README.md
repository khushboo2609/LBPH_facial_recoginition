# LBPH Face Recognition using OpenCV

A Python-based face recognition system using **Local Binary Patterns Histograms (LBPH)** and **Haar Cascade face detection**. This project demonstrates face detection, image preprocessing, model training, face recognition, and real-time webcam testing using OpenCV.

## 📌 Project Overview

Local Binary Patterns Histograms (LBPH) is a traditional face recognition algorithm that identifies faces using local texture patterns. It extracts texture features from facial images, divides them into regions, and generates histograms to represent each face.

During recognition, the trained model compares the input face with learned facial patterns and predicts the closest identity.

## 🎯 Objectives

- Detect faces using Haar Cascade.
- Capture and prepare facial images for training.
- Train an LBPH face recognition model.
- Recognize faces from images or a webcam.
- Use a threshold to classify predictions as recognized or unknown.
- Evaluate recognition performance using test images.

## 🛠️ Technologies Used

- Python 3.10
- OpenCV Contrib
- NumPy
- Haar Cascade Classifier
- LBPH Face Recognizer
- Visual Studio Code
- PowerShell

## ⚙️ How LBPH Works

1. **Face Detection:** Haar Cascade identifies the face region in an image.
2. **Preprocessing:** The detected face is converted to grayscale and resized to a consistent dimension.
3. **Local Binary Pattern Extraction:** Each pixel is compared with neighboring pixels to generate a binary texture pattern.
4. **Histogram Generation:** The face is divided into regions, and histograms of local binary patterns are calculated.
5. **Model Training:** The LBPH recognizer learns from labeled facial images.
6. **Face Recognition:** A new face is compared with the trained model to predict an identity.
7. **Thresholding:** The predicted distance is compared with a configured threshold to decide whether to accept the match.

**Note:** A lower LBPH distance indicates a closer match. The threshold should be selected using validation data rather than assumed to work for every dataset.

## 📁 Project Structure

```text
04_lbph/
├── dataset/
│   └── person_name/
│       └── face_images/
├── trainer/
│   ├── lbph_model.yml
│   └── labels.txt
├── haarcascade_frontalface_default.xml
├── capture_faces.py
├── train_lbph.py
├── test_lbph.py
├── recognize_lbph.py
├── requirements.txt
└── README.md
```

*The directory and filenames above are a suggested structure. Adjust them to match the actual files in your repository.*

## 💻 Requirements

- Python 3.10
- pip
- OpenCV Contrib
- NumPy
- A webcam for live recognition
- Facial images organized with identity labels

See [`requirements.txt`](requirements.txt) for the Python dependencies.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd 04_lbph
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's actual URL.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv face_env
.\face_env\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, you can run the Python executable directly from `face_env\Scripts\python.exe`.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify OpenCV

```bash
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('LBPH available:', hasattr(cv2, 'face'))"
```

The final line should display `True`.

## ▶️ Usage

Run the following commands from the project root, using the filenames that exist in your repository.

### Step 1: Capture face images

```bash
python capture_faces.py
```

Capture and save face samples for the identities you want to recognize.

### Step 2: Train the model

```bash
python train_lbph.py
```

The training script should load labeled face images and save the trained LBPH model.

### Step 3: Evaluate the model

```bash
python test_lbph.py
```

Use images that were not included in training to evaluate recognition performance.

### Step 4: Start live recognition

```bash
python recognize_lbph.py
```

The recognition script should access the webcam, detect faces, and display predicted identities according to the configured threshold.

**Important:** These commands assume the scripts use the filenames shown in this README. Update the commands if your actual scripts have different names.

## 📊 Experimental Results

The recorded experiment used 50 images of one identity, with 40 training images and 10 testing images.

| Metric | Result |
|---|---:|
| Total images | 50 |
| Training images | 40 |
| Testing images | 10 |
| Correct test predictions | 10 / 10 |
| Test accuracy | 100% |
| Average LBPH distance | 22.58 |
| Median LBPH distance | 22.69 |
| Observed test distance range | 21.62–23.18 |

These results apply only to the recorded experiment. They do not establish accuracy for multiple identities, unknown-person rejection, or varied real-world conditions.

## ✅ Advantages

- Lightweight and CPU-friendly.
- Simple to implement with OpenCV.
- Useful for learning classical face recognition.
- Supports real-time webcam experiments.
- Does not require a deep-learning model or GPU.

## ⚠️ Limitations

- Sensitive to changes in lighting, pose, and image quality.
- Recognition may be affected by facial expressions and occlusion.
- Performance depends on the quality and diversity of training images.
- A distance threshold may produce false matches or false rejections.
- Testing one identity is insufficient to validate multi-person recognition.

## 🔮 Future Improvements

- Add support for multiple identities.
- Increase the size and diversity of the dataset.
- Evaluate unknown-person rejection using unseen identities.
- Tune the recognition threshold on a validation dataset.
- Record precision, recall, F1-score, and false-acceptance rates where applicable.
- Compare LBPH with Eigenfaces, Fisherfaces, and ArcFace.

## 🔒 Privacy and Responsible Use

Obtain appropriate consent before collecting or storing facial images. Protect the dataset and trained models, restrict access to personal data, and avoid relying on face recognition as the sole basis for important decisions.

## 📚 References

- [OpenCV LBPHFaceRecognizer Documentation](https://docs.opencv.org/4.8.0/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Contrib Repository](https://github.com/opencv/opencv_contrib)

## 👩‍💻 Author

**Khushboo Kumari**

Computer Vision | Face Recognition | Python | OpenCV
