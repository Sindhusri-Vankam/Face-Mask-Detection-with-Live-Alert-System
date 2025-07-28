## 😷 Face Mask Detection with Live Alert System

### 📌 Project Overview
This project is a real-time face mask detection system that identifies whether a person is wearing a mask using a webcam. If a person is not wearing a mask, the system raises an alert. It uses a trained CNN model integrated with OpenCV for live video stream analysis.

---

### 🌟 Objective
To detect face masks in real-time and provide an alert if a face without a mask is detected.

---

### 🧰 Tools & Libraries Used
- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Haar Cascade Classifier
- Flask *(optional for web deployment)*

---

### 📁 Dataset
Dataset used: [Face Mask Dataset from Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-detection)

It contains two categories:
- With Mask
- Without Mask

---

### 🛠️ Steps Involved
1. **Data Preprocessing**  
   - Resized images to 100x100 pixels  
   - Normalized pixel values

2. **Model Building**  
   - CNN model with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers  
   - Binary classification (mask vs no mask)

3. **Model Training**  
   - Trained for 10 epochs on training and validation sets

4. **Model Integration with OpenCV**  
   - Used Haar Cascade to detect faces  
   - Passed detected face regions to CNN model  
   - Displayed bounding boxes and labels in real-time

5. **Live Detection with Alerts**  
   - Webcam stream analyzed frame-by-frame  
   - Alert shown on screen when "No Mask" is detected

---

### 📦 Output
- A trained model: `mask_detector_model.h5`
- Real-time webcam mask detection
- Option to extend with sound alerts or web-based UI using Flask

---

### ▶️ How to Run
1. Train the model using `model_training.py`  
2. Run real-time detection using `mask_detector_live.py`  
3. To test a single image, run `test_single_image.py` with a path to your image

---

### 📌 Author
Sindhusri Vankam

