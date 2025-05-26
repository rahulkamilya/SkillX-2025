
# 🧠 Brain Tumor Detection Using YOLOv10

This project focuses on detecting brain tumors from MRI scans using the YOLOv10 deep learning model. We trained the model using annotated data and deployed it through a web interface using Gradio for real-time tumor detection.

---

## 📌 Project Overview

- **Model**: YOLOv10n (Nano version)
- **Dataset**: Roboflow annotated MRI brain images
- **Training Platform**: Google Colab with **T4 GPU**
- **Interface**: Real-time prediction using Gradio
- **Precision Achieved**: 100%

---

## 🚀 Setup Instructions

### Step 1: Install Required Libraries

```bash
!pip install -q git+https://github.com/THU-MIG/yolov10.git
!pip install -q roboflow
!pip install -q gradio
```

### Step 2: Download Pretrained YOLOv10n Weights

```bash
!wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt
```

### Step 3: Load the Dataset from Roboflow

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("brain-mri").project("mri-rskcu")
version = project.version(3)
dataset = version.download("yolov8")
```

💡 Ensure your Colab runtime is set to **T4 GPU**:  
`Runtime > Change runtime type > Hardware Accelerator > GPU (T4)`

---

## 🧠 Model Training

```bash
!yolo task=detect mode=train epochs=25 batch=32 plots=True \
model='/content/yolov10n.pt' \
data='/content/MRI-3/data.yaml'
```

- **Epochs**: 25  
- **Batch Size**: 32  
- **Data Format**: YOLOv8-compatible YAML  
- **Output Model**: `/runs/detect/train/weights/best.pt`

---

## 📈 Inference and Testing

### Predict on Full Validation Set

```python
from ultralytics import YOLO
model_path = "/content/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

result = model(source="/content/MRI-3/valid/images", conf=0.25, save=True)
```

### Predict on Single Image

```python
result = model.predict(
  source="/content/MRI-3/valid/images/Tr-glTr_0000_jpg.rf.ee4ad3ca5d0eafd1f482988b89457634.jpg", 
  imgsz=640, 
  conf=0.25
)
annotated_img = result[0].plot()
annotated_img[:, :, ::-1]
```

---

## 🌐 Gradio Web Interface

Create an easy-to-use interface to upload MRI images and receive predictions:

```python
import gradio as gr
import numpy as np
import cv2

def predict(image):
    result = model.predict(source=image, imgsz=640, conf=0.25)
    annotated_img = result[0].plot()
    return annotated_img[:, :, ::-1]

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload an image"),
    outputs=gr.Image(type="numpy", label="Detect Brain Tumor"),
    title="Brain Tumor Detection Using YOLOv10",
    description="Upload an image and the model will detect tumor regions."
)

app.launch()
```

---

## 📊 Results and Evaluation

| Metric     | Value     |
|------------|-----------|
| Precision  | 100%      |
| Recall     | 66.67%    |
| F1 Score   | 80%       |
| mAP@0.5    | >85%      |

🔎 **Visual Output**  
Bounding boxes clearly show tumor areas on MRI scans.  
High precision ensures reliability in diagnosis support.

---

## 🔮 Future Scope

- Improve recall by expanding the dataset
- Add classification for tumor types (e.g., benign vs malignant)
- Optimize for mobile/edge deployment
- Integrate with hospital systems for real-time use
- Collaborate with radiologists for validation and feedback

---

## 👨‍💻 Authors

- **Rahul Kamilya** – 22CS011130  
- **Priyanshu Das** – 22CS011126  
- **Ranojit Das** – 22CS011136  
- **Prasun Bhattacharya** – 22CS011121  
- **Partha Sarathi Maity** – 22CS011115

---

## 📄 License

This project is intended for educational and research purposes only. Dataset rights remain with Roboflow and respective contributors.
