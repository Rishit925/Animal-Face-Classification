# 🐾 Animal Face Classifier

An end-to-end Deep Learning application that classifies animal face images into predefined categories using a Convolutional Neural Network (CNN) built with PyTorch. The project features a Streamlit frontend, a FastAPI backend, and Docker support for containerized deployment.

---

## 📌 Features

- Train a Convolutional Neural Network (CNN) for animal face classification
- Upload an image and receive instant predictions
- Displays prediction confidence score
- Interactive Streamlit web interface
- FastAPI REST API for model inference
- Dockerized application with Docker Compose
- Modular and production-ready project structure

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- PyTorch
- TorchVision
- NumPy
- Pandas
- Scikit-learn

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit

### Deployment & Tools
- Docker
- Docker Compose
- Git
- GitHub

---

## 📂 Project Structure

```text
animal-face-classifier/
│
├── app.py                  # Streamlit frontend
├── api.py                  # FastAPI backend
├── predict.py              # Model inference logic
├── model.py                # CNN architecture
│
├── models/
│   ├── best_model.pth
│   └── label_encoder.pkl
│
├── sample_images/
├── notebooks/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

---

## 🧠 Model Architecture

The Convolutional Neural Network consists of:

- 3 Convolutional Layers
- ReLU Activation
- Max Pooling Layers
- Flatten Layer
- Fully Connected Layer
- Output Layer

**Input Image Size**

```
128 × 128 × 3
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/animal-face-classifier.git

cd animal-face-classifier
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

### Start FastAPI

```bash
uvicorn api:app --reload
```

### Start Streamlit

```bash
streamlit run app.py
```

### Open in Browser

**Streamlit**

```
http://localhost:8501
```

**FastAPI Swagger Documentation**

```
http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

Build the Docker images:

```bash
docker compose build
```

Run the application:

```bash
docker compose up
```

### Access the Application

**Streamlit**

```
http://localhost:8501
```

**FastAPI Swagger Documentation**

```
http://localhost:8000/docs
```

---

## 🔄 Application Workflow

```text
User Uploads Image
        │
        ▼
 Streamlit Frontend
        │
        ▼
 FastAPI Backend
        │
        ▼
 PyTorch CNN Model
        │
        ▼
Prediction + Confidence
        │
        ▼
 Display Results
```

---

## 📊 Sample Predictions

| Image | Prediction | Confidence |
|--------|------------|------------|
| Dog | Dog | 99.91% |
| Cat | Cat | 99.84% |
| Wild Animal | Wild | 99.76% |

---

## 📸 Screenshots

### Streamlit Interface

> *(Add homepage screenshot here)*

---

### Prediction Result

> *(Add prediction screenshot here)*

---

### FastAPI Swagger Documentation

> *(Add Swagger UI screenshot here)*

---

## 🔮 Future Improvements

- Replace the custom CNN with ResNet or EfficientNet
- Improve generalization using data augmentation
- Deploy the application to the cloud
- Add batch image prediction
- Implement model monitoring and logging
- Add CI/CD pipeline for automated deployment

---

## 👨‍💻 Author

**Rishit Mahindru**

**GitHub:** https://github.com/yourusername

**LinkedIn:** https://linkedin.com/in/yourprofile

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!