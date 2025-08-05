# 🌊 OceanSafeNet: AI-Powered Multi-Modal System for Marine Conservation

> A lightweight, eco-focused AI system using acoustic, sonar, and visual data for real-time detection of invasive species and environmental threats in marine ecosystems.

---

## 📌 Abstract

OceanSafeNet is a cutting-edge AI-powered system designed to safeguard marine biodiversity. It fuses sonar, acoustic, and optical image data using advanced neural architectures such as VAEs, CNNs, RNNs, and MobileNet. With modules for real-time monitoring, species detection, and anomaly identification, it enables proactive conservation, particularly in low-light and noisy underwater environments. OceanSafeNet-env, an additional module, continuously monitors environmental parameters, helping scientists and conservationists intervene early.

---

## 🎯 Objectives

- 🧠 Develop AI models to detect and classify invasive species and harmful algal blooms.
- ⚙️ Create early-intervention systems for alerting environmental threats like plastic pollution.
- 🔁 Fuse multi-modal data (acoustic, sonar, video, images) for real-time, high-accuracy detection.
- 🛥️ Ensure model compatibility with AUVs, ROVs, and other underwater drones.
- 🌿 Minimize environmental impact with lightweight and power-efficient AI modules.

---

## 🧠 System Architecture

The system is divided into 3 core modules:

### 1. **Multi-modal Data Acquisition & Preprocessing**
- **Input**: Acoustic, sonar, and optical image data.
- **Techniques**: CLAHE (for image contrast), STFT (for acoustic signal analysis).
- **Tools**: Parallel processing engine for scalable data processing.

### 2. **VAE-based Detection and Classification**
- **Encoders**:
  - CNNs for image data
  - RNNs for acoustic data
  - MobileNet for sonar data
- **Fusion Layer**: Combines all modalities into a shared latent space.
- **Threat Identification**: Analyzes output to detect invasive species, plastic, and anomalies.
- **Proactive Response**: Triggers automated conservation actions based on detected threats.

### 3. **Report Generation & Model Evaluation**
- **Reports**: Summarize species classification and threat detection.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, mAP, Sensitivity.
- **Feedback Loop**: Continuous learning from field data and expert feedback.

---

## 🧪 Evaluation Metrics

| Metric              | Description                                        |
|---------------------|----------------------------------------------------|
| Accuracy            | Overall correctness of predictions                |
| Precision           | True positives among all positive predictions     |
| Recall              | True positives among all actual positives         |
| F1-score            | Harmonic mean of precision and recall             |
| mAP                 | Mean Average Precision across all classes         |
| Sensitivity         | Detection performance under varied environments   |

---

## 📊 Sample Test Cases

| Test Case | Scenario                                 | Expected Outcome                                       |
|-----------|------------------------------------------|--------------------------------------------------------|
| 1         | Acoustic signal of invasive species       | Species correctly detected with high precision         |
| 2         | Underwater video with plastic debris      | Plastic detected accurately using YOLO/Faster R-CNN    |
| 3         | Low-light environment                     | System prioritizes acoustic data processing            |
| 4         | Multi-species input                       | Classifies native and invasive species accurately      |
| 5         | Obscured image                            | System handles occlusion and identifies objects        |

---

## 🧰 Technologies & Frameworks

- 🖼️ **Computer Vision**: OpenCV, CNNs, CLAHE
- 🔊 **Signal Processing**: STFT, RNNs
- 📦 **Model Architectures**: VAE, MobileNet, YOLO, Faster R-CNN
- 🚤 **Deployment**: AUVs/ROVs-compatible, eco-lightweight inference model
- 📈 **Evaluation**: Python, Scikit-learn

---

## 🔮 Future Work

- 📦 Integration of live drone feeds from marine research teams
- 📡 Real-time dashboard with alert triggers for underwater threats
- 🧠 Deployment of Edge AI models for low-power marine sensor nodes
- 🌍 Expansion to coral health monitoring and pollution tracking

---
