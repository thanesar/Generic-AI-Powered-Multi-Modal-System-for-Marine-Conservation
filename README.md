# 🌊 OceanSafeNet: AI-Powered Multi-Modal System for Marine Conservation

> A lightweight, eco-focused AI system using acoustic and optical image data for real-time detection of invasive species and environmental threats in marine ecosystems.

---

## 📌 Abstract

**OceanSafeNet** is a cutting-edge AI-powered system designed to safeguard marine biodiversity. It fuses **acoustic** and **optical image** data using advanced neural architectures such as **VAEs**, **CNNs**, and **RNNs**. With modules for real-time monitoring, species detection, and anomaly identification, it enables **proactive conservation**, particularly in **low-light** and **noisy underwater** environments. **OceanSafeNet-env**, an additional module, continuously monitors environmental parameters, helping scientists and conservationists intervene early.

---

## 🎯 Objectives

- 🧠 Develop AI models to detect and classify invasive species and harmful algal blooms.
- ⚙️ Create early-intervention systems for alerting environmental threats like plastic pollution.
- 🔁 Fuse multi-modal data (acoustic and image) for real-time, high-accuracy detection.
- 🛥️ Ensure model compatibility with AUVs, ROVs, and other underwater drones.
- 🌿 Minimize environmental impact with lightweight and power-efficient AI modules.

---

## 🧠 System Architecture

The system is divided into **3 core modules**:

### 1. Multi-modal Data Acquisition & Preprocessing

- **Input**: Acoustic and optical image data  
- **Techniques**:
  - CLAHE for enhancing image contrast
  - STFT for acoustic signal analysis  
- **Tools**: Parallel processing engine for scalable data ingestion and preprocessing

### 2. VAE-based Detection and Classification

- **Encoders**:
  - CNNs for optical image data
  - RNNs for acoustic signal streams  
- **Fusion Layer**: Integrates the encoded features into a shared latent space for unified decision-making  
- **Threat Identification**: Detects invasive species, plastic pollution, and anomalous patterns  
- **Proactive Response**: Triggers alerts or conservation actions via system APIs or edge deployments

### 3. Report Generation & Model Evaluation

- **Reports**: Automatically generated summaries of classification results and threat assessments  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, mAP, Sensitivity  
- **Feedback Loop**: Continuous retraining and improvement based on real-world feedback and expert annotation

---

## 🧪 Evaluation Metrics

| Metric     | Description                                      |
|------------|--------------------------------------------------|
| Accuracy   | Overall correctness of predictions               |
| Precision  | True positives among all positive predictions    |
| Recall     | True positives among all actual positives        |
| F1-score   | Harmonic mean of precision and recall            |
| mAP        | Mean Average Precision across all classes        |
| Sensitivity| Detection performance under varied environments  |

---

## 📊 Sample Test Cases

| Test Case | Scenario                             | Expected Outcome                                      |
|-----------|--------------------------------------|-------------------------------------------------------|
| 1         | Acoustic signal of invasive species  | Species correctly detected with high precision        |
| 2         | Detection of plastic using image data| Plastic identified accurately through classification  |
| 3         | Low-light environment                | System prioritizes acoustic data processing           |
| 4         | Multi-species input                  | Classifies native and invasive species accurately     |
| 5         | Obscured optical image               | System handles partial occlusion using learned features|

---

## 🧰 Technologies & Frameworks

- 🧠 **Machine Learning**: CNNs, RNNs, VAEs  
- 🧪 **Signal Processing**: STFT  
- 💡 **Image Processing**: CLAHE  
- 🧮 **Evaluation & Metrics**: Python, Scikit-learn  
- 🚤 **Deployment**: AUV/ROV-compatible, lightweight inference engines

---

## 🔮 Future Work

- 📦 Integration of live acoustic and optical feeds from marine research teams  
- 📡 Real-time dashboard with alert triggers for underwater threats  
- 🧠 Deployment of Edge AI models for low-power marine sensor nodes  
- 🌍 Expansion to coral health monitoring and pollution tracking  
