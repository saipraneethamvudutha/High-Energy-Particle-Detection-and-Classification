# High-Energy Particle Detection and Classification

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blueviolet)](https://huggingface.co/spaces/theunknowntechie1/high-energy-particle-detection-and-classification)

This repository presents a multi-modal machine learning approach to **high-energy particle detection and classification** using three distinct deep learning methodologies:

- **Convolutional Neural Networks (CNN)** for image-based particle detection  
- **Fully Connected Neural Networks** for numerical feature classification  
- **Graph Neural Networks (GNN)** for graph-structured particle interaction data  

The project integrates these models into a unified deployment interface designed for real-world particle physics research applications.

---

## 🚀 Project Overview

High-energy physics experiments generate complex data formats such as collision images, structured measurements, and particle interaction graphs. This repository tackles this challenge by providing three complementary models:

- CNN: Processes detector image data for particle signature recognition.
- Neural Network: Handles numerical data (energy, momentum, charge, etc.).
- GNN: Analyzes relational structures within particle interaction networks.

All models come with training scripts, pre-trained weights, and a web interface for real-time testing.

---

## 🔬 Technical Approaches

### 🧠 Convolutional Neural Network (CNN)
- Focuses on particle detector images.
- Uses deep convolutional layers with batch normalization and dropout.
- Data augmentation to improve generalization.
- Suitable for real-time inference on spatial patterns in collisions.

### 📊 Neural Network for Numerical Features
- Classifies particles using numerical vectors (energy, momentum, charge).
- Fully connected architecture with regularization.
- Normalization pipelines included for preprocessing.

### 🔗 Graph Neural Network (GNN)
- Processes graphs where particles are nodes and interactions are edges.
- Built using **PyTorch Geometric**.
- Custom graph construction algorithms and batch handling.

---

## 📁 Repository Structure

```bash
├── CNN/                      # CNN architecture, training, and inference
├── neural_net/              # Numerical data processing pipeline
├── gnn/                     # Graph-based modeling using PyTorch Geometric
├── assets/                  # Pretrained models and demo input samples
├── interface/               # Gradio-powered deployment interface
├── CNN.ipynb                # Experimental notebook for CNN
├── FINAL_1.ipynb            # Combined experimentation and evaluation
└── requirements.txt         # Required dependencies
