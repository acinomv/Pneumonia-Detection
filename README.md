# Pneumonia-Detection

Repository for Pneumonia Detection project

## Pneumonia Detection from Chest X-Ray Images Using Deep Learning

## Project Overview  
This project develops a deep learning model for classifying chest X-ray images as **Normal** or **Pneumonia** using Convolutional Neural Networks (CNNs). The goal is to create an automated tool that supports radiologists by improving diagnostic efficiency, consistency, and early pneumonia detection.

This work demonstrates practical expertise in **computer vision**, **medical AI**, **model optimization**, and **performance evaluation**—skills valuable to organizations building intelligent, data-driven systems.

---

## Problem Statement  
Pneumonia diagnosis via chest X-rays can be time-consuming and prone to human variability. This project addresses:

- The need for **fast and reliable image classification**  
- Reducing risk of missed diagnoses (false negatives)  
- Building AI that can **scale human expertise**  

---

## Dataset  

**Source:** Kaggle – *Chest X-Ray Images (Pneumonia)*  
**Total Images:** 5,863  

## Dataset Structure

dataset/
- train/
  - NORMAL/
  - PNEUMONIA/
- val/
  - NORMAL/
  - PNEUMONIA/
- test/
  - NORMAL/
  - PNEUMONIA/


**Image characteristics:**
- Grayscale chest X-ray images  
- Resized to **256 × 256 pixels**  
- Pixel intensity reflects tissue density (white = denser structures like bone, black = lungs)

This dataset represents diverse real-world variations of pneumonia, making it suitable for building a clinically meaningful classifier.

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA process included:

- **Class distribution analysis**  
- **Visualization of Normal vs. Pneumonia samples**  
- **Verification of image dimensions and resizing**  
- **Pixel intensity distribution analysis** to understand grayscale characteristics  

These steps ensured a deeper understanding of the data and guided preprocessing and model-building decisions.

---

## Model Development

The model was refined iteratively through:

- Adding more **Conv2D layers** for deeper feature extraction  
- Using **MaxPooling2D** to reduce spatial dimensions and control overfitting  
- Increasing training epochs (**5 → 10**) to improve learning  
- Applying **L2 regularization** to all convolutional layers to reduce weight overfitting and improve generalization  

### Final Model Architecture
- Conv2D → MaxPooling  
- Conv2D → MaxPooling  
- Conv2D → MaxPooling  
- Flatten  
- Dense(256, ReLU)  
- Dense(1, Sigmoid)

**Loss:** Binary Crossentropy  
**Optimizer:** Adam  
**Regularization:** L2 applied to Conv2D layers  

---

## Results

### **Final Model Performance**
| Metric      | Score |
|-------------|--------|
| **Accuracy** | 0.80 |
| **Precision** | 0.77 |
| **Recall** | 0.98 |
| **F1 Score** | 0.86 |


## Classification Report

```

              precision    recall  f1-score   support

      Normal       0.95      0.50      0.66       234
   Pneumonia       0.77      0.98      0.86       390

    accuracy                           0.80       624
   macro avg       0.86      0.74      0.76       624
weighted avg       0.84      0.80      0.78       624


```

### Interpretation  
- **High Recall (0.98):** The model detects nearly all pneumonia cases — crucial for minimizing false negatives in clinical settings.  
- **Moderate Precision (0.77):** Some normal cases are misclassified as pneumonia, indicating room to reduce false positives.  
- **F1 Score (0.86):** Strong balance between precision and recall.

Clinically, the model excels at **sensitivity**, making it useful as a supportive tool for triage or preliminary screening.

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- ImageDataGenerator  

---

## Future Improvements

Potential enhancements include:

- Increasing the number of **Normal** samples to improve precision  
- Further tuning the L2 regularization strength  
- Implementing **data augmentation** tailored for medical imaging  
- Incorporating **transfer learning** using models like ResNet or DenseNet  
- Exploring ensemble methods for more stable predictions  

---

## About This Project  
This project demonstrates:

- Real-world use of deep learning for healthcare applications  
- Ability to optimize models through iterative experimentation  
- Strong understanding of clinical trade-offs (prioritizing recall)  
- Skill in creating scalable and interpretable AI workflows  

---

