# NSCLC Subtype Classification Using Patch-based CNN with EM Algorithm

## Project Overview

This project implements a deep learning approach for classifying Non-Small Cell Lung Carcinoma (NSCLC) into its major subtypes: **Adenocarcinoma (ADC)** and **Squamous Cell Carcinoma (SCC)**. The system uses a **patch-based Convolutional Neural Network (CNN)** with an **Expectation-Maximization (EM) algorithm** to identify discriminative regions in CT scans, enabling accurate cancer subtype classification.

---

## Key Features

- Patch-based analysis of high-resolution CT scans  
- Automatic identification of discriminative regions using EM algorithm  
- Two-level classification approach with decision fusion  
- Interactive web interface for clinical use  
- High accuracy comparable to expert pathologists  

---

## Methodology

### 1. Patch-based CNN with EM Algorithm

This project implements the methodology described in the paper _"Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification."_ The key insight is that not all regions of a medical image are equally informative for diagnosis.

**Approach:**

- **Patch Extraction**: Extracts multiple high-resolution patches from CT scans  
- **EM-based Discriminative Patch Selection**:
  - Initially consider all patches as discriminative  
  - Train a CNN model to predict cancer subtypes  
  - Apply spatial Gaussian smoothing to probability maps  
  - Select patches with higher probability values as discriminative  
  - Iterate until convergence  

**Two-level Classification:**

- First level: Patch-level CNN classification  
- Second level: Decision fusion using logistic regression or SVM  

### 2. CNN Architecture

The CNN architecture consists of:

- 5 convolutional blocks with batch normalization  
- Adaptive pooling to ensure consistent feature map dimensions  
- Fully connected layers for final classification  

### 3. Decision Fusion

The system combines patch-level predictions using a **Count-based Multiple Instance (CMI)** learning approach:

- Creates histograms of patch-level predictions  
- Trains a second-level classifier (logistic regression or SVM)  
- Makes final image-level predictions  

---

## Technical Implementation

- **Framework**: PyTorch  
- **Dataset**: NSCLC-Radiomics dataset  
- **Web Interface**: Streamlit  
- **Image Processing**: OpenCV, scikit-image, pydicom  

---

## Installation and Usage

### Prerequisites

Install required packages:

```bash
pip install torch torchvision streamlit opencv-python scikit-image pydicom matplotlib numpy pillow
```

### Running the Application

```bash
streamlit run deploy.py
```

### Configuration

Create a `.streamlit/config.toml` file with the following content:

```toml
[server]
runOnSave = false
enableStaticServing = true

[runner]
fastReruns = false
```

---

## Results

The model achieves high accuracy in distinguishing between **ADC** and **SCC** subtypes, comparable to the performance of expert pathologists. The patch-based approach with EM algorithm effectively identifies the most discriminative regions in the CT scans, improving classification accuracy.

---

## Deployment

The project includes a **Streamlit web application** that allows medical professionals to:

- Upload DICOM files or CT scan images  
- Visualize the uploaded images  
- Get cancer subtype predictions with confidence scores  
- View probability distributions for different subtypes  

---

## Future Work

- Extend the model to classify additional NSCLC subtypes  
- Implement explainable AI techniques for better interpretability  
- Integrate with hospital PACS systems for seamless clinical workflow  
- Develop mobile applications for remote diagnosis  

---

## Disclaimer

> This tool is for research purposes only and should not be used for clinical diagnosis without proper validation.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Methodology based on the paper _"Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification"_  
- **NSCLC-Radiomics** dataset for providing the training and testing data