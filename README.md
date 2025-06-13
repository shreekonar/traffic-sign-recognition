
# Traffic Sign Detection and Recognition using Deep Learning

## 🔍 Overview
This project implements a traffic sign detection and recognition system using deep learning models (CNN, LeNet, AlexNet) on the GTSRB dataset. It was completed as part of a master's degree coursework.

## 🎯 Objectives
- Classify German traffic signs with high accuracy
- Compare model performance across CNN variants
- Use data augmentation to improve generalization
- Analyze performance using Accuracy, Precision, Recall, F1-score

## 🧠 Models Used
- CNN with Augmentation
- LeNet
- AlexNet

## 📊 Dataset
- [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- 50K+ images across 43 traffic sign classes

## 🧪 Results
| Model               | Accuracy  |
|--------------------|-----------|
| CNN + Augmentation | 96.90%    |
| LeNet              | 94.11%    |
| AlexNet            | 95.27%    |

## 🛠 Technologies Used
- Python, TensorFlow, Keras
- OpenCV, NumPy, Pandas, Seaborn
- Google Colab (for training)

## 📁 Files
- `G16_Code.ipynb`: All model training and testing
- `G16_Report.pdf`: Detailed project report
- `README.md`: Project overview and instructions

## 📌 How to Run
1. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn opencv-python
## 🙋‍♂️ My Role
As part of the group, my primary responsibility was focused on **model implementation and training**. I worked on:
- Building and training the CNN, LeNet, and AlexNet models using TensorFlow/Keras
- Applying data augmentation techniques to improve model generalization
- Tuning hyperparameters and analyzing performance metrics (accuracy, precision, recall, F1-score)
- Contributing to the integration of preprocessing steps such as contrast normalization and edge detection

