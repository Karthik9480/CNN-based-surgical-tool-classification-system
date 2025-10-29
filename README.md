# CNN-based-surgical-tool-classification-system
A deep learning project that uses ResNet50 for automatic classification of surgical tools from medical images.
This model helps in improving operating room efficiency, automating surgical video analysis, and assisting in surgical phase recognition.

## 🧠 Features
- **Image Classification**: Identifies different surgical tools from input images
- **Pretrained Model**: Uses ResNet50 pretrained on ImageNet for transfer learning
- **Data Augmentation**: Improves generalization using Keras image preprocessing techniques
- **Model Evaluation**: Includes accuracy, loss visualization, and confusion matrix analysis
- **Custom Dataset Support**: Can be easily trained on any labeled surgical tool dataset

## ⚙️ Requirements
- Python 3.8+
- TensorFlow / Keras for deep learning
- NumPy, Matplotlib, Pandas for data processing and visualization
- scikit-learn for evaluation metrics
- You can install all dependencies using:
- pip install -r requirements.txt

## 📂 Dataset
- This project can be trained on any surgical tools dataset.
- Example: Kaggle – Surgical Tool Images Dataset or a custom labeled dataset structured as:

- dataset

 **train :**
 forceps,
 scissors,
 clamp,
 retractor,

 **test :**
 forceps,
 scissors,
 clamp,
 retractor,



## 🚀 Model Overview
- Architecture: ResNet50 (transfer learning)
- Input Size: 224x224 pixels
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Training and evaluation code is provided in the Jupyter notebook:
- surgical-tools-model-code.ipynb

## 🧪 Usage
1. Clone this repository:
cd Surgical-Tool-Classification

2. Install dependencies:
pip install -r requirements.txt

3. Open and run the notebook:
jupyter notebook surgical-tools-model-code.ipynb

4. (Optional) Modify dataset paths in the notebook before training.

## 📊 Results
- Model: ResNet50 (fine-tuned)
- Accuracy: ~95% (on validation dataset)
- Visualization: Includes accuracy vs loss curves and confusion matrix

## 🔮 Future Enhancements
- Deploy the model using Streamlit or Flask for real-time classification
- Add Grad-CAM visualization for model explainability

- Extend to surgical phase detection from videos

- Integrate dataset versioning and model tracking with MLflow
