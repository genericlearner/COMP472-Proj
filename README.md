COMP 472 PROJECT
This project involves training and evaluating multiple models, including Decision Trees, Naive Bayes, Multi-Layer Perceptrons (MLPs), and Convolutional Neural Networks (CNNs). The models were trained on Google Colab. Therefore, it is strongly recommended to run the notebooks on Google Colab or Jupyter Notebook.

Models Overview
The trained models are too large to upload directly to GitHub. You can find all saved models on Google Drive:
Google Drive Link: [https://drive.google.com/drive/folders/1q1_0NarBIEW8wR-2UbezJEwjS3mhJW11?usp=drive_link](https://drive.google.com/drive/folders/1q1_0NarBIEW8wR-2UbezJEwjS3mhJW11?usp=sharing)

Saved Models:
Naive Bayes:

sklearn_nb: Scikit-learn Naive Bayes model saved using pickle.
manual_nb: Manual implementation of Naive Bayes saved using pickle.
Decision Trees:

sklearn_dt: Scikit-learn Decision Tree model saved using pickle.
manual_dt: Manual implementation of Decision Tree saved using pickle.
MLP (Multi-Layer Perceptron):

MLP.pth: Base MLP model saved using torch.
MLP_V1.pth: MLP model with extra layers.
MLP_V2.pth: MLP model with fewer layers.
MLP_V3.pth: MLP model with fewer weights.
MLP_V4.pth: MLP model with more weights.
CNN (Convolutional Neural Networks):

vgg11_base.pth: Base VGG11 model.
vgg11_extralayer2.pth: VGG11 with 2 extra layers.
vgg11_lesslayer2.pth: VGG11 with 2 fewer layers.
vgg11_kernel2.pth: VGG11 with kernel size 2.
vgg11_kernel5.pth: VGG11 with kernel size 5.
Directions to Load and Use Saved Models
Decision Trees and Naive Bayes:
The models for Decision Trees and Naive Bayes are saved using pickle. Use the following code snippet to load and use these models:

python
import pickle

# Load a saved model (e.g., Naive Bayes or Decision Tree)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use the loaded model for predictions
predictions = model.predict(X_test)
MLP and CNN:
The MLP and CNN models are saved using torch. Use the following code snippet to load and use these models:

python
import torch

# Load a saved model (e.g., MLP or CNN)
model.load_state_dict(torch.load('sample.pth'))

# Set the model to evaluation mode
model.eval()

# Use the model for predictions
predictions = model(X_test)
File Structure and Descriptions
Notebooks:

The DataVisualization.ipynb notebook shows the dataset and its distribution. The CNN_Notebook.ipynb covers preprocessing, training the VGG11 base model and its 4 variations, and evaluating their performance. The MLP_final.ipynb handles preprocessing, training the MLP base model and its 4 variations, and displaying metrics. The NaiveBayes_Notebook.ipynb includes Scikit-learn and Manual Naive Bayes models with training, evaluation, confusion matrices, and saving/loading models using pickle. The DecisionTrees_Notebook.ipynb features Scikit-learn and Manual Decision Tree models with training, evaluation, time comparisons, and saving/loading models using pickle.

Python scripts: Run scripts for the models with python sample_script.py
