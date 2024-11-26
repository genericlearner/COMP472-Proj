COMP 472 PROJECT

The models of this project were trained on Google colab. Therefore it is strongly recommended to run the notebooks on colab or Jupyter. The libraries

Models:
The Models are too big to upload on Github. Please find the models here on drive: https://drive.google.com/drive/folders/1q1_0NarBIEW8wR-2UbezJEwjS3mhJW11?usp=drive_link

DIRECTIONS TO RUN SAVED MODELS FOR DECISION TREES AND NAIVE BAYES: Import the Library pickle and run this command:
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

Model Files:
sklearn_nb = Sickit learn naive bayes model saved with pickle.
manual_nb = Manual implementation of naive bayes model saved with pickle.


DIRECTIONS TO RUN SAVED MODELS FOR DECISION TREES AND NAIVE BAYES: The MLP and CNN models are saved with torch. Therefore, to run the model, run this code:
model.load_state_dict(torch.load('simple_nn_model.pth'))

MLP- The base model MLP
MLP_V1 = MLP model with extra layers
MLP_V2= MLP model with less layers
MLP_V3= MLP model with less weights
MLP_V4 = MLP model with more weights

vgg11_base = VGG11 base model
vgg11_extralayer2 = VGG11 with 2 extra layers
vgg11_lesslayer2 = VGG11 with 2 less layers
vgg11_Kernel2 = VGG11 with kernel size 2
vg11_kernel5 = VGG11 with kernel size 5

Notebooks:
DataVisualization.ipynb = This notebook visualizes the data and it's distribution.
CNN_Notebook.ipynb = This notebook contains the code for preprocessing the dataset, training the base and 4 variations of VGG11 model. The notebook also displays the metric evaluations.
MLP_final.ipynb = This notebook contains the code for preprocessing the dataset, training the base and 4 variations MLP model. The notebook also displays the metric evaluations.

Python scripts:
CNN_script runs the CNN program with all the models
DataVisualization_script runs the Data Visualization program
MLP_scirpt runs the MLP program with all the models
