### **Lung Cancer Diagnosis Prediction Using SVM and CNN**

This Python application predicts lung cancer outcomes using **Support Vector Machine (SVM)** and **Convolutional Neural Networks (CNN)** by processing CT scan images. It compares the accuracy of both models and visualizes the results.

### **Key Features**:
- **Dataset Upload**: Load lung cancer dataset.
- **Data Preprocessing**: Split data into training and testing sets, apply PCA.
- **SVM Model**: Train an SVM classifier and evaluate accuracy.
- **CNN Model**: Train a CNN to classify images and compare accuracy.
- **Cancer Prediction**: Predict if a CT scan is normal or abnormal.
- **Accuracy Visualization**: Display a bar chart comparing SVM and CNN accuracy.

---

### **Required Libraries**:

To run the project, install the following Python libraries:

```bash
pip install opencv-python matplotlib scikit-learn keras tensorflow numpy pandas tkinter
```

**Libraries and Their Purpose**:
- `opencv-python`: For image processing.
- `matplotlib`: For plotting accuracy comparison graphs.
- `scikit-learn`: For machine learning algorithms (SVM, PCA).
- `keras` and `tensorflow`: For building and training the CNN model.
- `numpy`: For numerical operations and array manipulation.
- `pandas`: For handling datasets.
- `tkinter`: For building the graphical user interface (GUI).

---

### **Library Imports**:

Here’s the syntax for importing the required libraries:

```python
# Tkinter for GUI
import tkinter as tk
from tkinter import filedialog, messagebox

# Matplotlib for plotting accuracy comparison
import matplotlib.pyplot as plt
import numpy as np

# OpenCV for image processing
import cv2

# Scikit-learn for SVM and PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Keras for CNN model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.utils.np_utils import to_categorical

# File handling
import os
from tkinter.filedialog import askopenfilename
```

---

### **Key Functions**:

1. **uploadDataset()**: Load the lung cancer dataset.
2. **splitDataset()**: Preprocess the data, split into training and testing sets, apply PCA.
3. **executeSVM()**: Train and evaluate the SVM model.
4. **executeCNN()**: Define and train the CNN model for lung cancer classification.
5. **predictCancer()**: Predict if a user-uploaded CT scan is normal or abnormal.
6. **graph()**: Display a bar chart comparing SVM and CNN accuracy.

---

### **How to Install the Modules**:

1. **Open Command Prompt/Terminal**:
   - **Windows**: Open `cmd`.
   - **macOS/Linux**: Open the Terminal.

2. **Install the required libraries**:
   Run the following command:

   ```bash
   pip install opencv-python matplotlib scikit-learn keras tensorflow numpy pandas tkinter
   ```

3. **Verify Installation**:
   You can verify the installation of each library by running:

   ```bash
   pip show <module_name>
   ```

   Example:

   ```bash
   pip show opencv-python
   ```

4. **Alternative Method**:  
   Create a `requirements.txt` file with the following content:

   ```
   opencv-python
   matplotlib
   scikit-learn
   keras
   tensorflow
   numpy
   pandas
   tkinter
   ```

   Then install all dependencies at once by running:

   ```bash
   pip install -r requirements.txt
   ```

---

### **Run the Project**:
1. **Install the required modules** using the steps above.
2. **Run the Python script** to launch the GUI.
3. **Use the buttons** to upload datasets, train models, and predict cancer outcomes.

By following these instructions, you’ll be able to set up the lung cancer diagnosis prediction project, run the SVM and CNN models, and visualize the results.

--- 
