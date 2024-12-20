
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential

main = tkinter.Tk()
main.title("Prediction of time-to-event outcomes in diagnosing lung cancer based on  SVM and compare the accuracy of predicted outcome with Deep CNN algorithm")
main.geometry("1300x1200")

global filename
global classifier
global svm_acc, cnn_acc
global X, Y
global X_train, X_test, y_train, y_test
global pca

