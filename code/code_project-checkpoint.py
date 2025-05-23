import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=ConvergenceWarning)

##Dataset cleaning##
# Load data
filename = 'dataset_MLproject_new.xlsx'
dataframe = pd.read_excel('dataset_MLproject_new.xlsx')
print(dataframe.head())












