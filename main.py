from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

import pandas as pd
import matplotlib.pyplot as plt

wineData = load_wine()
print(wineData)