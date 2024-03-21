# Bibliotecas generales
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Modulos espcificos de Sklearn

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart  =pd.read_csv("./data/heart.csv")
    print(dt_heart.head())

    dt_features = dt_heart.drop(["target"], axis=1)
    dt_target = dt_heart["target"]
    #Normalizamos los datos (escalarlos)
    dt_features = StandardScaler().fit_transform(dt_features)
    # Realizamos la separación de datos entre entrenamiento y test 
    
    X_train, X_test, y_train, y_test = train_test_split(dt_features,dt_target,test_size=.3, random_state=42)

