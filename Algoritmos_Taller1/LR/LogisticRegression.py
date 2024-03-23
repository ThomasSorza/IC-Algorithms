
import pandas as pd
import numpy as np

dataset = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv") #Reading as a Python dict
dataset

dataset.isnull().sum()

column = 'NObeyesdad'  # Column name with the classification of
valores_columna = dataset[column].unique()
print("Values of Obesity level '{}':".format(column))
print(valores_columna)

columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
for col in columns:
    dataset[col] = dataset[col].astype('category')
    dataset[col] = dataset[col].cat.codes

mapeo_obesidad = {'Normal_Weight': 0,
                'Overweight_Level_I': 0,
                'Overweight_Level_II': 0,
                'Obesity_Type_I': 1,
                'Insufficient_Weight': 0,
                'Obesity_Type_II': 1,
                'Obesity_Type_III': 1}

# Map column 'NObeyesdad'
dataset['NObeyesdad'] = dataset['NObeyesdad'].map(mapeo_obesidad)

dataset

X = dataset[['Height','Weight']]
#X = dataset.drop(columns='NObeyesdad')
X

Y = dataset['NObeyesdad']
Y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=100)

X_train

X_test

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scalated = scaler.fit_transform(X_train)
X_test_scalated = scaler.transform(X_test)

X_train_scalated

X_test_scalated

from sklearn.linear_model import LogisticRegression
#log_reg = LogisticRegression(random_state=100).fit(X_train_scalated,y_train)
#log_reg.predict(X_train_scalated)
log_reg = LogisticRegression(random_state=100,max_iter=1000).fit(X_train,y_train)
log_reg.predict(X_train)

#log_reg.score(X_test_scalated, y_test)
log_reg.score(X_test, y_test)

from sklearn.metrics import confusion_matrix

# Predicciones en los datos de prueba
y_pred = log_reg.predict(X_test)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

from sklearn.metrics import accuracy_score
print(f"Precisión:")
print(accuracy_score(y_test,y_pred)*100)

import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Crear el heatmap de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Clases predichas en todo el espacio
if X.shape[1] <=2:
    X1, X2 = np.meshgrid(np.arange(start = X_train.iloc[:, 0].min() - 1, stop = X_train.iloc[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_train.iloc[:, 1].min() - 1, stop = X_train.iloc[:, 1].max() + 1, step = 0.01))
    Z = log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    # Gráfico de superficie de clasificación
    plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_train)):
        plt.scatter(X_train[y_train == j].iloc[:, 0], X_train[y_train == j].iloc[:, 1],
                    c = ListedColormap(('blue', 'black'))(i), label = j)
    plt.title('Clasificación de datos de entrenamiento')
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.legend()
    plt.show()