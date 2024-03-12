
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

mapeo_obesidad = {'Normal_Weight': 2,
                'Overweight_Level_I': 3,
                'Overweight_Level_II': 4,
                'Obesity_Type_I': 5,
                'Insufficient_Weight': 1,
                'Obesity_Type_II': 6,
                'Obesity_Type_III': 7}

# Map column 'NObeyesdad'
dataset['NObeyesdad'] = dataset['NObeyesdad'].map(mapeo_obesidad)

dataset

#Importing the Necessary Libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataset.shape

dataset[dataset.isnull().any(axis=1)].head()

clean_data = dataset.copy()

#definding obesity
clean_data['has_obesity'] = (clean_data['NObeyesdad']>4)*1
clean_data['has_obesity']

#target
y=clean_data[['has_obesity']].copy()
y

obesity_vars = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
x=clean_data[obesity_vars].copy()
x.columns

y.columns

"""##Model training"""

#splitting
X_train,X_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)
#Fitting des tree
obesity_classifier = DecisionTreeClassifier(max_leaf_nodes=3,random_state=100,min_samples_split=2)
#obesity_classifier = DecisionTreeClassifier(min_samples_split=2,random_state=99)
obesity_classifier.fit(X_train,y_train)

y_predicted = obesity_classifier.predict(X_test)
#acc
accuracy_score(y_test,y_predicted)*100

confusion_matrix(y_test,y_predicted)

dataset.info()

import seaborn as sns
import matplotlib.pyplot as plt

# Calcula la matriz de confusión
conf_mat = confusion_matrix(y_test, y_predicted)

# Crea el heatmap de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

from sklearn.tree import plot_tree

# Visualizar el árbol de decisiones
plt.figure(figsize=(12, 8))
plot_tree(obesity_classifier, feature_names=x.columns, class_names=['No Obesity', 'Obesity'], filled=True)
plt.title('Árbol de Decisiones para Predicción de Obesidad')
plt.show()