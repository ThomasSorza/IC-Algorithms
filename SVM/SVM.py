# Importar las bibliotecas necesarias
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

# Cargar el conjunto de datos desde el archivo CSV
file_path = '../ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# Separar las características (X) y la variable objetivo (y)
X = data[['Height', 'Weight']]
y = data['NObeyesdad']

# Convertir las etiquetas de la variable objetivo a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Imprimir los valores numericos de la variable objetivo junto con sus etiquetas
print('Valores numericos de la variable objetivo:')
for i, label in enumerate(label_encoder.classes_):
    print(f'{i}: {label}')


# Agrupar los valores de la variable objetivo en dos clases Obeso y No Obeso
y = np.where(y <= 2, "No Obeso", y)  # Bajo Peso y Peso Normal
y = np.where(y >= 2, "Obeso", y)  # Obeso


# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear un clasificador SVM
classifier = SVC(kernel='linear', C=1.0)

# Entrenar el clasificador
classifier.fit(X_train, y_train)

# Realizar predicciones
y_pred = classifier.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy}")
# ...

# Dibujar el limite de decisión y los margenes
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter', edgecolors='k')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title('Clasificador SVM')
plt.show()

# Guardar el modelo entrenado
model_path = 'SVM_model.pkl'
joblib.dump(classifier, model_path)
print('Modelo guardado en', model_path)

# ---------------------- Gráficos Adicionales ----------------------

# 1. Margen de Separación
margin = 1 / np.sqrt(np.sum(classifier.coef_ ** 2))
print(f"Margen de separación: {margin}")

# 2. Vectores de Soporte
support_vectors = classifier.support_vectors_
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter', edgecolors='k')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='r')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title('Vectores de Soporte en SVM')
plt.show()

# 3. Hiperplano de Decisión
w = classifier.coef_[0]
b = classifier.intercept_[0]
xx_hyperplane = np.linspace(xlim[0], xlim[1])
yy_hyperplane = - (w[0] / w[1]) * xx_hyperplane - b / w[1]
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter', edgecolors='k')
plt.plot(xx_hyperplane, yy_hyperplane, 'k-')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title('Hiperplano de Decisión en SVM')
plt.show()

# # 8. Curva ROC y Curva Precisión-Recall (para SVM binaria)
# from sklearn.metrics import roc_curve, precision_recall_curve
#
# y_scores = classifier.decision_function(X_test)
# fpr, tpr, _ = roc_curve(y_test, y_scores)
# precision, recall, _ = precision_recall_curve(y_test, y_scores)

# Curva ROC
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Curva ROC')
# plt.legend(loc='lower right')
# plt.show()

# Curva Precisión-Recall
# plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Curva Precisión-Recall')
# plt.legend(loc='lower left')
# plt.show()



# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión con seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valor Predicho')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# Realizar una predicción con el modelo entrenado
nuevo_registro = [[1.70, 70]]  # Altura: 1.70, Peso: 70
prediccion = classifier.predict(scaler.transform(nuevo_registro))

# Si es un problema de clasificación, puedes obtener la clase predicha
clase_predicha = np.argmax(prediccion)

# Imprimir la clase predicha o realizar la acción correspondiente
print(f'Clase predicha: {clase_predicha}')