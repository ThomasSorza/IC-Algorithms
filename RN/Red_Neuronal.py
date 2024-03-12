# Importar las bibliotecas necesarias
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

# Cargar el conjunto de datos desde el archivo CSV
file_path = '../ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)

# Convertir variables categóricas usando one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'CALC', 'SMOKE', 'SCC',
                                 'MTRANS'])

# Convertir la variable objetivo
le = LabelEncoder()
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])

# Dividir los datos en características (X) y etiquetas (y)
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Convertir las etiquetas de la variable objetivo a valores numéricos
y = le.fit_transform(y)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear la red neuronal
classifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)

# Entrenar la red neuronal
classifier.fit(X_train, y_train)

# Realizar predicciones
y_pred = classifier.predict(X_test)

# Calcular la precisión de la red neuronal
accuracy = classifier.score(X_test, y_test)
print(f"Precisión: {accuracy}")

# Guardar el modelo
joblib.dump(classifier, 'neural_network_model.pkl')

# Graficar la matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Dibujar el grafico de convergencia
plt.plot(classifier.loss_curve_)
plt.title('Curva de convergencia')
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.show()

