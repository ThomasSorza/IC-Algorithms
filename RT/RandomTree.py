import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Cargar datos
data = pd.read_csv('../ObesityDataSet_raw_and_data_sinthetic.csv')

# Exploración de datos
obesity_counts = data["NObeyesdad"].value_counts()
print(obesity_counts)

# Preprocesamiento de variables categóricas
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
label_encoders = {col: preprocessing.LabelEncoder().fit(data[col]) for col in categorical_columns}
for col, encoder in label_encoders.items():
    data[col] = encoder.transform(data[col])

# Separar datos de entrenamiento y prueba
X = data[['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE',
          'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']]
y = data['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# Modelo de bosque aleatorio
random_forest_model = RandomForestClassifier(n_estimators=19, random_state=4, min_samples_leaf=8)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=obesity_counts.index,
            yticklabels=obesity_counts.index)
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.show()

# Guardar el modelo
joblib.dump(random_forest_model, 'random_forest_model.pkl')

