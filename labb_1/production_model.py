import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model = joblib.load('cardio-classification-model.pkl')
data = pd.read_csv('./data/samples.csv', index_col=['id'])
X, y = data.drop('cardio', axis = 1), data['cardio']

y_pred = model.predict(X)

print(classification_report(y, y_pred))