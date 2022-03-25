import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model = joblib.load('cardio-classification-model.pkl')
soft_model = joblib.load('cardio-classification-model-soft-voting.pkl')#For being able to run predict proba (#AttributeError: predict_proba is not available when voting='hard')

data = pd.read_csv('./data/samples.csv', index_col=['id'])
X, y = data.drop('cardio', axis = 1), data['cardio']

y_pred = model.predict(X)

print(classification_report(y, y_pred))


y_pred_soft = soft_model.predict(X)
print(classification_report(y, y_pred_soft))

probabilities = soft_model.predict_proba(X)

X.reset_index(inplace=True)

result_hard = pd.DataFrame({'id': X.id, 'predicted': y_pred})
result_soft = pd.DataFrame({'id': X.id, 'predicted': y_pred_soft, 'probability_0': probabilities[:,0], 'probability_1': probabilities[:,1] })


result_hard.to_csv('./data/prediction.csv')
result_soft.to_csv('./data/prediction-soft.csv')

print('results saved to csv')