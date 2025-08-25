import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#Load the dataset
data = pd.read_csv('data/iris.csv)

#preprocess the dataset 
X = data.drop('species', axis=1)
y = data['species']

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.2,
random_state=42)

#train a random forest 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#save the model 
joblib.dump(model, 'model/iris_model.pkl')

#make pedictions 
y_pred = model.predict(X_test)

#evaluate the model 
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')