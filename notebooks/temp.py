import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("cleaned_dataset.csv")

X = data.drop(columns=['P_HABITABLE'])
y = data['P_HABITABLE']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))

X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

X_preprocessed = pd.concat([X[numerical_cols].reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

X_preprocessed.fillna(X_preprocessed.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "habitability_model.pkl")

new_data = X_test
