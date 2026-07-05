import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


df = pd.read_csv("DATA/student-mat.csv", sep=";")
print("Read data file successfully.")

#tiền xử lý dữ liệu ( xóa missing value)
df = df.dropna()
df['pass'] = df['G3'].apply(lambda x: 1 if x >=10 else 0)

X = df.drop(columns=['G1','G2','G3','pass'])
y = df['pass']

for col in X.select_dtypes(include = ['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data process successfully.")

models= {"Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":RandomForestClassifier(random_state = 42),
        "SVM" : SVC()
}
best_model = None
best_accuracy = 0
best_model_name = ""


for name, model in models.items():
    print(f"Training model : {name}")

    start_train = time.time()
    model.fit(X_train, y_train)  
    end_train = time.time()
    
    start_test = time.time()
    y_pred = model.predict(X_test) 
    end_test = time.time()
    
    acc = accuracy_score(y_test, y_pred) 
    print(f"Training time: {end_train - start_train:.4f}s | Testing time: {end_test - start_test:.4f}s")
    print("Performance Metrics (Precision, Recall, F1-score):")

    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"Best Model : {best_model_name} with Accuracy: {best_accuracy:.4f}.")
joblib.dump(best_model, "best_student_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved successfully as .pkl files.")
