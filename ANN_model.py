import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.activations import sigmoid,relu
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Loading dataset

df=pd.read_csv("student_data.csv")

# Top 5 rows

print(df.head())

# Mathematical Description

print(df.describe())

# Information

print(df.info())

# Null Values

print(df.isnull().sum())

# Drop Null Values

df=df.dropna()

# Converting Categorial Columns

categorical_columns=df.select_dtypes(include="object").columns
df=pd.get_dummies(df,columns=categorical_columns,drop_first=True)
df=df.astype(int)

# Target Columns

df["Result"]=df["G3"].apply(lambda x:1 if x>=10 else 0)
X= df[["studytime","absences","G1","G2","failures"]]
Y=df["Result"]

# Train Test Split Test

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2)

# Scaler

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Early Stopping

Early_stopping=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)

# Model with dropout (0.4 to randomly drop 40 precent of neurons during training)

model=Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32,activation=relu),
    Dropout(0.48),
    Dense(16,activation=relu),
    Dropout(0.48),
    Dense(1,activation=sigmoid)
])

# Compile model

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# fit / training model

history=model.fit(X_train_scaled,Y_train,epochs=30,batch_size=32,validation_split=0.2,callbacks=[Early_stopping])

# Evaluate Model

loss,accuracy=model.evaluate(X_test_scaled,Y_test)
print("Loss:",loss)
print("Accuracy:",accuracy)

# Predictions

Y_pred=model.predict(X_test_scaled)

# Converting Prediction into classes

Y_pred_classes=(Y_pred>0.5).astype(int)

# Classification report

report=classification_report(Y_test,Y_pred_classes)
print("Classification Report:",report)

# Confusion Matrix

cm=confusion_matrix(Y_test,Y_pred_classes)
print("Confusion Matrix:",cm)

# Visualising Confusion matrix

plt.figure(figsize=(8,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC-AUC Score

roc_score=roc_auc_score(Y_test,Y_pred)
print("ROC_AUC score:",roc_score)

# Loss Curve Training and Validation

plt.figure(figsize=(10,6))
plt.plot(history.history["loss"],label="Training_loss")
plt.plot(history.history["val_loss"],label="Validation_loss")
plt.title("Learning Curves Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy Curves Training and Validation

plt.figure(figsize=(10,6))
plt.plot(history.history["accuracy"],label="Training_Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation_Accuracy")
plt.title("Accuracy Curves Training and Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Saving Model

model.save("student_ann_model.h5")

joblib.dump(scaler,"scaler.pkl")