#Importing all the necessary libraries
import tkinter as tk
import tkinter.font as font
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
matplotlib.use("TKAgg")

df = pd.read_csv(r"C:\Users\DELL\Downloads\archive2\creditcard.csv") #Importing the dataset
df.drop_duplicates(inplace = True) #removing all the duplicates from the dataset

#splitting the data
X = df.drop(columns = ['Class', 'Time'], axis = 1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2)

#Training logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

#Pedicting by using testing dataset
y_pred = lr.pred(X_test)

#Creating confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

#Calculating accuracy score of the model
acsc = accuracy_score(y_pred, y_test)  

def predict(event):
    pred = lr.pred(inp)
    tk.Label(window, text = "Credit Card Class Prediction for features = " + str(inp) + "is " + str(pred))

window = tk.Tk()
tk.Label(window, text="Credit Card Fraud Detection", justify= tk.CENTER, font = font.Font(size = 32,weight='bold')).pack()
inp = pd.Series(map(float, tk.Entry(window).get().split()))
button = tk.Button(window, text = "Predict", width = 10, command = predict)
button.pack(padx = 10, pady = 10)
tk.Label(window, text = "Confusion Matrix is:")

fig = plt.figure(figsize = (1500, 1500))


    
