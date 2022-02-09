#Importing all the necessary libraries
import tkinter as tk
import tkinter.font as font
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
lr.fit(X_train. values, y_train.values)

#Pedicting by using testing dataset
y_pred = lr.predict(X_test.values)

#Creating confusion matrix
cm = metrics.confusion_matrix(y_test.values, y_pred)

#Calculating accuracy score of the model
acsc = metrics.accuracy_score(y_pred, y_test.values)  

#Predicting Class from features inputed
def predict_class():

    inp_s = pd.Series([float(v1.get()), float(v2.get()), float(v3.get()), float(v4.get()), float(v5.get()), float(v6.get()), float(v7.get()), float(v8.get()), float(v9.get()), float(v10.get()), float(v11.get()), float(v12.get()), float(v13.get()), float(v14.get()), float(v15.get()), float(v16.get()), float(v17.get()), float(v18.get()), float(v19.get()), float(v20.get()), float(v21.get()), float(v22.get()), float(v23.get()), float(v24.get()), float(v25.get()), float(v26.get()), float(v27.get()), float(v28.get()), float(amount.get())]) #Getting input as string and converting it to series to provide for prediction
    pred = lr.predict([inp_s]) 
    r_pred.configure(text = "Credit Card Class Prediction is " + str(pred)[-2]) #Outputting the predicted class

window = tk.Tk() #Creating a tkinter root window
tk.Label(window, text="Credit Card Fraud Detection", justify = tk.LEFT, font = font.Font(size = 32, weight = 'bold')).grid(row = 0, column = 2, columnspan = 7)
vl1 = tk.Label(window, text = "PCA transformed Feature 1 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 1, column = 0)
v1 = tk.Entry(window)
v1.grid(row = 1, column = 1) #taking input from user
vl2 = tk.Label(window, text = "PCA transformed Feature 2 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 1, column = 2)
v2 = tk.Entry(window)
v2.grid(row = 1, column = 3)
vl3 = tk.Label(window, text = "PCA transformed Feature 3 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 1, column = 4)
v3 = tk.Entry(window)
v3.grid(row = 1, column = 5)
vl4 = tk.Label(window, text = "PCA transformed Feature 4 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 1, column = 6)
v4 = tk.Entry(window)
v4.grid(row = 1, column = 7)
vl5 = tk.Label(window, text = "PCA transformed Feature 5 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 2, column = 0)
v5 = tk.Entry(window)
v5.grid(row = 2, column = 1)
vl6 = tk.Label(window, text = "PCA transformed Feature 6 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 2, column = 2)
v6 = tk.Entry(window)
v6.grid(row = 2, column = 3)
vl7 = tk.Label(window, text = "PCA transformed Feature 7 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 2, column = 4)
v7 = tk.Entry(window)
v7.grid(row = 2, column = 5)
vl8 = tk.Label(window, text = "PCA transformed Feature 8 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 2, column = 6)
v8 = tk.Entry(window)
v8.grid(row = 2, column = 7)
vl9 = tk.Label(window, text = "PCA transformed Feature 9 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 3, column = 0)
v9 = tk.Entry(window)
v9.grid(row = 3, column = 1)
vl10 = tk.Label(window, text = "PCA transformed Feature 10 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 3, column = 2)
v10 = tk.Entry(window)
v10.grid(row = 3, column = 3)
vl11 = tk.Label(window, text = "PCA transformed Feature 11 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 3, column = 4)
v11 = tk.Entry(window)
v11.grid(row = 3, column = 5)
vl12 = tk.Label(window, text = "PCA transformed Feature 12 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 3, column = 6)
v12 = tk.Entry(window)
v12.grid(row = 3, column = 7)
vl13 = tk.Label(window, text = "PCA transformed Feature 13 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 4, column = 0)
v13 = tk.Entry(window)
v13.grid(row = 4, column = 1)
vl14 = tk.Label(window, text = "PCA transformed Feature 14 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 4, column = 2)
v14 = tk.Entry(window)
v14.grid(row = 4, column = 3)
vl15 = tk.Label(window, text = "PCA transformed Feature 15 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 4, column = 4)
v15 = tk.Entry(window)
v15.grid(row = 4, column = 5)
vl16 = tk.Label(window, text = "PCA transformed Feature 16 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 4, column = 6)
v16 = tk.Entry(window)
v16.grid(row = 4, column = 7)
vl17 = tk.Label(window, text = "PCA transformed Feature 17 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 5, column = 0)
v17 = tk.Entry(window)
v17.grid(row = 5, column = 1)
vl18 = tk.Label(window, text = "PCA transformed Feature 18 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 5, column = 2)
v18 = tk.Entry(window)
v18.grid(row = 5, column = 3)
vl19 = tk.Label(window, text = "PCA transformed Feature 19 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 5, column = 4)
v19 = tk.Entry(window)
v19.grid(row = 5, column = 5)
vl20 = tk.Label(window, text = "PCA transformed Feature 20 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 5, column = 6)
v20 = tk.Entry(window)
v20.grid(row = 5, column = 7)
vl21 = tk.Label(window, text = "PCA transformed Feature 21 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 6, column = 0)
v21 = tk.Entry(window)
v21.grid(row = 6, column = 1)
vl22 = tk.Label(window, text = "PCA transformed Feature 22 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 6, column = 2)
v22 = tk.Entry(window)
v22.grid(row = 6, column = 3)
vl23 = tk.Label(window, text = "PCA transformed Feature 23 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 6, column = 4)
v23 = tk.Entry(window)
v23.grid(row = 6, column = 5)
vl24 = tk.Label(window, text = "PCA transformed Feature 24 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 6, column = 6)
v24 = tk.Entry(window)
v24.grid(row = 6, column = 7)
vl25 = tk.Label(window, text = "PCA transformed Feature 25 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 7, column = 0)
v25 = tk.Entry(window)
v25.grid(row = 7, column = 1)
vl26 = tk.Label(window, text = "PCA transformed Feature 26 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 7, column = 2)
v26 = tk.Entry(window)
v26.grid(row = 7, column = 3)
vl27 = tk.Label(window, text = "PCA transformed Feature 27 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 7, column = 4)
v27 = tk.Entry(window)
v27.grid(row = 7, column = 5)
vl28 = tk.Label(window, text = "PCA transformed Feature 28 : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 7, column = 6)
v28 = tk.Entry(window)
v28.grid(row = 7, column = 7)
al = tk.Label(window, text = "Amount : ", justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')).grid(row = 8, column = 0)
amount = tk.Entry(window)
amount.grid(row = 8, column = 1)
button = tk.Button(window, text = "Predict", width = 15, command = predict_class)
button.grid(row = 9, column = 3)
r_pred = tk.Label(window, justify = tk.LEFT, font = font.Font(size = 11, weight = 'bold')) #display box for outut
r_pred.grid(row = 10, column = 2, columnspan = 7)
tk.Label(window, text = "Confusion Matrix is:") # Creating a display box for graph
fig = plt.figure()
graph = fig.add_subplot(111)
sns.heatmap(cm, annot = True, fmt = "g", cmap = "Blues").set_title("Accuracy score: " + str(acsc)) #heatmap graph for visualizing confusion matrix
canvas = FigureCanvasTkAgg(fig, master = window)  #creating a drawing area for graph in tkinter window
canvas.draw()
canvas.get_tk_widget().grid(row = 11, column = 0, columnspan = 7)
window.mainloop()
    
