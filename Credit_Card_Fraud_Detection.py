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
def predict_class(event):
    inp_s = pd.Series(map(float, inp.get().split())) #Getting input as string and converting it to series to provide for prediction
    pred = lr.predict([inp_s]) 
    r_pred.configure(text = "Credit Card Class Prediction is " + str(pred)[-2]) #Outputting the predicted class

window = tk.Tk() #Creating a tkinter root window
tk.Label(window, text="Credit Card Fraud Detection", justify= tk.CENTER, font = font.Font(size = 32,weight='bold')).pack()
inp = tk.Entry(window) #taking input from user
inp.bind("<Return>",predict_class) #binding it with predict_class function
inp.pack()
r_pred = tk.Label(window) #display box for outut
r_pred.pack()
tk.Label(window, text = "Confusion Matrix is:") # Creating a display box for graph
fig = plt.figure()
graph = fig.add_subplot(111)
sns.heatmap(cm, annot = True, fmt = "g", cmap = "Blues").set_title("Accuracy score: " + str(acsc)) #heatmap graph for visualizing confusion matrix
canvas = FigureCanvasTkAgg(fig, master = window)  #creating a drawing area for graph in tkinter window
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
window.mainloop()
    
