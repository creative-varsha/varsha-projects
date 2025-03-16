import warnings
warnings.filterwarnings('ignore')
import pandas as pd  
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.preprocessing import LabelEncoder

main = tk.Tk()
main.title("NOVEL ML  APPROACH FOR ANDROID MALWARE DETECTION BASED ON THE CO-EXISTENCE OF FEATURES") 
main.geometry("1400x1400")

font = ('times', 16, 'bold')
title = Label(main, text='NOVEL ML  APPROACH FOR ANDROID MALWARE DETECTION BASED ON THE CO-EXISTENCE OF FEATURES', font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=108)
title.place(x=0, y=2)

font1 = ('times', 12, 'bold')
text = Text(main, height=14, width=110)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

# Upload Function
def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    # Fill missing values with 0
    df.fillna(0, inplace=True)  # Fill missing values with 0
    
    # Convert non-numeric values to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(df.head()) + "\n")

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=20)
uploadButton.place(x=50, y=400)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50, y=450)

def graph():
    target_count = df.malware.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    text.insert(END, 'Class 0:', target_count[0])
    text.insert(END, 'Class 1:', target_count[1])

    sns.countplot(x='malware', data=df)
    plt.title("Comparison graph of malware ")
    plt.show()

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test
    X = df.drop(columns=['malware'])  # Assuming 'malware' is the target column
    y = df['malware']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n\n")
    text.insert(END, str(X))
    text.insert(END, str(y))
    return X, y, X_train, X_test, y_train, y_test

# Decision Tree function
def run_decision_tree():
    text.delete('1.0', END)
    global dt_classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    accuracy = dt_classifier.score(X_test, y_test)
    text.insert(END, "Decision Tree Accuracy: " + str(accuracy) + "\n")
    return accuracy

# Random Forest function
def run_random_forest():
    text.delete('1.0', END)
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    accuracy = rf_classifier.score(X_test, y_test)
    text.insert(END, "Random Forest Accuracy: " + str(accuracy) + "\n")
    return accuracy

# K-Nearest Neighbors function
def run_knn():
    text.delete('1.0', END)
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    accuracy = knn_classifier.score(X_test, y_test)
    text.insert(END, "K-Nearest Neighbors Accuracy: " + str(accuracy) + "\n")
    return accuracy

# Naive Bayes function
def run_naive_bayes():
    text.delete('1.0', END)
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    accuracy = nb_classifier.score(X_test, y_test)
    text.insert(END, "Naive Bayes Accuracy: " + str(accuracy) + "\n")
    return accuracy

# Function to predict results using Decision Tree model
def predict():
    # Open file dialog to select CSV file
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Read the CSV file
        data_to_predict = pd.read_csv(file_path)
        
        # Fill missing values with 0 and encode non-numeric values
        data_to_predict.fillna(0, inplace=True)
        label_encoder = LabelEncoder()
        for column in data_to_predict.columns:
            if data_to_predict[column].dtype == 'object':
                data_to_predict[column] = label_encoder.fit_transform(data_to_predict[column])
        
        # Predict using the Decision Tree model
        predictions = dt_classifier.predict(data_to_predict)
        result = "Detected" if predictions[0] == 1 else "Not Detected"
        
        # Display predictions
        text.delete('1.0', END)
        text.insert(END, "Prediction Result:\n")
        text.insert(END, result)

# Button for Predict
predictButton = Button(main, text="Predict", command=predict, bg="orange", width=20)
predictButton.place(x=650, y=600)
predictButton.config(font=font1)

# Buttons for each algorithm
graphButton = Button(main, text="Graph", command=graph, bg="light green", width=20)
graphButton.place(x=40, y=500)
graphButton.config(font=font1)

splitButton = Button(main, text="Split Dataset", command=splitdataset, bg="turquoise", width=20)
splitButton.place(x=250, y=500)
splitButton.config(font=font1)

dtButton = Button(main, text="Decision Tree", command=run_decision_tree, bg="coral", width=20)
dtButton.place(x=450, y=500)
dtButton.config(font=font1)

rfButton = Button(main, text="Random Forest", command=run_random_forest, bg="gold", width=20)
rfButton.place(x=50, y=600)
rfButton.config(font=font1)

knnButton = Button(main, text="KNN", command=run_knn, bg="violet", width=20)
knnButton.place(x=250, y=600)
knnButton.config(font=font1)

nbButton = Button(main, text="Naive Bayes", command=run_naive_bayes, bg="green", width=20)
nbButton.place(x=450, y=600)
nbButton.config(font=font1)


main.config(bg='#F08080')
main.mainloop()
