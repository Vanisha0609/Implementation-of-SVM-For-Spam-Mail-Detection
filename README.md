# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect a labeled dataset of emails, distinguishing between spam and non-spam.
2. Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.
3. Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF
4. Split the dataset into a training set and a test set.
5. Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.
6. Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.
7. Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.
8. Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.
9. Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vanisha Ramesh
RegisterNumber:  212222040174
*/

import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(1000000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
1. Result output

![image](https://github.com/Vanisha0609/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104009/62d896ad-11b7-43f2-a8a3-51dd63c339b0)

2.data.head()

![image](https://github.com/Vanisha0609/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104009/c8e74c40-02db-450a-9382-054b51fc0043)

3.data.info()

![image](https://github.com/Vanisha0609/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104009/45002392-c2eb-427f-91b7-7a83ae8eb9ec)

4.data.isnull().sum()

![image](https://github.com/Vanisha0609/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104009/ebf336f1-54c3-4613-9754-c42394097f3a)

5.Y_prediction value

![image](https://github.com/Vanisha0609/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104009/f30508ca-3390-460b-a2ab-a842cfc6bdcb)

6.Accuracy value

![image](https://github.com/Vanisha0609/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104009/ba89b7d4-5dc7-4891-a1ae-0a311bc375c7)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
