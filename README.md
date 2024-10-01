Machine Learning Stock sentiment Analysis comparing 14 models with Python

Abstract
Stock sentiment analysis using news headlines machine learning
News headlines are one of the most important factors in stock market that influence the most.
News headlines and stock market price have a direct relation. News headline plays a major role in
the stock price fluctuation. In this research, sentiment analysis has done on more than 100000 news
headlines and predict the whether the stock price will rise up or down using machine learning. 7
Machine learning algorithms and 2 natural language processing techniques at their default
parameter are used to get the predictions. A high level of programming techniques and 14 different
machine learning algorithm’s combinations were experimented to get best results. Seven powerful
machine learning algorithm with 2 natural language processing technique, in total 14 combination
were used in this study. In this thesis there were analyzed news headlines data and made
predication whether the stock price would increase or decrease using machine learning techniques.
The stock price goes up and down based on daily news headlines. So there is direct relation
between stock prices a news headlines. Using machine learning algorithms and Python
programming language it was made 14 models, which were able to predict whether stock price
would go up or down. This study shows the effect of emotion sentiment of financial news to the
stock market prices.
Keywords: Machine Learning, Sentiment Analysis, Python, Natural language processing,
classification algorithms, sklearn


----------------------------------------------------------------------------------------------------------

Python code


import pandas as pd
import numpy as np
import seaborn as sb
df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")
type(df)
df
51
sb.heatmap(df.isnull())
df.info()
df=df.dropna()
df.info()
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']
# Removing punctuations
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)
# Convertng headlines to lower case
for index in new_Index:
 data[index]=data[index].str.lower()
data.head(1)
' '.join(str(x) for x in data.iloc[1,0:25])
headlines = []
for row in range(0,len(data.index)):
 headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
len(headlines)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
## implement BAG OF WORDS
countvector=CountVectorizer()
traindataset=countvector.fit_transform(headlines)
type(traindataset)
## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# implement RandomForest Classifier
randomclassifier=RandomForestClassifier()
randomclassifier.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27])) #join all
test data headlines
test_dataset = countvector.transform(test_transform) #perform countvectrozer for
test data
predictions = randomclassifier.predict(test_dataset) #give test data to random
forest,did prediction and store into prediction
52
## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
from sklearn.naive_bayes import MultinomialNB
naive=MultinomialNB()
naive.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = naive.predict(test_dataset)
type(predictions)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
from sklearn.tree import DecisionTreeClassifier
test_decisiontree_classifier=DecisionTreeClassifier()
test_decisiontree_classifier.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = test_decisiontree_classifier.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
#gradient boosting classifier
from sklearn import ensemble
gd_clf=ensemble.GradientBoostingClassifier()
gd_clf.fit(traindataset,train['Label'])
53
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = gd_clf.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier()
knn_clf.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = knn_clf.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = lr_clf.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
from sklearn.svm import SVC
sv_clf= SVC()
sv_clf.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
54
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = sv_clf.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
## implement TF-IDF
tfidfVectorizer=TfidfVectorizer()
traindataset=tfidfVectorizer.fit_transform(headlines)
# implement RandomForest Classifier for TF-IDF
randomclassifier=RandomForestClassifier()
randomclassifier.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfVectorizer.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
naive=MultinomialNB()
naive.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfVectorizer.transform(test_transform)
predictions = naive.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
test_decisiontree_classifier=DecisionTreeClassifier()
test_decisiontree_classifier.fit(traindataset,train['Label'])
55
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfVectorizer.transform(test_transform)
predictions = test_decisiontree_classifier.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
gd_clf=ensemble.GradientBoostingClassifier()
gd_clf.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfVectorizer.transform(test_transform)
predictions = gd_clf.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
knn_clf=KNeighborsClassifier()
knn_clf.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfVectorizer.transform(test_transform)
predictions = knn_clf.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
lr_clf = LogisticRegression()
lr_clf.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfVectorizer.transform(test_transform)
predictions = lr_clf.predict(test_dataset)
56
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
sv_clf= SVC()
sv_clf.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
 test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfVectorizer.transform(test_transform)
predictions = sv_clf.predict(test_dataset)
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)

-------------------------------------------------------------------------------------------------------------------
 

 Conclusion of the study
This research can lead to few conclusions, which are as follows:
1. Accuracy score is the best measure to measure the accuracy of machine learning model,
2. Precision is the second-best priority as per this research's type,
3. F1 score and at least recall are important measures for this research.
43
Based on accuracy measure top 3 models were chosen:
1 Random forest with Bag of Words
2 Naïve Bayes with Bag of Words
3 SVM with TF-IDF
Based on precision top 3 models were chosen:
1 SVM with TF-IDF
2 Random forest with Bag of Words
3 Naïve Bayes with Bag of Words
Based on the F1 score vale top 3 models
1 Random forest with Bag of Words
2 Naïve Bayes with Bag of Words
3 SVM with TF-IDF
Based on Recall vale top 3 models
1 Random forest with Bag of Words
2 Naïve Bayes with Bag of Words
3 SVM with TF-IDF
Based on the results of this research it can be concluded that Random forest with Bag of Words
was the best performing model, the second-best model used SVM with TF-IDF, and the third
was Naïve Bayes with Bag of Words. So in this study where 7 machine learning algorithms
with their default parameter with 2 texts to vector technique, in total from 14 combinations of
44
models, Random forest with Bag of Words found the best combination for solving this research
problem

---------------------------------------------------------------------------------------------------------------------------------------------------

