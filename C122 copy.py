import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = np.load('image.npz')['arr_0'] 
y = pd.read_csv("labels.csv")["labels"] 
print(pd.Series(y).value_counts()) 
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses=len(classes)
samplesperclasses=5
figure=plt.figure(figsize=(nclasses*2,(1+samplesperclasses*2)))
idxcls=0
for cls in classes:
    idxs=np.flatnonzero(y==cls)
    idxs=np.random.choice(idxs,samplesperclasses,replace=False)
    i = 0
    for idx in idxs:
        plt_idx=i*nclasses+idxcls+1
        p=plt.subplot(samplesperclasses, nclasses, plt_idx)
        p=sns.heatmap(np.reshape(X[idx],(22,30)),cmap=plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False)
        p=plt.axis('off')
        i+=1
    idxcls+=1
print(len(X))
print(len(X[0]))
xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=0, train_size=7500, test_size=2500)
xtrainscale=xtrain/255
xtestscale=xtest/255
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscale,ytrain)
ypred=clf.predict(xtestscale)
print('accuracy',accuracy_score(ytest,ypred))
cm=pd.crosstab(ytest,ypred,rownames=['actual'], colnames=['predicted'])
p=plt.figure(figsize=(10,10))
p=sns.heatmap(cm,annot=True,fmt='d',cbar=False)