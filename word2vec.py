
# coding: utf-8

# In[69]:


import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[14]:


data = []
for line in open('complaints.json','r'):
    data.append(json.loads(line))


# In[15]:


titles = []
categories = []
for x in data:
    titles.append(x['title'])
    categories.append(x['category'])


# In[16]:


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin',binary=True)


# In[47]:


x = []
for title in titles:
    temp = np.zeros((len(model['the'])))
    count = 0
    title = title.split(' ')
    for word in title:
#         print(word)
        try:
            temp += model[word]
            count += 1
        except:
#             print(word + " not found")
            pass
#     print(count)
    temp = temp/(count+1)
    x.append(temp)


# In[64]:


label = {list(set(categories))[i]:i for i in range(10)}
# print(label)
y = [label[word] for word in categories]
label_names = list(set(categories))


# In[73]:


# print(label)
# print(categories[0],y[0])
# print(len(y),len(x))
# print(y[50:])


# In[70]:

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.33)

# clf = svm.LinearSVC()
clf = RandomForestClassifier(n_estimators=50)
# clf = svm.LinearSVC()
clf.fit(xtrain,ytrain)
ypred = clf.predict(xtest)

print ('Accuracy : ', accuracy_score(ypred, ytest))
print ('-------------------------------------------------------------')
print ('Confusion Matrix : ')
m = confusion_matrix(ypred, ytest)
top3 = {}
for i in range(len(m)):
    temp = [(x,y) for x,y in enumerate(m[i])]
    temp = sorted(temp,key=lambda x:x[1],reverse=True)
    temptop = [(label_names[temp[0][0]],temp[0][1]/sum(m[i])), (label_names[temp[1][0]], temp[1][1]/sum(m[i])), (label_names[temp[2][0]], temp[2][1]/sum(m[i]))]
    top3[label_names[i]] = temptop
    print(label_names[i], top3[label_names[i]])

print ('-------------------------------------------------------------')
print ('classification_report : ')
print (classification_report(ypred, ytest, target_names=label_names))