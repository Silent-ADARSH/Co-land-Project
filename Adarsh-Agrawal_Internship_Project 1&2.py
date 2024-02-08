#!/usr/bin/env python
# coding: utf-8

# # News_Classification Project 1
# 

# In[1]:


pip install nltk


# In[2]:


import nltk


# In[3]:


nltk.download()


# In[4]:


import pandas as pd


# In[5]:


fake= pd.read_csv("Fake.csv")
genuine= pd.read_csv("True.csv")


# In[6]:


display(fake.info())


# In[7]:


display(genuine.info())


# In[8]:


display(genuine.head(10))


# In[9]:


display(fake.subject.value_counts())


# In[10]:


fake['target']=0
genuine['target']=1


# In[11]:


display(genuine.head(10))


# In[12]:


display(fake.head(10))


# In[13]:


data=pd.concat([fake,genuine],axis=0)


# In[14]:


data=data.reset_index(drop=True)


# In[15]:


data=data.drop(['subject','date','title'],axis=1)


# In[16]:


print(data.columns)


# ## Tokenization

# In[17]:


from nltk.tokenize import word_tokenize


# In[18]:


from nltk.tokenize import word_tokenize


# In[19]:


data['text']=data['text'].apply(word_tokenize)


# In[20]:


import nltk
nltk.download('punkt')


# In[21]:


data['text']=data['text'].apply(word_tokenize)


# ## STEMMING

# In[ ]:


print(data.head(10))


# In[ ]:


from nltk.stem.snowball import SnowballStemmer
porter =SnowballStemmer("english")


# In[ ]:


def stem_it(text):
    return [porter.stem(word) for word in text]


# In[ ]:


data['text']=data['text'].apply(stem_it)


# In[ ]:


print(data.head(10))


# ## STOPWORD REMOVAL

# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


def stop_it(t):
    dt=[word for word in t if len(word)>2]
    return dt


# In[ ]:


data['text']=data['text'].apply(stop_it)


# In[ ]:


print(data.head(10))


# In[ ]:


data['text']=data['text'].apply(' '.join)


# ## Splitting 
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data['text'],data['target'])
display(X_train.head())
print('\n')
display(y_train.head())


# ## Vectorization

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


my_tfidf = TfidfVectorizer(max_df=0.7)

tfidf_train = my_tfidf.fit_transform(X_train)
tfidf_test = my_tfidf.transform(X_test)


# In[ ]:


print(tfidf_train)


# ## LogisticRegression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


model_1 = LogisticRegression(max_iter=900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
cr1 = accuracy_score(y_test,pred_1)
print(cr1*100)


# ## PassiveAggressiveClassifier

# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)


# In[ ]:


y_pred = model.predict(tfidf_test)
accscore = accuracy_score(y_test, y_pred)
print('The accuray score predivtion is  ',accscore*100)


# In[ ]:





# ### End of project 1

# # Landmark Detection Project 2nd

# In[27]:


pip install tensorflow


# In[23]:


import numpy as np
import keras
import cv2
from matplotlib import pyplot as plt
import os
import random 
from PIL import Image
import pandas as pd


# In[24]:


df = pd.read_csv("train.csv")
base_path = "./images"


# In[25]:


df


# In[26]:


samples = 20000
df = df.loc[df["id"].str.startswith('00' , na=False), :]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)


# In[27]:


num_classes


# In[28]:


num_data


# In[29]:


data = pd.DataFrame(df["landmark_id"].value_counts())

data.reset_index(inplace=True)
data.head()


# In[30]:


data.tail()


# In[31]:


data.columns=["landmark_id","count"]


# In[32]:


data["landmark_id"].describe()


# In[33]:


plt.hist(data["count"], 100, range = (0,945), label = "test")


# In[34]:


data["count"].between(0,945).sum()


# In[35]:


data["count"].between(0,5).sum()


# In[36]:


data["count"].between(5, 10).sum()


# In[37]:


plt.hist(df["landmark_id"], bins=df["landmark_id"].unique())


# In[38]:


#Training of model
from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()
lencoder.fit(df["landmark_id"])


# In[39]:


df.head()


# In[40]:


def encode_label(lb1):
    return lencoder.transform(lb1)


# In[41]:


def decode_label(lb1):
    return lencoder.inverse_transform(lb1)


# In[56]:


def get_image_from_number(num, df):
    fname, label = df.iloc[num, :]
    fname = fname + '.jpg'
    f1, f2, f3, fname = fname[0], fname[1], fname[2], fname
    path = os.path.join(fname)
    final_path = os.path.join(base_path, path)
    print(final_path)
    im = cv2.imread(final_path)
    return im, label


# In[43]:


print("4 sample images from random classes ")
fig = plt.figure(figsize=(16,16))
for i in range(1,5):
#     ri = random.choices(os.listdir(base_path), k=3)
#     folder = base_path + '/' + ri[0] + '/' + ri[2]
    random_img = random.choice(os.listdir(base_path))
    img = np.array(Image.open(base_path+'/'+random_img))
    fig.add_subplot(1,4,i)
    plt.imshow(img)
    plt.axis('off')
plt.show()                         


# In[44]:


from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.models import Sequential
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# In[45]:


#Parameters
learning_rate = 0.0001
decay_speed   = 1e-6
momentum      = 0.09
loss_function = "sparse_categorical_crossentropy"
source_mode = VGG19(weights=None)
drop_layer = Dropout(0.5)
drop_layer2 = Dropout(0.5)


# In[46]:


model = Sequential()
for layer in source_mode.layers[:-1]:
    if layer == source_mode.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Dense(num_classes, activation="softmax"))
model.summary()


# In[47]:


optim1 = keras.optimizers.RMSprop(lr=learning_rate)
model.compile(optimizer=optim1,
              loss=loss_function,
              metrics=["accuracy"])


# In[48]:


def image_reshape(im, target_size):
    
    return cv2.resize(im, target_size)


# In[49]:


def get_batch(dataframe, start, batch_size):
    image_array = []
    label_array = []
    
    end_img = start + batch_size
    if(end_img) > len(dataframe):
        end_img = len(dataframe)
    
    for idx in range(start, end_img):
        n = idx
        im, label = get_image_from_number(n, dataframe)
        im = image_reshape(im, (224, 224)) / 255.0
        image_array.append(im)
        label_array.append(label)
    
    label_array = encode_label(label_array)
    
    return np.array(image_array), np.array(label_array)


# In[50]:


batch_size = 16
epoch_shuffle = True
weight_classes = True
epochs = 1

#split data
train, val = np.split(df.sample(frac=1),[int(0.8*len(df))])
print(len(train))
print(len(val))


# In[57]:


image, l = get_image_from_number(0, train)
print(image)


# In[216]:


for e in range(epochs):
    print("Epoch :" + str (e+1) + "/"+ str(epochs))
    if epoch_shuffle:
        train = train.sample(frac = 1)
        for it in range(int(np.ceil(len(train)/batch_size))):
            X_train, y_train = get_batch(train, it * batch_size, batch_size)
            
        
            print("Image paths:", X_train)
            model.train_on_batch(X_train, y_train)

model.save("Model")


# In[217]:


# Test
batch_size = 16

errors = 0
good_preds = []
bad_preds = []

for it in range(int(np.ceil(len(val)/batch_size))):
    X_val, y_val = get_batch(val, it*batch_size, batch_size)
    
    
    print("Validation Image paths:", X_val)
    
    result = model.predict(X_val)
    cla = np.argmax(result, axis=1)
    for idx, res in enumerate(result):
        if cla[idx] != y_val[idx]:
            errors = errors + 1
            bad_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])
        else:
            good_preds.append([batch_size*it + idx, cla[idx], res[cla[idx]]])

model.save("Model")


# In[218]:


good_preds = np.array(good_preds)
good_preds = np.array(sorted(good_preds, key = lambda x: x[2], reverse=True))


# In[209]:


fig = plt.figure(figsize=(16, 16))
for i in range(1, 6):
    if i < len(good_preds):
        n = int(good_preds[i][0])
        img, lbl = get_image_from_number(n, val)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(1, 5, i)
        plt.imshow(img)
        lbl2 = np.array(int(good_preds[i][1])).reshape(1, 1)
        sample_cnt = list(df.landmark_id).count(lbl)
        plt.title("Label: " + str(lbl) + "\nClassified as: " + str(decode_label(lbl2)) + "\nSamples in class " + str(lbl) + ": " + str(sample_cnt))
        plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:




