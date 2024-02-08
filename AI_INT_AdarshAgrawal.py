#!/usr/bin/env python
# coding: utf-8

# # Google LandMark Detection 

# In[2]:


#consist of visual analysis based on neural networks ....
#this is the project of google landmark detection...
#Used 'google landmark detection dataset v2'
#CNN is very suited for image analysis


# ### Steps invovled
# - data collection from 'https://github.com/cvdfoundation/google-landmark'
# - preprocessing 
#    - resizing images
#    - normalization of pixels
#    - augmentations
#    - rotating
#    - flipping
#    - scaling
#    - encoding
# - model creation 
#    - Sequential layer
#    - input layer
#    - hidden layer
#        - CNN
#        - Max Polling
#        - Full connected layers
# - split the data
# - train data
# - test the model
#     -metric,accuracy,f1 score
#     - model used is vgg19

# In[3]:


#Importing begins!
#tensorflow==2.8.0 keras=2.8.0, pillow, sklearn(encoder), open cv


# In[4]:


get_ipython().system('pip install keras==2.8.0')


# In[5]:


get_ipython().system('pip install --upgrade tensorflow')


# In[6]:


import numpy as np
import pandas as pd
from tensorflow import keras 
import cv2
from matplotlib import pyplot as plt
import os
import random
from PIL import Image


# In[7]:


df = pd.read_csv("train.csv")


# In[8]:


df


# In[9]:


df1 = pd.read_csv("train(1).csv")


# In[10]:


df1


# In[11]:


df = df.loc[df["id"].str.startswith(('b1','00'), na=False), :]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)


# In[12]:


num_classes


# In[13]:


num_data


# In[14]:


data = pd.DataFrame(df["landmark_id"].value_counts())
data.reset_index(inplace = True)


# In[15]:


data.head()


# In[16]:


data.tail()


# In[17]:


data.columns = ["landmark_id", "count"]


# In[18]:


data


# In[19]:


data['count'].describe()


# In[20]:


plt.hist(data['count'], 100
        ,range = (0,64), label = 'test')


# In[21]:


data['count'].between(0,5).sum()


# In[22]:


data['count'].between(5,10).sum()


# In[23]:


data['count'].between(10,15).sum()


# In[24]:


data['count'].between(15,20).sum()


# In[25]:


plt.hist(df["landmark_id"], bins=df["landmark_id"].unique())


# In[26]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelencoder.fit(df["landmark_id"])


# In[27]:


df


# In[28]:


filtered_df = df[df['id'].str.startswith('000')]


# In[29]:


df = filtered_df


# In[30]:


#transform before feeding the data
def encode_label(label):
    return labelencoder.transform(label)


# In[31]:


def decode_label(label):
    return labelencoder.inverse_transform(label)


# In[32]:


import os
import cv2

def get_img_from_num(num, df):
    if 0 <= num < len(df):  # Check if the index is within the DataFrame length
        fname, label = df.iloc[num, :]
        print(fname)
        f1, f2, f3 = fname[:3]  # Extracting first three characters
        path = os.path.join("./images", f"images_{f1}{f2}{f3}", f1, f2, f3, f"{fname}.jpg")

        try:
            img = cv2.imread(path)
            if img is not None:
                print(img)
                return img, label
            else:
                print(f"Error: Unable to read image at path: {path}")
                return None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None
    else:
        print("Error: Index out of bounds or DataFrame length is insufficient.")
        return None, None

# Example usage:
# img, label = get_img_from_num(3, your_dataframe)


# In[33]:


fig = plt.figure(figsize=(16, 16))

for i in range(1, 5):
    rimg = random.choices(os.listdir("./images/images_000/0/0"), k=3)
    print(rimg)
    
    folder = "./images/images_000/0/0" + "/" + rimg[2]
    random_img = random.choice(os.listdir(folder))
    
    iimg = np.array(Image.open(folder + "/" + random_img))
    fig.add_subplot(1, 4, i)
    plt.imshow(iimg) 
    plt.axis('off')

plt.show()


# ### Model Building

# In[34]:


from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras import Sequential


# In[35]:


#Parameters 
learning_rate = 0.001
decay_speed = 1e-6
momentum = 0.09
loss_function = "sparse_categorical_crossentropy"
source_model = VGG19(weights=None)
drop_layer = Dropout(0.5)
drop_layer2 = Dropout(0.5)


# In[36]:


model = Sequential()
for layer in source_model.layers[:-1]:
    if layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Dense(num_classes, activation="softmax"))
model.summary()


# #### Model Training

# In[37]:


model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate),
              loss=loss_function,
              metrics=['accuracy'])


# In[38]:


def image_reshape(img, size):
    return cv2.resize(img, size)


# In[39]:


def get_batch(dataframe, start, batch_size):
    image_array = []
    label_array = []
    
    end_image = start + batch_size
    if(end_image) > len(dataframe):
        rend_image = len(dataframe)
        
    for idx in range(start, end_image):
        n = idx
        img, label = get_img_from_num(n, dataframe)
        img = image_reshape(img, (224, 224)) / 255.0
        image_array.append(img)
        label_array.append(label)
    label_array = encode_label(label_array)
    
    return np.array(image_array), np.array(label_array)


# In[40]:


batch_size = 64
epoch_shuffle = True
weight_classes = True
epochs = 1

#Splitting the data with 80% of length
train, val = np.split(df.sample(frac=1),[int(0.8*len(df))])
print(len(train))
print(len(val))


# In[41]:


df


# In[42]:


import cv2
for e in range(epochs):
    print("Epoch : " + str(e+1) + "/" + str(epochs))
    if epoch_shuffle:
        train = train.sample(frac = 1)
    for it in range(int(np.ceil(len(train)/batch_size))):
        X_train, y_train = get_batch(train, it*batch_size, batch_size)
        model.train_on_batch(X_train, y_train)


# In[43]:


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


# In[44]:


good_preds = np.array(good_preds)
good_preds = np.array(sorted(good_preds, key = lambda x: x[2], reverse=True))


# In[45]:


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


# #### Name- Adarsh Agrawal
# #### Student Enrolled in AI with python Course

# ###### The End
