#!/usr/bin/env python
# coding: utf-8

# # AI PYTHON ASSIGNMENT
# 
# 
# ## PROJECT 1
# 

# Q.2.	Write a function which prints all the numbers divisible by 3 and 5
# 

# In[32]:


n=int(input("Enternumber:\n"))
a=n%15
if(a==0):
    print("{} is divisible by 3 and 5".format(n))
else:
    print("{} is not divisible by 3 and 5".format(n))


# Q.3.	Write a program to check whether a given letter is vowel or consonant

# In[22]:


l=['a','e','i','o','u']
l2=['A','E','I','O','U']
l3=["I"]
char=0
char=str(input("ENTER LETTER:\n"))
if char==l[0] or char==l[1] or char==l[2] or char==l[3] or char==l[4]:
    print("{} is a vowel".format(char))
elif char==l2[0] or char==l2[1] or char==l2[3] or char==l2[4]:
    print("{} is a vowel".format(char))
elif char==l3[0]:
    print("{} is a vowel".format(char))
else:
    print("{} is not a vowel".format(char))


# Q1.Take list of elements from the user and find the square root of each number in the list and store in it another list and print that list.

# In[3]:


import math
l=[]
x=int(input("Enter list size:\n"))

a=input("Enter numbers and separate them with single spaces:\n")
l=a.split()
l=[float(i) for i in l]

print("List:", l)

l2=[]
for a in l:
    l2.append(math.sqrt(a))

print("Square root of each element in the list:", l2)


# Q4.	Calculate the distance between any two characters given by user
# 
#    (Example distance between “a” and “d” is 3)
# 

# In[4]:


a= input("first character: ")
b= input("second character: ")

dist = abs(ord(b) - ord(a))
print(f"distance b/w {a} and {b} is {dist}.")


# Q.5.	Write a function which returns the number of vowels present in the given string

# In[5]:


def n(string):
    vowels="aeiouAEIOU"
    num = 0
    for char in vowels:
        if char in vowels:
            num += 1
    return num
string = "Hello This is python assignment."
vowelnumber= n(string)
print(f"the num of vowels in '{string}' is {vowelnumber}.")


# Q6.	Print all the alphabets by using loop and ascii code

# In[10]:


print("Small alphabets:")
for i in range(97,123):
    print(chr(i), end=" ")
    
print("\nCapital alphabets:")
for j in range(65, 91):
    print(chr(j), end=" ")


# Q7.	write a program find the sum of all the even numbers of the list?

# In[17]:


def p(numbers):
    sum=0
    for num in numbers:
        if num % 2 == 0:
            sum += num
    return sum
a=input("Enter nums separated by commas:")
b=[int(n) for n in a.split(",")]

sumofevennumbers = p(b)
print(f"sum of all even nums in {b} is {sumofevennumbers}.")


# Q8.	Write a program for print the squares of all the numbers, except for factors of 3

# In[22]:


x=int(input("Enter the last number for range of numbers you want:\n"))
for i in range(x):
    if i % 3 == 0:
        continue
    print(i**2)


# Q.9.	Take 2 strings from user and then replace all the A’s with a’s and then concatenate the 2 strings and print

# In[24]:


str1 = input("1st str elements:")
str2 = input("2nd str elements:")

str1 = str1.replace('A', 'a')
str2 = str2.replace('A', 'a')

answer = str1 + str2
print("Final str: ", answer)


# Q10.	 write a program to get a list of odd number from the list of numbers given by user (use list comprehension)

# In[30]:


l1 = input("Enter num with commas: ")
l1=l1.split(",")
l1=[int(n) for n in l1]

oddnumbers = [n for n in l1 if n % 2 != 0]

print("list of odd numbers is:", oddnumbers)


# Q11.	write a program to print lower when you have upper letter in string and vice versa
# 
# 
#         (if your input is “aBcD” your output should be “AbCd”)
# 

# In[31]:


str = input("Enter str: ")
str2 = ""
for i in str:
    if i.isupper():
        str2 += i.lower()
    else:
        str2 += i.upper()
        
print(str2)


# ### END OF PROJECT 1

# ## PROJECT 4
# 

# Q.>>
# 1. Study about boston housing price classifier/prediction, make a similar model.

# In[5]:


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


# ####  Fetch Model

# In[6]:


boston = fetch_openml(name='boston')

data = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.DataFrame(boston.target, columns=['MEDV'])

print(data.head())
print(target.head())


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state= 42)
linear_reg_model = linear_model.LinearRegression()
print(x_train)
linear_reg_model.fit(x_train, y_train)


print(linear_reg_model.coef_)
print(data.dtypes)
print(target.dtypes)
print("x train: " , x_train.isnull().sum())
print(y_train.isnull().sum())




# In[13]:


data['CHAS'] = pd.to_numeric(data['CHAS'])

print(x_test.isnull().sum())


y_pred = linear_reg_model.predict(x_test)


print(y_pred)
print(y_test)

mse = mean_squared_error(y_test, y_pred)
print(mse)

scores = cross_val_score(linear_reg_model, data, target, cv=10)
print("Cross-validation scores: ", scores)
print("Avg scores: ", np.mean(scores))


# ### END OF PROJECT 4

# ## PROJECT 3

# Q.>
# 1. Study about haar cascade algoritms. 
# 2. Try to import haarcascade algoritms for face detection in ide (.xml).
# 3. Prepare a model which will detect the face and boundry it using blue box (rectangle). 
# 

# In[56]:


import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# img = cv2.imread("C:/Users/cheen/anaconda3/Lib/site-packages/cv2/data/img.jpg")
img = cv2.imread("C:/Users/cheen/OneDrive/Desktop/robert.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
eyes = eye_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 2)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x-10, y-10), (x + w+10, y + h+10), (0, 255, 0), 2)
    

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.imshow(img)


# ### END OF PROJECT 3

# ## PROJECT 2

# Q>>
# 1. Implement Iris Classifier project. 
# 2. Get the data from local system not from web. 
# 3. Try to evaluate the performance of the model by changing various parameters ex: Split rario etc
# 4. Use other algoritms and evaluate the performance of each algoritms in this dataset.
# 

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

columns = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Class_labels']

df = pd.read_csv("C:\\Users\\cheen\\Downloads\\iris-flower-classification-project\\iris.data", names=columns)
df.head()

print(df.head())

df.describe()

sns.pairplot(df, hue='Class_labels')

df.info()
data = df.values

X=data[:,0:4]
Y=data[:,4]
#print(X)
#print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(X_train)

from sklearn.svm import SVC

model_svc = SVC()
model_svc.fit(X_train,y_train)
prediction1 = model_svc.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction1)*100)
for i in range(len(prediction1)):
    print(y_test[i],prediction1[i])

    
from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(X_train,y_train)



# In[12]:


print(y_train)


# In[20]:


pred2 = model_LR.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred2)*100)
for i in range(len(prediction1)):
    print(y_test[i],prediction1[i])


# In[28]:


from sklearn.tree import DecisionTreeClassifier
model_DTC=DecisionTreeClassifier()
model_DTC.fit(X_train,y_train)


# In[27]:


pred3 = model_svc.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred3))


# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pred2))


# ### END OF PROJECT 2

# # END OF ALL PROJECT 

# # NAME OF STUDENT = ADARSH AGRAWAL
# # COLLEGE = SGSITS
# # ENROLLED IN COURSE = AI WITH PYTHON
# # SUBMISSION OF FINAL ASSIGNMENT

# In[ ]:




