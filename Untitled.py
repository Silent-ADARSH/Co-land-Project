#!/usr/bin/env python
# coding: utf-8

# In[30]:


n = int(input())
for i in range(0,255):
    if n in range(0,255):
        print("It exist")
        break
    else:
        print("not in range")
        break


# In[33]:


n = tuple(range(0,256,25))
n


# In[34]:


n =  list(range(0,256,25))
x = n[5]
n[5] = n[6]
n[6] = x
print(n)


# In[36]:


a = range(1000,10000)
i = int(input())
if i != a:
    print("not there")
else:
    print("there")


# In[ ]:




