
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project 2: Exploring the dataset of Student Intervention System

# In this notebook I'm trying to visualize the dataset in order to get a better understanding of it.

# In[1]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"


# In[2]:

print "number of instances:\t" + str(student_data.shape[0]) + "\nnumber of features:\t" + str(student_data.shape[1])
print student_data.columns
student_data.describe()
student_data.dtypes


# In[3]:

def occurenceplot(col):
    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes

    width = 0.4
    #col= "absences"
    #TODO: merge indexes
    all_values = pd.DataFrame(np.unique(student_data.absences))
    all_values.set_index(0)
    
    # calculate the relative occurences
    passed = student_data_passed[[col]].apply(pd.value_counts)/student_data_passed.shape[0]
    failed = student_data_failed[[col]].apply(pd.value_counts)/student_data_failed.shape[0]
    
    passed.sort_index().plot(kind='bar', color='red', ax=ax, width=width, position=1)
    failed.sort_index().plot(kind='bar', color='blue', ax=ax, width=width, position=0)
    
    ax.legend(["passed", "failed"]);
    ax.set_ylabel('Amount')
    ax.set_xlabel(col)
    plt.xticks(x=np.unique(student_data[col]),rotation='horizontal')
    plt.gca().set_xlim(left=-0.5)
    plt.show()
    return


# In[4]:

student_data_passed = student_data[student_data.passed=='yes']
student_data_failed = student_data[student_data.passed=='no']

x = student_data_failed[["absences"]].apply(pd.value_counts)/student_data_passed.shape[0]
print np.unique(student_data_passed.absences)
print np.unique(student_data_failed.absences)
print np.unique(student_data.absences)
occurenceplot("absences")
#for column in student_data.columns: occurenceplot(column)


# In[5]:

import seaborn as sns
_ = sns.factorplot("sex", col="age", col_wrap=4,
                   data=student_data, hue='passed',
                   kind="count", size=3, aspect=.8)


# In[6]:

sns.pairplot(student_data, hue="passed")


# In[ ]:



