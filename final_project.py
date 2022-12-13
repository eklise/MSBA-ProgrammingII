#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Emma O'Connor
# ### 12/05/22

# #### Q1: Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[222]:


import pandas as pd
import os
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# os.getcwd()
# os.chdir('C:\\Users\\eocon\\OneDrive\\Documents\\MSBA\\Programming 2 - Data Infrastructure')

s = pd.read_excel("social_media_usage.xlsx")

#s.head()

#s.shape


# ***

# #### Q2: Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[223]:


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# df = pd.DataFrame({'A': [86, 75, 1],
#                    'B': [1, 30, 9]})

#print(clean_sm(df))


# ***

# #### Q3: Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[224]:


s['sm_li'] = clean_sm(s['web1h'])
ss = s[['sm_li','income', 'educ2', 'par', 'marital', 'gender', 'age']]


# In[226]:


ss['income'] = np.where(ss['income'] <= 9, ss['income'], None)
ss['educ2'] = np.where(ss['educ2'] <= 8, ss['educ2'], None)
ss['par'] = np.where(ss['par'] <= 2, ss['par'], None)
ss['marital'] = np.where(ss['marital'] <= 6, ss['marital'], None)
ss['gender'] = np.where(ss['gender'] <= 2, ss['gender'], None)
ss['age'] = np.where(ss['age'] <= 98, ss['age'], None)

ss = ss.dropna()

#ss.head(10)


# In[227]:


# ## exploratory analysis
# ss['sm_li_cat'] = ss['sm_li'].astype('category')

# # income
# print(ss.groupby(['sm_li_cat'])['income'].mean())

# alt.Chart(ss).mark_area(
#     opacity=0.5,
#     interpolate='step'
# ).encode(
#     alt.X('income', bin=alt.Bin(maxbins=10)),
#     alt.Y('count()', stack=None),
#     alt.Color('sm_li_cat')
# ).properties(
#     title='Income Bracket Between Linkedin Users and Not'
# )


# # From the histogram above, we can see that the income distribuition for Linkedin users tends to be higher than non-users.

# # In[228]:


# # educ2

# alt.Chart(ss).mark_area(
#     opacity=0.5,
#     interpolate='step'
# ).encode(
#     alt.X('educ2', bin=alt.Bin(maxbins=10)),
#     alt.Y('count()', stack=None),
#     alt.Color('sm_li_cat')
# ).properties(
#     title='Education Level Between Linkedin Users and Not'
# )


# # Similarily, linkedin users tend to be more educated than non-users. 

# # In[230]:


# # par
# #alt.Chart(ss.groupby(["age", "marital"], as_index=False)["sm_li"].mean()).\
# #mark_circle().\
# #encode(x="age",
#  #     y="sm_li",
#  #     color=":N")

# print(ss.groupby(['sm_li_cat'])['par'].mean())


# # In[231]:


# # marital
# print(ss.groupby(['sm_li_cat'])['marital'].mean())


# # Parent/Non-Parent and Marital Status do not seem to be helpful in predicting whether or not someone is a linkedin user. 

# # In[243]:


# # gender
# ss['gender_cat'] = ss['gender'].astype('category')
# #alt.Chart(ss).mark_bar().encode(
#  #   alt.X('count()'),
#  #   alt.Y('gender_cat'),
#  #   alt.Color('sm_li_cat')
# #).properties(
# #    title='Gender and Linkedin Usage'
# #)


# alt.Chart(ss).transform_aggregate(
#     count='count()',
#     groupby=['sm_li_cat', 'gender_cat']
# ).transform_joinaggregate(
#     total='sum(count)',
#     groupby=['gender_cat']  
# ).transform_calculate(
#     frac=alt.datum.count / alt.datum.total
# ).mark_bar().encode(
#     x="gender_cat:O",
#     y=alt.Y('count:Q', stack="normalize", axis=alt.Axis(title="Percent", format="%")),
#     color='sm_li_cat:N',
#     tooltip=[
#         alt.Tooltip('count:Q', title="Count"),
#         alt.Tooltip('frac:Q', title="Percentage", format='.0%')
#     ]
# )


# # From the above figure, we can see that the proportion of males to females using Linkedin is not equal. In this dataset, 36% of the men are linkedin users compared to only 29% of the women (check out the tooltip). 

# # In[244]:


# # age
# alt.Chart(ss).mark_area(
#     opacity=0.5,
#     interpolate='step'
# ).encode(
#     alt.X('age', bin=alt.Bin(maxbins=10)),
#     alt.Y('count()', stack=None),
#     alt.Color('sm_li_cat')
# ).properties(
#     title='Distribution of Age Between Linkedin Users and Not'
# )


# The distribution of Linkedin users is narrower than non-users. Those who use Linkedin tend to be younger than those who do not. 

# ***

# #### Q4: Create a target vector (y) and feature set (X)

# In[205]:


y = ss['sm_li']
x = ss[['income','educ2','par','marital','gender','age']]


# ***

# #### Q5: Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[206]:


x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 120822)


# 1. x_train: This object contains 80% of the data and will be used to train our classification model in predicting the target (linkedin user y/n). The variables saved in x_train are all the features/predictor variables we will be using, but does not include the target.
# 2. x_test: This object contains 20% of the data and again, only included the features used to predict the target.
# 3. y_train: This object contains 80% of the data but only contains the target data.
# 4. y_test: Similarily, this object only contains target data. 20% of the data is save in this object.

# ***

# #### Q6: Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[207]:


# train classificaton model

# initialize algorithm
lr = LogisticRegression(class_weight='balanced')

# fit algorithm to training data
lr.fit(x_train, y_train)


# ***

# #### Q7:Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[208]:


# evaluate model
y_pred = lr.predict(x_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)


# From the confusion matrix above, we can see that we correctly predicted that 109 individuals were not linkedin users while 20 of the individuals who we predicted were not linkedin users actually were. Of the 120 individuals who we predicted were linkedin users, 57 of them were not and 63 of them were. In other words, our model made the correct predictions for 172 individuals and made an incorrect prediction 77 times.

# ***

# #### Q8: Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[209]:


# pd.DataFrame(confusion_matrix(y_test, y_pred),
#             columns=["Predicted negative", "Predicted positive"],
#             index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PuBu")


# ***

# #### Q9: Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[188]:


# # recall
# #63/(63+20)
# rec = cm[1,[1]]/(cm[1,[1]]+cm[1,[0]])
# print(f"Recall: {rec}")

# # precision
# pre = cm[1,[1]]/(cm[1,[1]]+cm[0,[1]])
# print(f"Precision: {pre}")

# # F1 score
# f1 = 2*((pre*rec)/(pre+rec))
# print(f"F1 score: {f1}")


# 1. Recall: The higher the recall, the lower the chance of missing positive cases. This evaluation metric would likely be prioritized in a situation where its important that a life-threatening malfunction in a product is caught before the product is approved, such as car manufacturing. 
# 2. Precision: This metric prioritizes minimizing incorrectly predicting positive cases. The classic example where this would be the goal is cancer diagnosis. The risk of incorrectly returning a positive diagnosis and undergoing uneccessary treatment is more could be very harmful.
# 3. F1 score: This evaluation metric takes into account both recall and precision in its evaluation of the model. It is a weighted average of both and is best for cases where there isn't any particular emphasis placed in either reducing false positives of reducing false negatives. 
# 
# In our case, I would say that an F1 score would be the best evaluation metric to use because there isn't any particular risk in either a false positive or false negative in this case. Our F1 score is not very high, indicating that the model is not very good overall. We can see that it is more prone to false positives looking at the low precision rate.
# 
# We can see from the classification report below that our accuracy is 69% - not great, but again, predicting whether or not someone is a linkedin user is a relatively low risk problem if we get it wrong.

# In[189]:


#print(classification_report(y_test, y_pred))


# ***

# #### Q10: Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# # In[192]:


# newdata = pd.DataFrame({
#     'income': [8, 8],
#     'educ2': [7, 7],
#     'par': [2, 2],
#     'marital': [1, 1],
#     'gender': [2, 2],
#     'age': [42, 82]
# })
# newdata


# # In[193]:


# newdata["prediction_linkedin_user"] = lr.predict(newdata)
# newdata


# ***

# In[ ]:

# Streamlit App

import streamlit as st

st.title('Can I predict if _you_ are a Linkedin user?')
st.subheader("Enter your information below to see if the model predicts that you are a Linkedin user!")


user_age = st.slider("How old are you?", min_value=5, max_value=95,value = 50, step=1)
user_gender = st.selectbox("What is your sex?",options=['Female', 'Male'])
user_married = st.selectbox("What is your marital status?",options=['Married', 'Living with a partner', 'Divorced', 'Separated', 'Widowed', 'Never been married'])
user_parent = st.selectbox("Do you have any kids?",options=['Parent', 'Non-Parent'])
user_educ = st.selectbox("What is your highest level of education completed?", options=['Less than High School', 'High School incomplete', 'High School diploma', 'Some College', 'Two-Year College Completed', 'Four-Year College Completed', 'Some Graduate School', 'Postgraduate or Professional degree recieved'])
user_income = st.selectbox('What is your household income?', options=['Less than 10,000', '10,000 to under 20,000', '20,000 to under 30,000', '30,000 to under 40,000','40,000 to under 50,000', '50,000 to under 60,000', '70,000 to under 100,000', '100,000 to under 150,000','150,000+'])

if user_gender == 'Female':
    user_gender = 2
else:
    user_gender = 1

if user_married == 'Married':
    user_married = 1
elif user_married == 'Living with a partner':
    user_married = 2
elif user_married == 'Divorced':
    user_married = 3
elif user_married == 'Separated':
    user_married = 4
elif user_married == 'Widowed':
    user_married = 5
else: user_married = 6

if user_parent == 'Parent':
    user_parent = 1
else: user_parent = 2

if user_educ == 'Less than High School':
    user_educ = 1
elif user_educ == 'High School incomplete':
    user_educ = 2
elif user_educ == 'High School diploma':
    user_educ = 3
elif user_educ == 'Some College':
    user_educ = 4
elif user_educ == 'Two-Year College Completed': 
    user_educ = 5
elif user_educ == 'Four-Year College Completed':
    user_educ = 6
elif user_educ == 'Some Graduate School':
    user_educ = 7
else: user_educ = 8

if user_income == 'Less than 10,000':
    user_income = 1
elif user_income == '10,000 to under 20,000':
    user_income = 2
elif user_income == '20,000 to under 30,000':
    user_income = 3
elif user_income == '30,000 to under 40,000':
    user_income = 4
elif user_income == '40,000 to under 50,000': 
    user_income = 5
elif user_income == '50,000 to under 60,000':
    user_income = 6
elif user_income == '70,000 to under 100,000':
    user_income = 7
elif user_income == '100,000 to under 150,000':
    user_income = 8
else: user_income = 9


# user = pd.DataFrame({
#     'income': [user_income],
#     'educ2': [user_educ],
#     'par': [user_parent],
#     'marital': [user_married],
#     'gender': [user_gender],
#     'age': [user_age]
# })

user = [user_income,user_educ,user_parent,user_married,user_gender,user_age]
# st.write(user)

predicted_class = lr.predict([user])

probs = lr.predict_proba([user])

probs = probs[0][1]*100

if predicted_class > 0:
    label= "I predict that you are a Linkedin user 🔥! \n(You should connect with me!)"
else:
    label= "I predict that you are not a Linkedin user 👀! \n(but you definitely should make an account!)"

if st.button('Reveal Your Results!'):
    st.subheader(label)
    st.caption(f'Based on your answers above there is a **{probs:.2f}%** probability that you are a Linkedin user.')
    

