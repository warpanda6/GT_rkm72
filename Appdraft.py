#Import Packages
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image


# Visuals
st.title("Are you a Linkedin User?")
image = Image.open("Linkedin.jfif")
st.image(image, caption = "Connecting the World's professionals")

## Age
age = st.slider("What is you Age?", 0, 100)

st.write("Age given =", age)


##Education
education = st.selectbox("Education level", 
              options = ["Less than High School (Grades 1-8 or no formal schooling)", "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
              "High school graduate (Grade 12 with diploma or GED certificate)", "Some college, no degree (includes some community college)", "Two-year associate degree from a college or university",
              "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)", "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
               "Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)"
                       ])

st.write (f"Education Level Selected: {education}")

if education == "Less than High School (Grades 1-8 or no formal schooling)":
    education = 1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education = 2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)":
    education = 3
elif education == "Some college, no degree (includes some community college)":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education = 7
elif education == "Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)":
    education = 8


##Income
income = st.selectbox("Household Income",
                options = ["Less than $10K", "$10-20K", "$20-30K", "$30-40K", "$40-50K",
                "$50-75K", "$75-100K", "$100-150K", "$150K+"]) 

st.write (f"Income Selected: {income}")

if income == "Less than $10K":
    income = 1
elif income == "$10-20K":
    income = 2
elif income == "$20-30K":
    income = 3
elif income == "$30-40K":
    income = 4
elif income == "$40-50K":
    income = 5
elif income == "$50-75K":
    income = 6
elif income == "$75-100K":
    income = 7
elif income == "$100-150K":
    income = 8
else:
    income = 9



##Married
married = st.selectbox("Are you Married?", options = ["Yes", "No"])

st.write (f"Marital Status Selected: {married}")
if married == "Yes":
    married = 1
else:
    married = 0


##Parent
parent = st.selectbox("Are you the Parent of a child under 18 living in your home?", options = ["Yes","No"])

st.write (f"Parental Status Selected: {parent}")

if parent == "Yes":
    parent = 1
else:
    parent = 0




##Gender
female = st.selectbox("What gender do you identify as?", options = ["Male","Female"])

st.write (f"Gender Selected: {female}")
if female == "Male":
    female = 0
else:
    female = 1


#Bring on the code

#Import Data
s = pd.read_csv("social_media_usage.csv")

#clean
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

#Data frame

ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    "income":np.where(s["income"]> 9,np.nan,s["income"]),
    "education":np.where(s["educ2"]> 8,np.nan,s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] >98, np.nan,s["age"])
})

ss = ss.dropna()

# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=2022) # set for reproducibility
                    
# Initialize algorithm 
lr = LogisticRegression(class_weight = "balanced")
# Fit algorithm to training data
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

NewD = pd.DataFrame({
    "income": [income],
    "education": [education],
    "parent": [parent],
    "married": [married],
    "female": [female],
    "age": [age]    
})


user_prob = (lr.predict_proba(NewD))[0][1]
user_prob = (round(user_prob, 4))
st.markdown ({user_prob})

st.write(f"Based on your information, we predict that the probability that you are a LinkedIn user is {user_prob*100}%.")

