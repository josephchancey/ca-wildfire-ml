# Imports
import streamlit as st
import pandas as pd

#"""
#--------------------------
#------ Introduction ------
#--------------------------
#"""

# Title text, seen at very top of webpage - Using .markdown to center this text
st.markdown("<h1 style='text-align: center;'>California Wildfire Unsupervised Machine Learning Model</h1>", unsafe_allow_html=True)

# Brief description of the web app
st.write("This machine learning (ML) model takes historical wildfire, drought, & precipitation data from 2013-2021" \
    " and is trained to predict the likelyhood of a wildfire given specific percipitation/drough" \
        " conditions within California.")

# Image of developers w/ a caption to put image into context
st.image("img/developers_img.png", caption="This app was created by Breanna Sewell, David Koski, and Joseph Chancey.")



#"""
#--------------------------
#------ DATA SECTION ------
#--------------------------
#"""

# Data section header
st.header("The Data")

st.write("Click the box below to get a view of all the data used in this project.")

if st.checkbox('Show Data'):

    st.write("This data comes from Calfire. It is a record of documented wildfires from 2013-2021." \
         "Here we can see a cleaned version of the data with only what will be fed into the machine learning algorith.")

    clean_fire = pd.read_csv("data/clean/fire_data_clean.csv")
    st.dataframe(clean_fire.head())

    clean_fire = pd.read_csv("data/clean/precip_data_clean.csv")
    st.dataframe(clean_fire.head())

    clean_fire = pd.read_csv("data/clean/drought_data_clean.csv")
    st.dataframe(clean_fire.head())





#""" TODO
#------------------------------
#------ TRAINING SECTION ------
#------------------------------
#"""



#""" TODO
#---------------------------
#------ MODEL SECTION ------
#---------------------------
#"""


#""" TODO
#--------------------------------
#------ CONCLUSION SECTION ------
#--------------------------------
#"""

