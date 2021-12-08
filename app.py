# Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# ML dependency imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split

# Page Settings
st.set_page_config(page_title="California Wildfire ML", page_icon="img/fav.png")


#"""
#--------------------------
#---- MACHINE LEARNING ----
#--------------------------
#"""

def main():
    
    print("WERE IN MAIN FUNCTION")
    print("CLEANING FIRE")
    clean_fire()
    print("CLEANING DROUGHT")
    clean_drought()
    print("CLEANING RAIN")
    clean_percip()
    print("ML")
    ml_model()

def clean_fire():

    # import fire data csv
    fireFile = "./data/fire_data.csv"

    # read the file and store in a data frame
    fireData = pd.read_csv(fireFile)
        
        # remove extraneous columns
    fireData = fireData[["incident_id","incident_name","incident_county","incident_acres_burned",
                        "incident_dateonly_created","incident_dateonly_extinguished"]]

    # rename columns
    fireData = fireData.rename(columns={"incident_id":"ID","incident_name":"Name","incident_county":"County",
                                        "incident_acres_burned":"AcresBurned","incident_dateonly_created":"Started",
                                    "incident_dateonly_extinguished":"Extinguished"})

    # check for duplicates, then drop ID column
    fireData.drop_duplicates(subset=["ID"])
    fireData = fireData[["Name","County","AcresBurned","Started","Extinguished"]]


    # create a column that contains the duration
    # first convert date columns to datetime
    fireData["Started"] = pd.to_datetime(fireData["Started"])
    fireData["Extinguished"] = pd.to_datetime(fireData["Extinguished"])

    # subtract the dates
    fireData["Duration"] = fireData["Extinguished"] - fireData["Started"]

    # convert duration to string and remove "days"
    fireData["Duration"] = fireData["Duration"].astype(str)
    fireData["Duration"] = fireData["Duration"].str.replace("days","")

    # replace NaT with NaN and convert back to float
    fireData["Duration"] = fireData["Duration"].replace(["NaT"],"NaN")
    fireData["Duration"] = fireData["Duration"].astype(float)

    # add one day to duration to capture fires that started and were extinguished in the same day
    fireData["Duration"] = fireData["Duration"] + 1

    # create a column for year and filter for fires during or after 2013
    fireData["Year"] = fireData["Started"].dt.year
    fireData = fireData.loc[(fireData["Year"]>=2013),:]

    # create a column to hold the year and month of the start date
    fireData["Date"] = fireData["Started"].apply(lambda x: x.strftime('%Y-%m'))

    fireData = fireData[["Date", "County", "Duration", "AcresBurned"]]

    # drop nulls
    fireData = fireData.dropna()

    # reset the index
    fireData.reset_index(inplace=True,drop=True)

        # export as csv
    return fireData.to_csv("./data/clean/fire_data_clean.csv",index=False)


def clean_percip():

    # import precipitation data csv
    precipFile = "./data/precip_data.csv"

    # read the file and store in a data frame
    precipData = pd.read_csv(precipFile)
    
    # remove extraneous columns
    precipData = precipData[["Date","Location","Value"]]

    # rename columns
    precipData = precipData.rename(columns = {"Location":"County","Value":"Precip"})

    # remove "county" from county column to be consistent with other datasets
    precipData["County"] = precipData["County"].astype(str)
    precipData["County"] = precipData["County"].str.replace(" County","")

    # convert date column
    precipData["Date"] = pd.to_datetime(precipData["Date"].astype(str), format='%Y%m')

    # create a column for year and filter for data during or after 2013
    precipData["Year"] = precipData["Date"].dt.year
    precipData = precipData.loc[(precipData["Year"]>=2013),:]

    # drop the year column
    precipData = precipData[["Date","County","Precip"]]

    # edit the date column to match the format of the other datasets
    precipData["Date"] = precipData["Date"].apply(lambda x: x.strftime('%Y-%m'))

    precipData = precipData.dropna()
    precipData.reset_index(inplace=True,drop=True)

    # export as csv
    return precipData.to_csv("./data/clean/precip_data_clean.csv",index=False)

def clean_drought():

    # import drought data csv
    droughtFile = "./data/drought_data.csv"

    # read the file and store in a dataframe
    droughtData = pd.read_csv(droughtFile)

    droughtData = droughtData[["ValidStart","County","None","D0","D1","D2",
                          "D3","D4"]]

    # rename columns
    droughtData = droughtData.rename(columns={"ValidStart":"Date"})

    # remove "county" from county column to be consistent with other datasets
    droughtData["County"] = droughtData["County"].astype(str)
    droughtData["County"] = droughtData["County"].str.replace(" County","")

    # edit the date column to match the format of the other datasets
    droughtData["Date"] = pd.to_datetime(droughtData["Date"])
    droughtData["Date"] = droughtData["Date"].apply(lambda x: x.strftime('%Y-%m'))

    # drop nulls and reset the index
    droughtData = droughtData.dropna()
    droughtData.reset_index(inplace=True,drop=True)

    # group by date and county and average the drought levels of each week to obtain a monthly summary
    groupedDrought = droughtData.groupby(["Date","County"])
    groupedDrought = groupedDrought.mean()

    # export as csv
    groupedDrought.to_csv("./data/clean/drought_data_clean.csv")


def ml_model():

    # import fire data
    fireFile = "./data/clean/fire_data_clean.csv"
    fireData = pd.read_csv(fireFile)

    droughtFile = "./data/clean/drought_data_clean.csv"
    droughtData = pd.read_csv(droughtFile)

    precipFile = "./data/clean/precip_data_clean.csv"
    precipData = pd.read_csv(precipFile)

    droughtMerged = pd.merge(droughtData, fireData, on = ["Date", "County"])
    precipMerged = pd.merge(precipData, fireData, on = ["Date","County"])
    masterMerge = pd.merge(droughtMerged, precipData, on = ["Date","County"])

    droughtML = pd.get_dummies(droughtMerged)
    precipML = pd.get_dummies(precipMerged)
    masterML = pd.get_dummies(masterMerge)
    masterML.drop(columns='None', inplace=True)


    df = masterML
    
    X = df
    y = df["AcresBurned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    reg = LinearRegression().fit(X_train_scaled, y_train)
    reg_score_val = reg.score(X_test_scaled, y_test)


    lasso = Lasso().fit(X_train_scaled, y_train)

    lasso_score_val = lasso.score(X_test_scaled, y_test)

    st.write(lasso_score_val)
    st.write(reg_score_val)
    
    return reg_score_val, lasso_score_val



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

if __name__ == "__main__":
    main()