# Imports
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os.path
import seaborn

# ML dependency imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from streamlit.type_util import Key

# Page Settings
st.set_page_config(page_title="California Wildfire ML", page_icon="./img/fav.png", initial_sidebar_state="collapsed")

#"""
#--------------------------
#---- MACHINE LEARNING ----
#--------------------------
#"""

def main():

    print("IN MAIN")

    # If data has not been cleaned, then clean it 
    if os.path.isfile("./data/clean/fire_data_clean.csv") == False:
        print("CLEANING FIRE")
        clean_fire()

    if os.path.isfile("./data/clean/drought_data_clean.csv") == False:
        print("CLEANING DROUGHT")
        clean_drought()

    if os.path.isfile("./data/clean/precip_data_clean.csv") == False:
        print("CLEANING RAIN")
        clean_percip()

    # # Init sidebar with header text
    # st.sidebar.header("Menu")

    # # Add URL for github repository
    # st.sidebar.write("[View on GitHub](https://github.com/josephchancey/ca-wildfire-ml)")


def old_fire_dataset():
    unclean_fire = pd.read_csv("./data/fire_data.csv")
    return unclean_fire

def old_precip_dataset():
    unclean_precip = pd.read_csv("./data/precip_data.csv")
    return unclean_precip

def old_drought_dataset():
    unclean_drought = pd.read_csv("./data/drought_data.csv")
    return unclean_drought


def clean_fire():

    if os.path.isfile("./data/clean/fire_data_clean.csv") == False:
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
        fireData.to_csv("./data/clean/fire_data_clean.csv",index=False)

        return fireData

    else: 
    # This prevents the cleaning from being ran each time this function is called, checks if cleaning is done already

        fireData = pd.read_csv("./data/clean/fire_data_clean.csv")
        return fireData



def clean_percip():


    if os.path.isfile("./data/clean/precip_data_clean.csv") == False:

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
        precipData.to_csv("./data/clean/precip_data_clean.csv",index=False)

        return precipData

    else:
        precipData = pd.read_csv("./data/clean/precip_data_clean.csv")
        return precipData





def clean_drought():

    if os.path.isfile("./data/clean/precip_data_clean.csv") == False:

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

        return groupedDrought
    else:
        
        groupedDrought = pd.read_csv("./data/clean/drought_data_clean.csv")
        return groupedDrought
    


def lin_model():

    print("MODEL RAN")
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
    
    return reg_score_val


def lasso_model():

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

    masterML = pd.get_dummies(masterMerge)
    masterML.drop(columns='None', inplace=True)

    df = masterML
    
    X = df
    y = df["AcresBurned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)


    lasso = Lasso().fit(X_train_scaled, y_train)

    lasso_score_val = lasso.score(X_test_scaled, y_test)

    return lasso_score_val


def random_forest():

    
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

    masterML = pd.get_dummies(masterMerge)
    masterML.drop(columns='None', inplace=True)

    df = masterML
    
    X = df
    y = df["AcresBurned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    clf.score(X_train, y_train)
    random_forest_val = clf.score(X_test, y_test)

    return random_forest_val


def plot_rnd_frst():

        
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

    masterML = pd.get_dummies(masterMerge)
    masterML.drop(columns='None', inplace=True)

    df = masterML
    
    X = df
    y = df["AcresBurned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    clf.score(X_train, y_train)
    random_forest_val = clf.score(X_test, y_test)

    plt.scatter(list(X_test["Precip"]), list(y_test.values), c="Green", label="Training Data")
    plt.scatter(list(X_test["Precip"]), clf.predict(X_test), c="Red", label="Prediction")
    plt.ylabel('Acres Burned')
    plt.xlabel('Precipitation Level by County')
    plt.legend()
    #plt.hlines(y=0, xmin=y.min(), xmax=y.max())
    plt.title("Random Forest Classification on Precipitation")

    return plt


def plot_lin_reg():
    
    print("MODEL RAN")
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

    
    plt.cla()
    plt.scatter(list(X_test["Precip"]), list(y_test.values), c="Green", label="Training Data")
    plt.scatter(list(X_test["Precip"]), reg.predict(X_test), c="Red", label="Predictions")
    plt.ylabel('Acres Burned')
    plt.xlabel('Precipitation Level by County')
    plt.legend()
    #plt.hlines(y=0, xmin=y.min(), xmax=y.max())
    plt.title("Linear Regression Model on Precipitation")


    return plt



#"""
#--------------------------
#------ Introduction ------
#--------------------------
#"""

# Title text, seen at very top of webpage - Using .markdown to center this text
st.markdown("<h1 style='text-align: center;'>California Wildfire Unsupervised Machine Learning Model</h1>", unsafe_allow_html=True)

# Brief description of the web app
st.write("""This machine learning (ML) model takes historical wildfire, drought, & precipitation data from 2013-2021
and is trained to predict the likelihood of a wildfire given specific percipitation/drought
conditions within California.""")

st.markdown("<hr>", unsafe_allow_html=True)

# Create About Section
# Init columns
dev_col1, dev_col2, dev_col3 = st.columns(3)

# Breanna About Column
dev_col1.image("https://avatars.githubusercontent.com/u/83804429?v=4")
dev_col1.header("Breanna S.")
if dev_col1.button("View Github Profile", key=999):
    dev_col1.markdown("[Link to Breanna's Github](https://github.com/bre-sew)")


# David About Column
dev_col2.image("https://avatars.githubusercontent.com/u/85533882?v=4")
dev_col2.header("David K.")
if dev_col2.button("View Github Profile", key=998):
    # webbrowser.open_new_tab("https://github.com/dkoski23")
    dev_col2.markdown("[Link to David's Github](https://github.com/dkoski23)")

# Joseph About Column

dev_col3.image("https://avatars.githubusercontent.com/u/84075822?v=4")
dev_col3.header("Joseph C.")
if dev_col3.button("View Github Profile", key=997):
    # webbrowser.open_new_tab("https://github.com/josephchancey")\
    dev_col3.markdown("[Link to Joseph's Github](https://github.com/josephchancey)")


# st.write(""" This app was created by Breanna Sewell, David Koski, and Joseph Chancey. Breanna collected the data for this project, dedicating her
# time to cleaning and ensuring the data was ML ready. David pre-processed the data to prepare for ML training and then trained the ML model with that data.
# Joseph collected this work and converted it into a Python script and turned it into a Streamlit application. """)


st.write("""

This app was created by Breanna Sewell, David Koski, and Joseph Chancey with the intention of 
determining if drought and precipitation are good indicators of wildfire extent in California. All 
data analyzed is from 2013 to 2021; data sources are listed below.

Breanna collected the data for this project, dedicating her time to cleaning and pre-processing 
ensuring the data to ensure it was machine learning (ML) ready. David pre-processedscaled the 
data to prepare for ML training and then trained the ML model with theat data. Joseph collected 
this work, and converted it into a Python script, and turned it into a Streamlit application.

""")

st.markdown("<hr>", unsafe_allow_html=True)


#"""
#--------------------------
#------ DATA SECTION ------
#--------------------------
#"""

# Data Pre-Cleaned section header
st.header("The Data (Pre-Cleaning)")

st.write("Click the box below to get a view of all the data used in this project.")

if st.checkbox('Show Wildfire Data'):

    fire_col1, fire_col2, fire_col3 = st.columns(3)
    fire_col1.metric(label="Columns", value="23")
    fire_col2.metric(label="Rows", value="1967")

    st.write("""
    This data is a record of documented California wildfires from 2013-2021.
    The data includes the start and extinguished dates, counties where the fire 
    occurred, acreage burned, latitude and longitude of origin, administrative unit assigned 
    to the fire, etc. Note that the data shown below is only the head, rather than the full 
    dataset.

    Sources: Incidents Overview, Historical Fire Data, California Department of Forestry and 
    Fire Protection [CALFIRE](https://www.fire.ca.gov/incidents/)

    """)
    
    st.write("Do note that this is only the head of the data, rather than the full dataset.")

    st.dataframe(old_fire_dataset().head())

# --------------------------
if st.checkbox('Show Precipitation Data'):

    prec_col1, prec_col2, prec_col3 = st.columns(3)
    prec_col1.metric(label="Columns", value="7")
    prec_col2.metric(label="Rows", value="88276")

    st.write("""
    
    This data records California precipitation levels from 1895-2021. The 
    data includes the county, county ID, date, monthly total precipitation (inches), rank of 
    precipitation compared to other counties, etc. Note that the data shown below is only the
    head, rather than the full dataset.

    Source: Climate at a Glance: County Mapping, National Oceanic and Atmospheric 
    Administration [NOAA](https://www.ncdc.noaa.gov/cag/county/mapping)

    """)

    st.dataframe(old_precip_dataset().head())

# --------------------------
if st.checkbox('Show Drought Data'):


    drought_col1, drought_col2, drought_col3 = st.columns(3)
    drought_col1.metric(label="Columns", value="13")
    drought_col2.metric(label="Rows", value="27028")

    st.write("""
    
    This data records California drought severity from 2013-2021. The 
    data includes the county, county ID, start date of the recorded week, end date of the 
    recorded week, and, most importantly, the percent acreage of each county designated 
    by the following drought metrics: none (no drought); D0 (abnormally dry); D1 (moderate 
    drought); D2 (severe drought); D3 (extreme drought); and D4 (exceptional drought). 
    Note that the data shown below is only the head, rather than the full dataset.

    Source: U.S. Drought Monitor, Comprehensive Statistics, Collaboration by National 
    Drought Mitigation Center, U.S. Department of Agriculture, and National Oceanic and 
    Atmospheric Administration [NOAA](https://droughtmonitor.unl.edu/DmData/DataDownload/ComprehensiveStatistics.aspx)

    """)

    st.dataframe(old_drought_dataset().head())



st.markdown("<hr>", unsafe_allow_html=True)



# Data Post-Cleaned section header
st.header("The Data (Post-Cleaning)")

if st.checkbox('Show Clean Wildfire Data'):

    fire_col1, fire_col2, fire_col3 = st.columns(3)
    fire_col1.metric(label="Columns", value="4", delta="-19")
    fire_col2.metric(label="Rows", value="1800", delta="-167")

    st.write("""
    
    Here we can see a cleaned version of the 2013-2021 California wildfire data, which only 
    includes what will be fed into the machine learning algorithm. The data now includes 
    only the month and year of the start date of each fire, the county where the fire 
    occurred, the duration of the fire (days), and the acreage burned. Note that the data 
    shown below is only the head, rather than the full dataset.

    Source: Incidents Overview, Historical Fire Data, California Department of Forestry and 
    Fire Protection [CALFIRE](https://www.fire.ca.gov/incidents/)

    """)

    st.dataframe(clean_fire().head())

# --------------------------
if st.checkbox('Show Clean Precipitation Data'):

    prec_col1, prec_col2, prec_col3 = st.columns(3)
    prec_col1.metric(label="Columns", value="3", delta="-4")
    prec_col2.metric(label="Rows", value="6148", delta="-82,128")

    st.write("""
    
    Here we can see a cleaned version of the 2013-2021 California precipitation data, which 
    only includes what will be fed into the machine learning algorithm. The data now 
    includes only the month and year of the data collection, the precipitation total for the 
    month (inches), and the county. Note that the data shown below is 
    only the head, rather than the full dataset.


    Source: Climate at a Glance: County Mapping, National Oceanic and Atmospheric Administration [NOAA](https://www.ncdc.noaa.gov/cag/county/mapping)
    
    """)

    st.dataframe(clean_percip().head())

# --------------------------
if st.checkbox('Show Clean Drought Data'):

    drought_col1, drought_col2, drought_col3 = st.columns(3)
    drought_col1.metric(label="Columns", value="8", delta="-5")
    drought_col2.metric(label="Rows", value="6206", delta="-20,822")

    st.write("""
    
    Here we can see a cleaned version of the 2013-2021 California drought data, which only 
    includes what will be fed into the machine learning algorithm. The data now includes 
    only the month and year of the data collection, the county, and the percent acreage of 
    each county designated by the following drought metrics: none (no drought); D0 
    (abnormally dry); D1 (moderate drought); D2 (severe drought); D3 (extreme drought); 
    and D4 (exceptional drought). Note that the data shown below is only
    the head, rather than the full dataset.


    Source: U.S. Drought Monitor, Comprehensive Statistics, Collaboration by National 
    Drought Mitigation Center, U.S. Department of Agriculture, and National Oceanic and 
    Atmospheric Administration [NOAA](https://droughtmonitor.unl.edu/DmData/DataDownload/ComprehensiveStatistics.aspx)

    """)

    st.dataframe(clean_drought().head())

st.markdown("<hr>", unsafe_allow_html=True)


#""" TODO
#------------------------------
#------ TRAINING SECTION ------
#------------------------------
#"""

# Data get_dummies & type checking
st.header("ML Data Pre-Processing")

st.write("The clean data was merged into one training dataset.")
st.code("""
# merge the drought, precipitation, and fire data
masterMerge = pd.merge(droughtMerged, precipData, on = ["Date","County"])
masterMerge
""")


st.write("This loop was used to check for null values in the final merged dataset.")
st.code("""
# find null values
for column in masterMerge.columns:
    print(f"Column {column} has {masterMerge[column].isnull().sum()}
    null values")
""")
st.write("Datatypes were checked with `.dtypes`.")
# Code for checking datatypes
st.code("""
# check the data types
masterMerge.dtypes
""")

st.write("Pandas `get_dummies` was used to encode the dataframe.")
st.code("""
# Get dummy data
masterML = pd.get_dummies(masterMerge)
""")

st.markdown("<hr>", unsafe_allow_html=True)


#"""
#---------------------------
#------ MODEL SECTION ------
#---------------------------
#"""

# Lasso and Lin-Reg Models 
st.header("ML Models")

st.write("Here we assign our variables to begin testing and training.")
st.code("""
# Assign data
X = df
y = df["AcresBurned"]
""")

st.write("Here we scale the data before fitting it to our selected models.")
st.code("""
# Scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
""")

st.write("Here we fit the data to the models.")
st.code("""
# Linear Regression
reg = LinearRegression().fit(X_train_scaled, y_train)
reg.score(X_test_scaled, y_test) # test data

# Lasso 
lasso = Lasso().fit(X_train_scaled, y_train)
lasso.score(X_test_scaled, y_test) # test data
""")


st.write(f'**Linear Regression** Model Score: `{lin_model()}`')

st.write(f'**Lasso** Model Score: `{lasso_model()}`')

st.markdown("<hr>", unsafe_allow_html=True)
# END OF ML MODEL SECTION


# New Section 
st.header("Hmmm, What Happened?")

st.write(f"""
As we can see, a
Linear Regression model score of `{round(lin_model(), 7)}` is likely not something we can trust. 
The same goes for the Lasso model score of `{round(lasso_model(), 7)}`. 

These models seem to be victim to *overfitting* with our dataset. 

Let's try this again, but use a different model. We can compare the results and come to some meaningful
conclusions.
""")

st.header("Adding Another Model")

if st.checkbox("Run Random Forest Classification Model"):

    st.write(f"""
    **Random Forest Classification** will be our third model. Below, you can see
    the code for our implementation.
    """)

    st.code("""
    # Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    """)

    st.write(f"**Random Forest Classification** gave us a score of `{round(random_forest(), 7)}`.")
    st.write("""This score reflects the opposite of what our previous models were doing. Rather than
    memorizing the data and being victim to overfitting, Random Forest is wildly innacurate and is not suitable for our data.""")

    st.markdown("<hr>", unsafe_allow_html=True)

    # New Section 
    st.header("Comparing Results")

    st.write(f"""
    After testing three different models and getting different reactions from each, we beed to visualize the predictions
    to better understand what is going on behind the scenes.
    """)

    col1, col2 = st.columns(2)


    col1.pyplot(plot_rnd_frst())
    col1.caption("Random Forest Model Predictions")

    col2.pyplot(plot_lin_reg())
    col2.caption("Linear Regression Model Predictions")

    st.write("""
    Here, we can start to get an idea of just how overfitted our Linear Regression model is, as well as how 
    poorly our Random Forest model performs.
    """)


st.markdown("<hr>", unsafe_allow_html=True)
# END OF WHAT HAPPENED SECTION

#"""
#--------------------------------
#------ CONCLUSION SECTION ------
#--------------------------------
#"""

# Conclusion Summary
st.header("Summary")

st.write("""
The Linear Regression model and Lasso model are
susceptible to overfitting with our data. The Random Forest classification model was 
wildly inaccurate, but it did not suffer from overfitting. From this we can 
conclude that drought and precipitation data are good linear indicators of wildfire 
acreage in California, although these models are not suitable to predict acreage.

The repository for this project can be found on [Github](https://github.com/josephchancey/ca-wildfire-ml)

""")

# st.subheader("Linear Regression")

# st.subheader("Lasso")

# st.subheader("Random Forest")

st.markdown("<hr>", unsafe_allow_html=True)


# # Data Post-Cleaned section header
# st.header("Interested?")
# # TODO ADD POST CLEANED DATA HERE
# st.markdown("<hr>", unsafe_allow_html=True)

# Thank you celebration button
col1, col2, col3, col4, col5 = st.columns(5)
if col3.button('Thank You!'):
    st.balloons()


if __name__ == "__main__":
    main()