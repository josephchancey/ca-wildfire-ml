# Import dependencies
import urllib.request, json 
from bson.json_util import dumps, loads
import os, ssl
import pymongo
import itertools
import pandas as pd

def scrapeData():

    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
        ssl._create_default_https_context = ssl._create_unverified_context
    
    # Not Active 2021
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2021") as url:
        inactive_2021 = json.loads(url.read().decode())

    # Active 2021
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2021") as url:
        active_2021 = json.loads(url.read().decode())
    
    # Not Active 2020
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2020") as url:
        inactive_2020 = json.loads(url.read().decode())

    # Active 2020
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2020") as url:
        active_2020 = json.loads(url.read().decode())

    # Not Active 2019
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2019") as url:
        inactive_2019 = json.loads(url.read().decode())

    # Active 2019
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2019") as url:
        active_2019 = json.loads(url.read().decode())
        
    # Not Active 2018
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2018") as url:
        inactive_2018 = json.loads(url.read().decode())

    # Active 2018
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2018") as url:
        active_2018 = json.loads(url.read().decode())

    # Not Active 2017
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2017") as url:
        inactive_2017 = json.loads(url.read().decode())

    # Active 2017
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2017") as url:
        active_2017 = json.loads(url.read().decode())

    # Not Active 2016
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2016") as url:
        inactive_2016 = json.loads(url.read().decode())

    # Active 2016
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2016") as url:
        active_2016 = json.loads(url.read().decode())

    # Not Active 2015
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2015") as url:
        inactive_2015 = json.loads(url.read().decode())

    # Active 2015
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2015") as url:
        active_2015 = json.loads(url.read().decode())

    # Not Active 2014
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2014") as url:
        inactive_2014 = json.loads(url.read().decode())

    # Active 2014
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2014") as url:
        active_2014 = json.loads(url.read().decode())

    # Not Active 2013
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=2013") as url:
        inactive_2013 = json.loads(url.read().decode())

    # Active 2013
    with urllib.request.urlopen("https://www.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=false&year=2013") as url:
        active_2013 = json.loads(url.read().decode())

    scraped_data = list(itertools.chain(active_2021,inactive_2021, active_2020, inactive_2020, active_2019,inactive_2019, 
    active_2018, inactive_2018, active_2018, inactive_2017,active_2016, inactive_2016,
    active_2015, inactive_2015, active_2014, inactive_2014, active_2013, inactive_2013))

    # convert to DataFrame to add duration and years columns
    fireData = pd.DataFrame(scraped_data)

    # create a column that contains the duration of each fire
    # first convert the date columns to datetime
    fireData["ExtinguishedDateOnly"] = pd.to_datetime(fireData["ExtinguishedDateOnly"])
    fireData["StartedDateOnly"] = pd.to_datetime(fireData["StartedDateOnly"])

    # subtract the two dates
    fireData["Duration(Days)"] = fireData["ExtinguishedDateOnly"] - fireData["StartedDateOnly"]

    # convert duration to string and remove "days"
    fireData["Duration(Days)"] = fireData["Duration(Days)"].astype(str)
    fireData["Duration(Days)"] = fireData["Duration(Days)"].str.replace("days","")

    # convert NaT to NaN and convert back to float
    fireData["Duration(Days)"] = fireData["Duration(Days)"].replace(["NaT"],"NaN")
    fireData["Duration(Days)"] = fireData["Duration(Days)"].astype(float)

    # add 1 day so fires that start and end on the same day do not have a duration of 0
    fireData["Duration(Days)"] = fireData["Duration(Days)"] + 1

    # create a column that holds the year of each start date
    fireData["Year"] = fireData["StartedDateOnly"].dt.year

    # drop the extraneous columns
    fireData = fireData.drop("ExtinguishedDateOnly",1)
    fireData = fireData.drop("StartedDateOnly",1)

    # drop the NaNs
    fireData = fireData.fillna(0)

    # convert the data back to JSON
    final_data = fireData.to_dict(orient='records')
    
    # Initialize PyMongo to work with MongoDBs
    conn = 'mongodb://localhost:27017'
    client = pymongo.MongoClient(conn)

    # Define database and collection
    db = client.wildfire_ml

    try:
        db.fires.drop()
        print("Dropped Fires")
    except:
        print("Database not dropped")

    collection = db.fires

    # Loop through list and add each dictionary item to MongoDB
    for item in final_data:
        collection.insert_one(item)

    # # Converting to the JSON
    # json_data = dumps(scraped_data, indent = 4) 

    # # Writing data to file data.json - Easy for JavaScript Access
    # with open('data/fires.json', 'w') as file:
    #     file.write(json_data)

    # !TODO! Gather Rain Data and add it to
    # wildfire_ml database under the collection of
    # "rain" - That way we have a 'fires' & 'rain' 
    # collection under wildfire_ml

    # !NOTE! Please do not merge or file upload anything into the main branch
    # The main branch is heavily relied upon for stablinity and controlled testing
 
    
    print("Scrape Done!")
    return final_data