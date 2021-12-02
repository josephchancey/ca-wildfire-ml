# Import dependencies
from flask import Flask, render_template, url_for, redirect, jsonify
from flask_pymongo import PyMongo
import json
import pymongo
from pymongo.common import partition_node
from scrape import scrapeData

app = Flask(__name__)

# Use PyMongo to establish Mongo connection
mongo = PyMongo(app, uri="mongodb://localhost:27017/wildfire_ml")


### ----- ROUTES ----- ###

# Home Page Route
@app.route('/')
@app.route('/home')
def home():

    return render_template('index.html')



# Scrape Route for scrape.py function
@app.route('/scrape', methods=['GET', 'POST'])
def scrape():

    scraped_data = scrapeData()
    for i in scraped_data:
        mongo.db.fires.replace_one({'_id': i['_id']}, i, upsert=True)

    return redirect("/")



# Machine Learning Calculator Page
@app.route('/ml')
def ml():

    return render_template('ml.html')



# About Page
@app.route('/about')
def about():

    return render_template('about.html')



# Active Fires
@app.route('/active/fires')
def activeFires():

    db_data = list(mongo.db.fires.find({'IsActive': True}, {'_id': False}))
    parsed = [x for x in db_data]
    return jsonify(parsed)



# Inactive Fires
@app.route('/inactive/fires')
def inactiveFires():

    db_data = list(mongo.db.fires.find({'IsActive': False}, {'_id': False}))
    parsed = [x for x in db_data]
    return jsonify(parsed)



# Wildfire Fires
@app.route('/wildfire/fires')
def wildfireType():

    db_data = list(mongo.db.fires.find({'Type': "Wildfire"}, {'_id': False}))
    parsed = [x for x in db_data]
    return jsonify(parsed)


# Debugger
if __name__ == '__main__':
    app.run(debug=True)
