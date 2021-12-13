# California Wildfire Machine Learning 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/josephchancey/ca-wildfire-ml/main/app.py)

Our aim was to utilize unsupervised machine learning to determine if wildfire burn acreage in California can be accurately predicted by the amount of precipitation or the severity of drought. We collected data from CALFIRE and National Oceanic and Atmospheric Administration (NOAA) that included historic datapoints for wildfires, drought conditions, and precipitation by county in California.

## Repository Structure

Main application | [app.py](app.py)

Cleaning Notebooks | [Three Notebooks](/data)

Machine Learning Notebook | [ML Notebook](/data/clean/Preprocessing-&-Machine-Learning.ipynb)

## Authors

**Joseph Chancey:** Git/Github management, Python3 Streamlit back/front end, Jupyter Conversion, Refactoring.

**Breanna Sewell:** Data collection, Data transformation, Data encoding, Data cleaning.

**David Koski:** Statistical analysis, Data visualization, Machine learning, Model fitting.

## Tools Used

Streamlit, Matplotlib, SciKitLearn, Pandas, HTML, & numpy.

## Workflow

Breanna kickstarted this project by collecting and cleaning the data we had all agreed would best suit our desired implementation. From there, David took that cleaned data and fit it to machine learning models (linear regression, lasso, random forest) to see how our data interacted with these models. After that, Joseph collected the work from these notebooks and converted them into the app.py file and implemented Streamlit to display what was going on throughout the process. Once this was all done, the three of us assessed the results and constructed a presentation to reflect the efficacy of our models.

## Running the Application

If you would like to run this application locally, you will need to clone the repository and ensure each item within the [requirements.txt](requirements.txt) is installed on your machine. Then, double check to ensure streamlit is installed on your machine `pip install streamlit`. Path into the project folder and use the command `streamlit run app.py` to launch the application. 

You could also visit the live build of this project here  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/josephchancey/ca-wildfire-ml/main/app.py)

## Application

![1](https://i.imgur.com/pV3NE2a.png)

![2](https://i.imgur.com/iaoCbwJ.png)

![3](https://i.imgur.com/4DLLQSg.png)

![4](https://i.imgur.com/bXfQpyj.png)

![5](https://i.imgur.com/VsT8Qq4.png)

![6](https://i.imgur.com/xZYpbnf.png)

![7](https://i.imgur.com/uzR2Xz2.png)


## Summary 

From all of this, we can conclude that the Linear Regression model and Lasso model are susceptible to overfitting with our data. The Random Forest classification model was wildly innacurate, but did not suffer from overfitting.
