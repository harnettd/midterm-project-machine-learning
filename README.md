# Data Science Midterm Project: Predicting House Prices

## Project/Goals

The goal of this project was to develop and deploy a machine-learning algorithm to predict home sales prices in the United States.

## Process


### Collect and Clean Data

The data was provided to us in the form of 250 JSON files, each containing information on a collection of home sales in some city in the United States. In general, each city of the dataset had several corresponding JSON files. We looped through all of the JSON files, collecting information on individual home sales including date, sale price, location, home description, and home tags and flags. We then converted the resulting dataset to a pandas DataFrame.

The DataFrame contained many nearly-empty, redundant, and/or irrelevant columns which were dropped. Also, the DataFrame contained some rows without a sales price. These rows were dropped as sales price was the intended target of the followup machine-learning analysis. Furthermore, the DataFrame contained many missing values which we imputed using either 0, False, a median, or a mode as appropriate.

We also engineered a handful of new features. Some features were engineered by combining several existing features together into a single new one. As an example, we merged the existing features 'community_outdoor_space', 'park', and 'trails' into a new feature, 'near_outdoors'. We also added a new feature, 'season', based on the time of year of the home sale. We note that a feature related to location really ought to be included in the analysis. We decided to use median sale price by postal code. However, to prevent data leakage, this feature must be added later on in the process, *after* the train-test split.

We one-hot encoded categorical features and converted Boolean features to integer features in preparation for a machine-learning analysis.

We plotted some univariate EDA visualizations, *i.e.,* box plots and histograms. From these, we noticed that our dataset contained many outliers. Rather than trim them all, we decided to trim those outliers stemming from features with at least a moderate correlation with home sales price (erring on the side of caution). Choosing those features was an iterative process based on correlation matrix elements and heat maps.

### (your step 2)



### (your step 2)



## Results
(fill in how your model performed)



## Challenges 

Challenges we faced while working through this project included:
- deciding on a suitable strategy for dealing with outliers
- feature engineering



## Future Goals
(what would you do if you had more time?)
