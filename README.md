# AI Hack 2018 Git Repository - Road Accidents Data Analysis
### Group Members:
[Rajat Rasal](https://github.com/RajatRasal)

[Ashly Lau](https://github.com/ashlylau)

[Avish](https://github.com/avishvj)

[Sunny](https://github.com/wendysun1)

[Vinul](https://github.com/VinDel77)



### Challenge: UK Car Accidents

###### Content and Source of Data:

This dataset provides detailed information about the circumstances of personal injury road accidents in UK in 2015. The accidents were recorded using STATA19 accident reporting form by the police. The source of this dataset is Open Data UK.

The dataset has 285332 rows and 70 features. A detailed glossary which explains each feature is attached.

The aim of our project was to come up with some sort of model that would predict the class of severity of accident casualties. To do this, we thought of various models we could use, including using a Time Series Analysis using RNN, and using a Random Tree Classifier to make sense of the various features given to us in the data. We soon also realised that a major part of this task was cleaning up the data to remove noise and make it more analysable.

Here is the challenge documentation: https://harrisonzhu508.github.io/Documentation/ACCIDENTS.html 


### Initial Analysis

One of the things we did first was to analyse the data given to try to find some trends. We plotted some of these trends to visualise them better:
![graph1](Graphs\ \&\ Pictures//Correlation\ Graphs//casualty_severity_to_age_band_of_casualty.jpg)
![](Graphs\ \&\ Pictures//Correlation\ Graphs//casualty_severity_to_day_of_week.jpg)

Then, to visualise the geographical representation of these accidents over time, we replaced individual longitudes and latitudes based on mean value of each police force's location, to plot frequencies by sector.

![Frequency of Incidents by Location]('Graphs & Pictures/Pictures/uk_map.png')

We then also categorised the data by month, and plotted these results on 12 graphs, to show change over time.



### Data Cleaning

The first round of cleaning the data came from us realising that many columns had a large percentage of NaN values. We resolved this by dropping the columns that had more than 40% NaN values. 

We then used a pearson correlation function to iterate through the various features, finding the pairs of features that had a correlation value of more than 0.9. We deduced that because they are so closely correlated, having two such features would be redundant and leaving it in would actually add more noise to our data.

Here are some of the graphs that we plotted:
![](Graphs\ \&\ Pictures//Correlation\ Graphs//age_band_of_driver_to_age_of_driver.jpg)
![](Graphs\ \&\ Pictures//Correlation\ Graphs//No_of_Vehicles_involved_unique_to_accident_index_to_number_of_vehicles.jpg)
![](Graphs\ \&\ Pictures//Correlation\ Graphs//police_force_to_local_authority_(district).jpg)

Next, for the rows with many null values, we used a Random Forest Classifier to fill in the null values. The RFC generates many different decision trees which use different vairables from our initial data set. SciKit Learn functions were used for this task.

The final step of data cleaning was removing outliers to avoid under-fitting. The way we did this was finding columns which had continuous data and then removed any rows which had data that was above or below two standard deviations from the mean.

### PCA



### Further Analysis
We used a RFC from SciKit Learn to factor in how a change in time affects the severity. We merged our initially cleaned up data to another pandas dataframe with the dates in sequential order. Our classifier gave us the following confusion matrix:
![](Graphs\ \&\ Pictures//Correlation\ Graphs//RandomForestClassifierResults.png)

