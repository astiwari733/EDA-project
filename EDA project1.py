#!/usr/bin/env python
# coding: utf-8

# # Problem Definition : Predict the total ride duration of taxi trips in New York City.

# Hypothesis Generated:
# 
# Trip duration will be affected by following -
# 
# 1.Pickup datetime of trips
# 
# 2.location mismatched
# 
# 3.Number of passengers(more than a specified number)
# 
# 4.Traffic condition
# 
# 5.Road condition
# 
# 6.Weather Condition
# |
# 7.Engine,tyre related problem

# # Data Loading After Extraction:

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action = 'ignore')
import datetime as dt


# In[2]:


data = pd.read_csv('Desktop/nyc_taxi_trip_duration.csv')


# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


# converting strings to datetime features

data['pickup_datetime'] = pd.to_datetime(data.pickup_datetime)
data['dropoff_datetime'] = pd.to_datetime(data.dropoff_datetime)


# In[6]:


data.isnull().sum()


# In[7]:


get_ipython().system('pip install geopy')


# In[44]:


from geopy.distance import great_circle
#( hevershine formula for calculating surface distances)


# In[9]:


# Defining a function to take coordinates as inputs and return us distance
def cal_distance(pickup_lat,pickup_long,dropoff_lat,dropoff_long):
    start_coordinates=(pickup_lat,pickup_long)
    stop_coordinates=(dropoff_lat,dropoff_long)
    return great_circle(start_coordinates,stop_coordinates).km


# In[10]:


data['distance'] = data.apply(lambda x: cal_distance(x['pickup_latitude'],x['pickup_longitude'],
                                                     x['dropoff_latitude'],x['dropoff_longitude']), axis=1)


# In[11]:


data.head()


# #Target Variable:
# 
# Let us analyse our target variable trip_duration.

# In[12]:


# Trip duration in hours

data['trip_duration'].mean()/3600


# In[13]:


data['trip_duration'].std()/3600


# In[14]:


data['trip_duration'].max()/3600


# In[15]:


data['trip_duration'].median()/3600


# In[16]:


data['trip_duration'].describe()/3600


# In[17]:


sns.distplot(data['trip_duration'], bins = 100)
plt.show()


# In[18]:


sns.kdeplot(data['trip_duration'], shade = True)
plt.show()


# Since there is a huge outlier, we will take log of the trip_duration for visualising it better.

# In[19]:


data['log_trip_duration'] = np.log(data['trip_duration'].values + 1)
sns.kdeplot(data['log_trip_duration'], shade = True)
plt.show()


# 1.The trip duration of rides are forming almost normal curve.
# 
# 2.As noticed earlier, there is an outlier present near 12.
# 
# 3.Also there are very short rides present which are of less than 10 seconds, which are suspicious.

# # Univariate analysis

# We will check our hypothesis using univariate analysis of variables.

# In[20]:


data['distance'].value_counts()


# We see there are 2893 trips with 0 km distance.
# 
# Possible reasons for 0 km distance can be:
# 
# 1.The dropoff location couldn’t be tracked.
# 
# 2.The driver deliberately took this ride to complete a target ride number.
# 
# 3.The passengers canceled the trip.

# In[21]:


plt.figure(figsize=(19, 6))

plt.subplot(1, 3, 1)
sns.countplot(data['passenger_count'])
plt.xlabel('Passenger Count')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.countplot(data['vendor_id'])
plt.xlabel('Vendor ID')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.countplot(data['store_and_fwd_flag'])
plt.xlabel('Store And Fwd Flag')
plt.ylabel('Frequency')
plt.show()


# Observation -
# 
# *Most frequent rides include only 1 passenger, while some of the rides include 7 to 9 passengers too and they are very low in number.
# *Most of the rides have been completed by vendor 2 as compared to vendor 1.
# *There is almost no storing of data taking place in the taxi and being updated later. (Y - Yes, N - No)

# Observing trend in pickup datetime of trips

# In[22]:


data['day_of_week'] = data['pickup_datetime'].dt.weekday
data['hour_of_day'] = data['pickup_datetime'].dt.hour


# In[23]:


plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.countplot(data['day_of_week'])
plt.xlabel('Days of Week')
plt.ylabel('Total number of pickups')

plt.subplot(1, 2, 2)
sns.countplot(data['hour_of_day'])
plt.xlabel('Hour of Day')
plt.ylabel('Total number of pickups')
plt.show()


# Observation -
# 
# 1.Observing the above trend, we can see that the most of the rides are on Thursday, while on the weekends, there is lowest number of rides (0 is Sunday).
# 
# 2.Total number of rides in 24 hours are mostly around 18-19 hours, i.e. evening. While in the morning peak hour, it is lower than expected.
# 
# 
# 
# 
# Observing location of pickup and dropoff

# In[24]:


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(10, 10), sharex = False, sharey = False)
sns.despine(left=True)
sns.distplot(data['pickup_latitude'].values, label = 'pickup_latitude', color="b", bins = 100, ax=axes[0,0])
sns.distplot(data['pickup_longitude'].values, label = 'pickup_longitude', color="r", bins =100, ax=axes[1,0])
sns.distplot(data['dropoff_latitude'].values, label = 'dropoff_latitude', color="b", bins =100, ax=axes[0,1])
sns.distplot(data['dropoff_longitude'].values, label = 'dropoff_longitude', color="r", bins =100, ax=axes[1,1])
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


# Latitude - Blue
# 
# Longitude - Red
# 
# Observation -
# 
# 1.Pickup and drop latitudes are denser around 40 to 41, and longitude are denser around -74 to -73.
# 
# 2.Extreme values are present in the data which depicts higher value of distance.
# 
# 
# We will remove these outliers or extreme values and observe the data closely.

# In[25]:


# Removal of outliers.

data = data.loc[(data.pickup_latitude > 40.6) & (data.pickup_latitude < 40.9)]
data = data.loc[(data.dropoff_latitude>40.6) & (data.dropoff_latitude < 40.9)]
data = data.loc[(data.dropoff_longitude > -74.05) & (data.dropoff_longitude < -73.7)]
data = data.loc[(data.pickup_longitude > -74.05) & (data.pickup_longitude < -73.7)]
data_new = data.copy()


# In[26]:


# Visualisation after removing outliers

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(data_new['pickup_latitude'].values, label = 'pickup_latitude',color="b",bins = 100, ax=axes[0,0])
sns.distplot(data_new['pickup_longitude'].values, label = 'pickup_longitude',color="r",bins =100, ax=axes[1,0])
sns.distplot(data_new['dropoff_latitude'].values, label = 'dropoff_latitude',color="b",bins =100, ax=axes[0, 1])
sns.distplot(data_new['dropoff_longitude'].values, label = 'dropoff_longitude',color="r",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


# Observation -
# 
# 
# Most of the rides are located between these locations, apart from few outliers outside the above range.

# # Bivariate analysis

# We will compare each of the variables with the target variable, 'trip_duration', in order to derive the relation between them.

# In[27]:


data.columns


# In[28]:


data.head()


# # Trip duration and Weekdays

# Do the trips on weekdays have higher trip duration?
# 
# 1.We will use Time series plot, 'tsplot', to plot between datetime and a continuous variable.
# 
# 2.For plotting each day, we will take central tendency, i.e. median of each day's trip_duration and plot it against the days of week.

# In[29]:


average_duration_day = pd.DataFrame(data.groupby(['day_of_week'])['trip_duration'].median())
average_duration_day.reset_index(inplace = True)
average_duration_day['unit']=1


# In[30]:


average_duration_day


# In[31]:


get_ipython().system('pip install seaborn==0.9.0')


# In[35]:


sns.tsplot(data=average_duration_day, time="day_of_week", unit = 'unit', value="trip_duration");


# Observation -
# 
# Longest trip duration has been observed on Wednesday.
# 
# Opposite to expectation, trip duration on weekends are lowest.

# In[36]:


average_duration_hour = pd.DataFrame(data.groupby(['hour_of_day'])['trip_duration'].median())
average_duration_hour.reset_index(inplace = True)
average_duration_hour['unit']=1


# In[37]:


sns.tsplot(data=average_duration_hour, time='hour_of_day', unit = 'unit', value='trip_duration')
plt.show()


# Observation -
# 
# 
# Trip duration during early morning are comparatively lesser which may be because of low traffic, and highest during evening peak hour.
# 
# There is a correlation between the number of pickups and trip duration as it follows the similar trend.

# # Trip Duration and Vendor ID

# Do the vendors have any impact on the trip duration?
# We will check the duration of trip for each vendor.

# In[38]:


plt.figure(figsize=(15, 6))
sns.boxplot(x = 'vendor_id', y = 'trip_duration', data = data)
plt.show()


# Observation -
# 
# 
# As we can see, there is a huge outlier/extreme point for vendor 1 as compared to vendor 2.
# 
# 
# Let's remove the outliers and observe the above data closely.

# In[39]:


plt.figure(figsize=(15, 6))
trip_no_outliers = data[data['trip_duration'] < 50000]
sns.boxplot(x = 'vendor_id', y = 'trip_duration', data = trip_no_outliers)
plt.show()


# Observation -
# 
# 
# Here we can see that vendor 2 has much outliers than vendor 1, and we know that the median for trip duration lies around 600.

# # Trip duration and Passenger Count

# Are passengers with higher count, taking longer duration to complete the trip?
# 
# We will check the trend in duration of trips as compared to the number of passengers for the trip.

# In[40]:


data.passenger_count.value_counts()


# As we know the median of trip_duration lies around 600 and we have huge outliers present in the trip_duration data, we will consider the trip_duration data of only less than 10,000 seconds.

# In[41]:


plt.figure(figsize=(16, 6))
trip_duration_new = data[data['trip_duration'] < 10000]
sns.boxplot(x="passenger_count", y="trip_duration", data = trip_duration_new)
plt.show()


# Observation -
# 
# 1.There are few trips recorded without any passenger.
# 
# 2.Trips with 1 and 2 numbers of passengers have high amount of outliers present.
# 
# 3.As the number of passengers are increasing, the outliers are decreasing.

# # Correlation Heatmap

# We will check the correlations amongst all of the variables using heatmap.

# In[42]:


data.head()


# In[43]:


# From the above dataset, we will drop those columns which are irrelevant with our target variable trip_duration.

data_drop = data.drop(['id', 'pickup_datetime','dropoff_datetime', 'log_trip_duration'], axis=1)

plt.figure(figsize=(12, 6))
corr = data_drop.corr('pearson')
sns.heatmap(corr, linewidth=2)
plt.show()


# Observation -
# 
# 
# From the above correlation heatmap, we see that the latitude and longitude have higher correlation with the target as compared to the others.

# # Conclusion

# 1.The trip duration of rides are forming almost normal curve.
# 
# 2.As noticed earlier, there is an outlier present near 12.
# 
# 3.Also there are very short rides present which are of less than 10 seconds, which are suspicious.
# 
# 4.Most frequent rides include only 1 passenger, while some of the rides include 7 to 9 passengers too and they are very low in number.
# 
# 5.Most of the rides have been completed by vendor 2 as compared to vendor 1.
# 
# 6.There is almost no storing of data taking place in the taxi and being updated later. (Y - Yes, N - No)
# 
# 7.Observing the above trend, we can see that the most of the rides are on Thursday, while on the weekends, there is lowest number of rides (0 is Sunday).
# 
# 8.Total number of rides in 24 hours are mostly around 18-19 hours, i.e. evening. While in the morning peak hour, it is lower than expected.
# 
# 9.Pickup and drop latitudes are denser around 40 to 41, and longitude are denser around -74 to -73.
# 
# 10.Extreme values are present in the data which depicts higher value of distance.
# 
# 11.Most of the rides are located between these locations, apart from few outliers outside the above range.
# 
# 12.Longest trip duration has been observed on Wednesday.
# 
# 13.Opposite to expectation, trip duration on weekends are lowest.
# 
# 14.Trip duration during early morning are comparatively lesser which may be because of low traffic, and highest during evening peak hour.
# 
# 15.There is a correlation between the number of pickups and trip duration as it follows the similar trend.
# 
# 16.Here we can see that vendor 2 has much outliers than vendor 1, and we know that the median for trip duration lies around 600.
# 
# 17.There are few trips recorded without any passenger.
# 
# 18.Trips with 1 and 2 numbers of passengers have high amount of outliers present.
# 
# 19.As the number of passengers are increasing, the outliers are decreasing.
# 
# 20.From the above correlation heatmap, we see that the latitude and longitude have higher correlation with the target as compared to the others.

# In[ ]:




