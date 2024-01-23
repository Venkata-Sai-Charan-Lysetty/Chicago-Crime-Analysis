# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Data 603 Project
# MAGIC Topic: Analysing Chicago crime data
# MAGIC Datasets used:
# MAGIC Chicago City Crime Data(https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
# MAGIC The data i took is from 2013 to present.
# MAGIC First i am creating a table from the the data set
# MAGIC references: 
# MAGIC [1] Ernest-Kiwele. (n.d.). chicago-crime-analysis-apache-spark/chicago-crime-data-on-spark.ipynb at master · ernest-kiwele/chicago-crime-analysis-apache-spark. GitHub. https://github.com/ernest-kiwele/chicago-crime-analysis-apache-spark/blob/master/spark-ml/chicago-crime-data-on-spark.ipynb.
# MAGIC [2] https://bpagare6.github.io/Chicago-Crime-Investigation/

# COMMAND ----------

pip install folium

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/crimes_2013_to_present_csv.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

crime_temp_table = "crimes_2013_to_present_csv_csv"

df.createOrReplaceTempView(crime_temp_table)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `crimes_2013_to_present_csv_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "crimes_2013_to_present_csv_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# selecting only required particular columns.
df = df[['ID', 'Case Number' , 'Date', 'Block', 'Community Area', 'Primary Type', 'Description', 'Location',
                 'Location Description', 'Arrest', 'Domestic', 'Latitude', 'Longitude']]

# COMMAND ----------

crime_df=df

# COMMAND ----------

# displaying the data frame
display(crime_df)

# COMMAND ----------

# knowing the rows and columns of dataframe.
print((crime_df.count(), len(crime_df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC Now i want to rename some columns that i want to use in analysis so that it is easy for me to use the columns in the feature.

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.functions import from_unixtime, unix_timestamp, col
crime_df=crime_df.withColumnRenamed("Community Area","Community_Area")
crime_df=crime_df.withColumnRenamed("Case Number", "Case_Number")
crime_df=crime_df.withColumnRenamed("Primary Type","Primary_Type")
crime_df=crime_df.withColumnRenamed("Location Description", "Location_Description")
crime_df=crime_df.withColumnRenamed("FBI Code", "FBI_Code")
crime_df=crime_df.withColumnRenamed("X Coordinate", "X_Coordinate")
crime_df=crime_df.withColumnRenamed("Y Coordinate", "Y_Coordinate")
crime_df=crime_df.withColumnRenamed("Updated On", "Updated_On")
crime_df = crime_df.cache()

# COMMAND ----------

# To know how many different types of crimes that are registered
dsnt_cnt = crime_df.select("Primary_type").distinct().count()

# Print the distinct count
print(dsnt_cnt)

# COMMAND ----------

# let me see all the columns that i have .
crime_df.select("Primary_Type").distinct().show(n = 35)

# COMMAND ----------

# to print the total crimes occured per year.
from pyspark.sql.functions import col, year, from_unixtime, unix_timestamp
# Set the correct date format
date_format = "MM/dd/yyyy hh:mm:ss a"
crime_df = crime_df.withColumn("Date", from_unixtime(unix_timestamp(col("Date"), date_format)))

# to get the year from the 'Date' column
crime_df = crime_df.withColumn("Year", year(col("Date")))

# filtering the null values pf year .
crime_df = crime_df.filter(col("Year").isNotNull())

# Group by Year and count the number of crimes
crimes_per_year = crime_df.groupBy("Year").count().orderBy("Year")

# Show the results
crimes_per_year.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Convert the PySpark DataFrame to a Pandas DataFrame
crimes_per_year_pd = crimes_per_year.toPandas()

# Create a bar plot of the number of crimes per year
plt.bar(crimes_per_year_pd['Year'], crimes_per_year_pd['count'])

# Set the plot title and labels
plt.title("Number of Crimes occured per Year")
plt.xlabel("Year")
plt.ylabel("The total number of Crimes")

# Set the x-axis ticks to display all years
plt.xticks(crimes_per_year_pd['Year'], rotation=90)

# Display the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Finding top happend crimes

# COMMAND ----------

crime_type = crime_df.groupBy('Primary_Type').count()
crime_type_count = crime_type.orderBy('count', ascending=False)
crime_type_count.show(8)

# COMMAND ----------

# MAGIC %md
# MAGIC To create three bar charts that show the distribution of crimes based on the type of crime, the location description, and the community area.
# MAGIC
# MAGIC The code first creates a temporary view of the crime data and extracts relevant information from the dataset using SQL queries. It queries the count of each type of crime, the count of crimes in each location, and the count of crimes in each community area.
# MAGIC

# COMMAND ----------

crime_df.createOrReplaceTempView('Crime_data')

# COMMAND ----------

# Query the count of each type of crime
import seaborn as sns
crime_type_query = '''SELECT Primary_Type, COUNT(*) AS Count 
                      FROM crime_data 
                      GROUP BY Primary_Type 
                      ORDER BY Count DESC 
                      LIMIT 37'''
crime_type_df = spark.sql(crime_type_query)

# Convert the Spark DataFrame to a Pandas DataFrame
crime_type_pd = crime_type_df.toPandas()

# Set the figure size to 10 inches wide and 6 inches tall
plt.figure(figsize=(10, 6))

# Create a bar chart for the count of each type of crime
sns.barplot(x='Primary_Type', y='Count', data=crime_type_pd)
plt.title('Distribution of Types of Crimes')
plt.xlabel('Type of Crime')
plt.ylabel('Count')
plt.xticks(rotation=40, horizontalalignment='right', fontsize='medium')
plt.show()



# COMMAND ----------

# Query the count of crimes in each community area
query_community_area = '''SELECT Community_Area, COUNT(*) AS Count 
                          FROM crime_data 
                          GROUP BY Community_Area 
                          ORDER BY Count DESC 
                          LIMIT 25'''
community_area_df = spark.sql(query_community_area)

# Convert the Spark DataFrame to a Pandas DataFrame
community_area = community_area_df.toPandas()

# Create a bar chart for the count of crimes in each community area
sns.barplot(x='Community_Area', y='Count', data=community_area)
plt.title('Number of Crimes in Each Community Area (Top 25)')
plt.xlabel('Community Area')
plt.ylabel('Count')
plt.xticks(rotation=40, horizontalalignment='right', fontsize='medium')
plt.show()


# COMMAND ----------

# Query the count of crimes in each location
query_location_description = '''SELECT Location_Description, COUNT(*) AS Count 
                                 FROM crime_data 
                                 GROUP BY Location_Description 
                                 ORDER BY Count DESC 
                                 LIMIT 25'''
location_description_df = spark.sql(query_location_description)

# Convert the Spark DataFrame to a Pandas DataFrame
location_description = location_description_df.toPandas()

# Create a bar chart for the count of crimes in each location
sns.barplot(x='Location_Description', y='Count', data=location_description)
plt.title('Count of Crimes in Each Location (Top 25)')
plt.xlabel('Location Description')
plt.ylabel('Count')
plt.xticks(rotation=40, horizontalalignment='right', fontsize='medium')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC The three graphs provide important information about crime patterns in Chicago. The first graph shows that theft, battery, and criminal damage offenses are much more common than other types of crimes. The second graph indicates that street crimes are the most prevalent, but pubs, restaurants, and retail establishments tend to be secure. The third graph demonstrates a significant variation in the number of reported crimes across different community areas in Chicago, with community areas 25 and 8 having the highest number of reported offenses. The data from these graphs can help law enforcement prioritize their resources and take appropriate measures to reduce crime and improve public safety.

# COMMAND ----------

# let me see which crime occured the most in each year
from pyspark.sql.functions import col, count, max

#  let me group the data by year and crime type, counting the number of occurrences
crime_cnt_year = crime_df.groupBy('Year', 'Primary_Type').agg(count('*').alias('Count'))

# i want to find the maximum crime count for each year
max_crime_cnt_by_each_year = crime_cnt_year.groupBy('Year').agg(max('Count').alias('Max_Count'))

# now joining the two DataFrames to get the crime type with the highest count for each year
hst_crime_by_each_year = crime_cnt_year.join(max_crime_cnt_by_each_year, on=['Year'], how='inner') \
    .filter(col('Count') == col('Max_Count')) \
    .drop('Max_Count') \
    .orderBy('Year')

# Show the results
hst_crime_by_each_year.show()

# COMMAND ----------

from pyspark.sql.types import ArrayType, StructType, StructField, StringType, LongType
from pyspark.sql.functions import col, count, row_number, collect_list, reverse, element_at, udf
from pyspark.sql.window import Window
crime_cnt_by_each_year = crime_df.groupBy('Year', 'Primary_Type').agg(count('*').alias('Count'))

# now i am define a Window specification for ranking crime types within each year
window_spec = Window.partitionBy('Year').orderBy(col('Count').desc())

# Rank the crime types within each year
rank_crime_cnt_by_each_year = crime_cnt_by_each_year.withColumn('Rank', row_number().over(window_spec))

# Filter the DataFrame to get the top 3 crime types for each year
top_3_crimes_by_year = rank_crime_cnt_by_each_year.filter(col('Rank') <= 3).drop('Rank').orderBy('Year', col('Count').desc())

# Grouping by year and collect the crime types and their counts as arrays
grouped_top_3_crimes_by_year = top_3_crimes_by_year.groupBy('Year') \
    .agg(collect_list('Primary_Type').alias('Crime_Types'),
         collect_list('Count').alias('Counts')) \
    .orderBy('Year')

# To define schema of sorted array
sorted_schema = ArrayType(StructType([
    StructField("Count", LongType(), nullable=True),
    StructField("Primary_Type", StringType(), nullable=True)
]))

# I used udf to sort arrays
@udf(sorted_schema)
def sort_arrays(crime_types, counts):
    sorted_tuples = sorted(zip(counts, crime_types), reverse=True)
    return sorted_tuples

# Use the UDF to sort the arrays and extract the elements for each position
top_3_crimes_separated = grouped_top_3_crimes_by_year \
    .withColumn('Sorted_Indices', sort_arrays(col('Crime_Types'), col('Counts'))) \
    .withColumn('First_Highest_Crime', element_at('Sorted_Indices', 1)) \
    .withColumn('Second_Highest_Crime', element_at('Sorted_Indices', 2)) \
    .withColumn('Third_Highest_Crime', element_at('Sorted_Indices', 3)) \
    .drop('Crime_Types', 'Counts', 'Sorted_Indices')

# Show the results
top_3_crimes_separated.show()

# COMMAND ----------

#To plot i used pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Convert the top_3_crimes_separated DataFrame to a Pandas DataFrame
top_3_crimes_pd = top_3_crimes_separated.toPandas()

# Set the figure size and DPI
plt.figure(figsize=(12, 6), dpi=80)

# Define the width of the bars
bar_width = 0.25

# Define the x-axis positions for each group of bars
x1 = top_3_crimes_pd.index
x2 = [x + bar_width for x in x1]
x3 = [x + bar_width for x in x2]

# Plot the bars
# Plot the bars
plt.bar(x1, top_3_crimes_pd['First_Highest_Crime'].apply(lambda x: x['Count']), width=bar_width, label=top_3_crimes_pd['First_Highest_Crime'].apply(lambda x: x['Primary_Type']).iloc[0])
plt.bar(x2, top_3_crimes_pd['Second_Highest_Crime'].apply(lambda x: x['Count']), width=bar_width, label=top_3_crimes_pd['Second_Highest_Crime'].apply(lambda x: x['Primary_Type']).iloc[0])
plt.bar(x3, top_3_crimes_pd['Third_Highest_Crime'].apply(lambda x: x['Count']), width=bar_width, label=top_3_crimes_pd['Third_Highest_Crime'].apply(lambda x: x['Primary_Type']).iloc[0])

# Add labels, a legend, and a title
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.legend(title='Crime Types')
plt.title('Top 3 Crimes by Year')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC To groups a DataFrame by crime type and calculates the total number of arrests for each type using PySpark's sum and when functions. It then orders the resulting DataFrame in descending order by the number of arrests and displays the results.To know which crime has more number of arrests.

# COMMAND ----------

# MAGIC %md
# MAGIC calculating number of crimes for each crime type

# COMMAND ----------

from pyspark.sql.functions import sum, when

# Group by Primary_Type and calculate the sum of arrests
arrests_by_crime = crime_df.groupBy("Primary_Type") \
                             .agg(sum(when(crime_df.Arrest == "true", 1).otherwise(0)).alias("Arrests"))

# Order the data in descending order by number of arrests
arrests_by_crime = arrests_by_crime.orderBy("Arrests", ascending=False)

# Show the results
arrests_by_crime.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert the PySpark DataFrame to a Pandas DataFrame
arrests_pd = arrests_by_crime.toPandas()

# Plot the data as a pie chart
plt.pie(arrests_pd["Arrests"], labels=arrests_pd["Primary_Type"], autopct="%1.1f%%")
plt.title("Number of Arrests by Crime Type in Chicago")

# Show the plot
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert the PySpark DataFrame to a Pandas DataFrame
arrests_pd = arrests_by_crime.toPandas()

# Plot the data as a bar graph
plt.bar(arrests_pd["Primary_Type"], arrests_pd["Arrests"])
plt.xticks(rotation=90)
plt.xlabel("Crime Type")
plt.ylabel("Number of Arrests")
plt.title("Number of Arrests by Crime Type in Chicago")

# Show the plot
plt.show()


# COMMAND ----------

# Convert date column to timestamp and extract year
from pyspark.sql.functions import to_timestamp

n_crime_df = crime_df.withColumn('date_time', to_timestamp('date', 'MM/dd/yyyy hh:mm:ss a'))
n_crime_df = n_crime_df.withColumn('year', year('date_time'))

# Group by year and Arrest and count occurrences
arrest_counts = crime_df.groupBy('year', 'Arrest').agg(count('*').alias('count'))

# Pivot the data to show counts for Arrest and not Arrest separately
arrest_pivot = arrest_counts.groupBy('year').pivot('Arrest', ['true', 'false']).agg(sum('count').alias('count'))

# Display the results
arrest_pivot.show() 

# COMMAND ----------

# sorting by year
arrest_pivot = arrest_counts.groupBy('year').pivot('Arrest', ['true', 'false']).agg(sum('count').alias('count')).orderBy('year')
arrest_pivot.show() 

# COMMAND ----------

from pyspark.sql.functions import col

# Add a new column that contains the sum of arrests and not arrested
arrest_pivot = arrest_pivot.withColumn('total_count', col('true') + col('false'))
# Add a new column that contains the arrest percentage
arrest_pivot = arrest_pivot.withColumn('arrest_percentage', (col('true') / col('total_count')) * 100)


# COMMAND ----------

arrest_pivot.show() 

# COMMAND ----------

import matplotlib.pyplot as plt

# Convert the DataFrame to a Pandas DataFrame for plotting
arrest_pivot_pd = arrest_pivot.toPandas()

# Plot the line graph
plt.plot(arrest_pivot_pd['year'], arrest_pivot_pd['true'], label='Arrested')
plt.plot(arrest_pivot_pd['year'], arrest_pivot_pd['false'], label='Not Arrested')

# Set the chart title and axis labels
plt.title('Arrests vs Not Arrests over Time')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')

# Show the legend and the chart
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC It is clearly visulaized that the arrest efficency is gradually decreased. year 2022 which has lowest efficency . The number of decreased from 2020 to 2021 i feel this is due to covid 19.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Now let us do analysis by taking each crime so that we can draw insights which crime has highest number of arrests.

# COMMAND ----------

from pyspark.sql.functions import sum, count, when

# Calculating the number of arrests and offenses for each primary type of crime
arrest_count_df = (crime_df.groupBy('Primary_Type')
                  .agg(sum(when(col('Arrest') == 'true', 1).otherwise(0)).alias('Total_Number_of_Arrests'),
                       count('*').alias('Total_Number_of_crimes')))

# Calculating the arrest rate for each primary type of crime
arrest_rate_df = (arrest_count_df.withColumn('Arrest_Percentage', 
                                         (col('Total_Number_of_Arrests') / col('Total_Number_of_crimes') * 100))
                             .orderBy('Arrest_Percentage', ascending=False))

# Show the results
arrest_rate_df.show(15, False)


# COMMAND ----------

# MAGIC %md
# MAGIC It is clearly seen that prostitution , gambling ,narcotics and  public indecency has most arrest percentage .

# COMMAND ----------

 
import matplotlib.pyplot as plt

# Convert the DataFrame to a Pandas DataFrame for plotting
arrest_rates_pd = arrest_rate_df.toPandas()

# Set the figure size
plt.figure(figsize=(10, 8))

# Plot the horizontal bar chart
plt.barh(arrest_rates_pd['Primary_Type'], arrest_rates_pd['Arrest_Percentage'])

# Set the chart title and axis labels
plt.title('Arrest Percentage by Crime Type')
plt.xlabel('Arrest Percentage')
plt.ylabel('Crime Type')

# Show the chart
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC The analysis on crimes  effected area by taking area codes and matching with their names.

# COMMAND ----------

display(crime_df)

# COMMAND ----------

area_names = """
01	Rogers Park	
40	Washington Park
02	West Ridge	
41	Hyde Park
03	Uptown	
42	Woodlawn
04	Lincoln Square	
43	South Shore
05	North Center	
44	Chatham
06	Lakeview	
45	Avalon Park
07	Lincoln Park	
46	South Chicago
08	Near North Side	
47	Burnside
09	Edison Park	
48	Calumet Heights
10	Norwood Park	
49	Roseland
11	Jefferson Park	
50	Pullman
12	Forest Glen	
51	South Deering
13	North Park	
52	East Side
14	Albany Park	
53	West Pullman
15	Portage Park	
54	Riverdale
16	Irving Park	
55	Hegewisch
17	Dunning	
56	Garfield Ridge
18	Montclare	
57	Archer Heights
19	Belmont Cragin	
58	Brighton Park
20	Hermosa	
59	McKinley Park
21	Avondale	
60	Bridgeport
22	Logan Square	
61	New City
23	Humboldt Park	
62	West Elsdon
24	West Town	
63	Gage Park
25	Austin	
64	Clearing
26	West Garfield Park 	
65	West Lawn
27	East Garfield Park	
66	Chicago Lawn
28	Near West Side	
67	West Englewood
29	North Lawndale	
68	Englewood
30	South Lawndale	
69	Greater Grand Crossing
31	Lower West Side	
70	Ashburn
32	Loop	
71	Auburn Gresham	
33	Near South Side	
72	Beverly
34	Armour Square	
73	Washington Heights
35	Douglas	
74	Mount Greenwood
36	Oakland	
75	Morgan Park
37	Fuller Park	
76	O'Hare
38	Grand Boulevard	
77	Edgewater
39	Kenwood	
"""

# COMMAND ----------

import re
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

area_names_dict = dict(re.findall(r'(\d+)\s+(.+)', area_names))
area_names_udf = udf(lambda area_code: area_names_dict.get(area_code, ""), StringType())

crime_df = crime_df.withColumn("area names", area_names_udf(crime_df["Community_Area"]))


# COMMAND ----------

new_df = crime_df.select("Area Names", "Arrest", "Community_Area")

# COMMAND ----------

# import necessary modules
from pyspark.sql.functions import col

# remove null values
new_df = new_df.filter(col("Area Names").isNotNull() & col("Arrest").isNotNull() & col("Community Area").isNotNull())


# COMMAND ----------

display(new_df)

# COMMAND ----------

# import necessary modules
from pyspark.sql.functions import col, count

# calculate total number of arrests per area
total_arrests = new_df.filter(col("Arrest") == "true").groupBy("Area Names").agg(count("Arrest").alias("Total Arrests"))

# calculate total number of non-arrest incidents per area
total_non_arrests = new_df.filter(col("Arrest") == "false").groupBy("Area Names").agg(count("Arrest").alias("Total Non-Arrests"))

# join the two DataFrames
total_incidents = total_arrests.join(total_non_arrests, "Area Names", "outer").fillna(0)

# calculate percentage of arrests per area
arrest_percentage = total_incidents.withColumn("Arrest Percentage", col("Total Arrests")/ (col("Total Arrests") + col("Total Non-Arrests")) * 100)

# display the result
arrest_percentage.show()


# COMMAND ----------

import matplotlib.pyplot as plt

# Convert PySpark DataFrame to Pandas DataFrame
pandas_df = total_incidents.toPandas()

# Set "Area Names" as index
pandas_df.set_index("Area Names", inplace=True)

# Create stacked bar chart
ax = pandas_df.plot(kind="bar", stacked=True, figsize=(12, 6))

# Set chart title and axis labels
ax.set_title("Total Incidents by Area")
ax.set_xlabel("Area Names")
ax.set_ylabel("Total Incidents")

# Set legend labels
ax.legend(["Total Non-Arrests", "Total Arrests"])

# Show the chart
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC Austin has more arrest percentage and Austin is place with more crimes recorded.
# MAGIC Forest glen and burnside are most safest places.
# MAGIC

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/crimes_2013_to_present_csv.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

crime_temp_table = "crimes_2013_to_present_csv_csv"

df.createOrReplaceTempView(crime_temp_table)

# COMMAND ----------

crime_df=df

# COMMAND ----------

display(crime_df)

# COMMAND ----------

# Display the schema of the DataFrame
crime_df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC whether there is a correlation between the frequency and type of crimes committed and the time of day. Specifically, we may want to investigate whether there are differences in the frequency and types of crimes committed at different times of day, and if these differences vary across different types of crimes.

# COMMAND ----------

display(crime_df)

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.functions import from_unixtime, unix_timestamp, col
crime_df=crime_df.withColumnRenamed("Community Area","Community_Area")
crime_df=crime_df.withColumnRenamed("Case Number", "Case_Number")
crime_df=crime_df.withColumnRenamed("Primary Type","Primary_Type")
crime_df=crime_df.withColumnRenamed("Location Description", "Location_Description")
crime_df=crime_df.withColumnRenamed("FBI Code", "FBI_Code")
crime_df=crime_df.withColumnRenamed("X Coordinate", "X_Coordinate")
crime_df=crime_df.withColumnRenamed("Y Coordinate", "Y_Coordinate")
crime_df=crime_df.withColumnRenamed("Updated On", "Updated_On")
crime_df = crime_df.cache()

# COMMAND ----------

crime_df_n = crime_df[['ID', 'Case_Number' , 'Date', 'Block', 'Primary_Type', 'Description', 'Location',
                 'Location_Description', 'Arrest', 'Domestic', 'Latitude', 'Longitude']]

# COMMAND ----------

pandas_df = crime_df_n.toPandas()

# print the Pandas DataFrame
pandas_df.head(3)

# COMMAND ----------

print('shape of data set before removing null values : ', pandas_df.shape)
pandas_df.dropna(inplace=True)
print('shape of data set before removing null values: ', pandas_df.shape)

# COMMAND ----------

# Converting dates into pandas datetime format
pandas_df.Date = pd.to_datetime(pandas_df.Date, format='%m/%d/%Y %I:%M:%S %p')
# Setting the index to be the date that will help us a lot
pandas_df.index = pd.DatetimeIndex(pandas_df.Date)

# COMMAND ----------

# MAGIC %md
# MAGIC first identifies the least frequent values in the columns Location_Description and Description, and changes them to the label "OTHER". Then, it converts the Primary_Type, Location_Description, and Description columns into categorical variables.

# COMMAND ----------

location_change  = list(pandas_df['Location_Description'].value_counts()[20:].index)
description_change = list(pandas_df['Description'].value_counts()[20:].index)
pandas_df.loc[pandas_df['Location_Description'].isin(location_change) , pandas_df.columns=='Location Description'] = 'OTHER'
pandas_df.loc[pandas_df['Description'].isin(description_change) , pandas_df.columns=='Description'] = 'OTHER'
pandas_df['Primary_Type']         = pd.Categorical(pandas_df['Primary_Type'])
pandas_df['Location_Description'] = pd.Categorical(pandas_df['Location_Description'])
pandas_df['Description']          = pd.Categorical(pandas_df['Description'])

# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(11,5))
pandas_df.resample('M').size().plot(legend=False)
plt.title('Number of crimes per month (2013 - 2023)')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC I did not find any good findings when i see crimes spread over months so i take each crime and do analysis over the years.

# COMMAND ----------

# MAGIC %md
# MAGIC now i want to know the crime trends of each crime.

# COMMAND ----------

import pandas as pd
import numpy as np

# create a DataFrame and pivot it
n_df = pandas_df.pivot_table('ID', aggfunc=np.size, columns='Primary_Type', index=pandas_df.index.date, fill_value=0)

# convert the index to a datetime index
n_df.index = pd.DatetimeIndex(n_df.index)

# plot the rolling sum
plot = n_df.rolling(365).sum().plot(figsize=(15, 50), subplots=True, layout=(-1, 2), sharex=False, sharey=False)


# COMMAND ----------

# MAGIC %md
# MAGIC From the above graphs :
# MAGIC It is clearly observed that all the crimes decreased in the years 2020 and 2021.
# MAGIC Crimes like criminal sexual assult,stalking and weapon violation have increasing trends .
# MAGIC These crimes are increasing over the years in Chicago.
# MAGIC The crimes like Burglary , gambling, kidnapping , narcotics cases have decreasing over the years.
# MAGIC

# COMMAND ----------

days = ['Monday','Tuesday','Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
pandas_df.groupby([pandas_df.index.dayofweek]).size().plot(kind='barh')
plt.ylabel('Days in a week')
plt.yticks(np.arange(7), days)
plt.xlabel('Number of crimes happend')
plt.title('count of crimes per each day')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC It is clearly seen that most crimes are happening on weekends.

# COMMAND ----------

pandas_df.groupby([pandas_df.index.month]).size().plot(kind='barh')
plt.ylabel('Months of the year')
plt.xlabel('Number of crimes')
plt.title('Number of crimes by month of the year')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The least number of crimes occurred in the month of December and i predict this is due to festival month. even February less number of crimes occurred but the point to be considered is February have 2 days less than other months.

# COMMAND ----------

plt.figure(figsize=(8, 10))
pandas_df.groupby([pandas_df['Location_Description']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of Crimes by location')
plt.ylabel('Crime Location')
plt.xlabel('Number of Crimes')
plt.show()

# COMMAND ----------

plt.figure(figsize=(8, 10))
pandas_df.groupby([pandas_df['Location_Description']]).size().sort_values(ascending=True).tail(20).plot(kind='barh')
plt.title('Number of Crimes by each location')
plt.ylabel('Crime Location name')
plt.xlabel('Number of Crimes occured')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC It is clearly observed that most of the crimes occurred on streets , residence apartment and sidewalks.
# MAGIC Schools and business places are the places with least number of crimes.

# COMMAND ----------

hour_by_location = pandas_df.pivot_table(values='ID', index='Location_Description', columns=pandas_df.index.hour, aggfunc=np.size).fillna(0)
hour_by_type     = pandas_df.pivot_table(values='ID', index='Primary_Type', columns=pandas_df.index.hour, aggfunc=np.size).fillna(0)
hour_by_week     = pandas_df.pivot_table(values='ID', index=pandas_df.index.hour, columns=pandas_df.index.day_name(), aggfunc=np.size).fillna(0)
hour_by_week     = hour_by_week[days].T # just reorder columns according to the the order of days
dayofweek_by_location = pandas_df.pivot_table(values='ID', index='Location_Description', columns=pandas_df.index.dayofweek, aggfunc=np.size).fillna(0)
dayofweek_by_type = pandas_df.pivot_table(values='ID', index='Primary_Type', columns=pandas_df.index.dayofweek, aggfunc=np.size).fillna(0)
location_by_type  = pandas_df.pivot_table(values='ID', index='Location_Description', columns='Primary_Type', aggfunc=np.size).fillna(0)


# COMMAND ----------

from sklearn.cluster import AgglomerativeClustering as AC

def scale_dataframe(df, axis=0):
    '''
    Scale numerical values in a DataFrame to have a mean of zero and unit variance.
    '''
    return (df - df.mean(axis=axis)) / df.std(axis=axis)

def plot_heatmap(df, ix=None, cmap='bwr'):
    '''
    Plot a heatmap to show temporal patterns.
    '''
    if ix is None:
        ix = np.arange(df.shape[0])
    plt.imshow(df.iloc[ix, :], cmap=cmap)
    plt.colorbar(fraction=0.03)
    plt.yticks(np.arange(df.shape[0]), df.index[ix])
    plt.xticks(np.arange(df.shape[1]))
    plt.grid(False)
    plt.show()
    
def scale_and_cluster(df, ix=None):
    '''
    Scale each row in a DataFrame, cluster the rows, and plot a heatmap of the scaled data.
    '''
    # Scale the data
    df_marginal_scaled = scale_dataframe(df.T).T
    # Cluster the rows and sort them according to the clustering labels to improve heatmap visualization
    if ix is None:
        ix = AC(4).fit(df_marginal_scaled).labels_.argsort()
    # Clip the scaled values to a symmetric range around zero to improve the color contrast in the heatmap
    cap = np.min([np.max(df_marginal_scaled.values), np.abs(np.min(df_marginal_scaled.values))])
    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)
    # Plot the heatmap
    plot_heatmap(df_marginal_scaled, ix=ix)

def normalize_dataframe(df):
    '''
    Normalize each column in a DataFrame to have values between 0 and 1.
    '''
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(15,12))
scale_and_cluster(hour_by_type)

# COMMAND ----------

# MAGIC %md
# MAGIC The investigation looks at crime hotspots to see if there are any changes between daytime and nighttime crime rates. The heatmap shows that certain areas have greater crime rates during the day, with a few of them peaking at nine in the morning. On the other hand, certain places witness a rise in crime at night. Notably, the heatmap reveals that crime at bars rises in the early morning.
# MAGIC

# COMMAND ----------

plt.figure(figsize=(12,4))
scale_and_cluster(hour_by_week, ix=np.arange(7))

# COMMAND ----------

# MAGIC %md
# MAGIC The weekdays are represented on the x-axis of the heatmap that is displayed below, which reveals some interesting findings. The middle and lower portions of the heatmap, respectively, show that some crime categories tend to occur more frequently on particular days, such as Fridays or Saturdays, while others tend to occur more frequently on weekdays or weekends.
# MAGIC

# COMMAND ----------

plt.figure(figsize=(17,17))
scale_and_cluster(dayofweek_by_type)

# COMMAND ----------

df = location_by_type.sum().nlargest(10).to_frame().join(location_by_type).fillna(0)
df_normalized = normalize_dataframe(df.iloc[:, 1:])
df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
ix = AC(3).fit(df_normalized.T).labels_.argsort()
plt.figure(figsize=(17,13))
plt.imshow(df_normalized.T.iloc[ix,:], cmap='Reds')
plt.colorbar(fraction=0.03)
plt.xticks(np.arange(df_normalized.shape[0]), df_normalized.index, rotation='vertical')
plt.yticks(np.arange(df_normalized.shape[1]), df_normalized.columns)
plt.title('location frequency for top 10 crimes')
plt.grid(False)
plt.show()



# COMMAND ----------

def Loc_extractor(Raw_Str):
    preProcess = Raw_Str[1:-1].split(',')
    lat =  float(preProcess[0])
    long = float(preProcess[1])
    return (lat, long)

# COMMAND ----------

import pandas as pd
unique_locations = pandas_df['Location'].value_counts()

crime_index = pd.DataFrame({"Raw_String" : unique_locations.index, "ValueCount":unique_locations})
crime_index.index = range(len(unique_locations))
crime_index.head()

# COMMAND ----------

crime_index['LocationCoord'] = crime_index['Raw_String'].apply(Loc_extractor)
crime_index  = crime_index.drop(columns=['Raw_String'], axis = 1)
print(crime_index.shape[0])

# COMMAND ----------

import folium
chicago_map_crime = folium.Map(location=[41.895140898, -87.624255632],
                        zoom_start=14,
                        tiles="CartoDB dark_matter")

for i in range(1000):
    lat = crime_index['LocationCoord'].iloc[i][0]
    long = crime_index['LocationCoord'].iloc[i][1]
    radius = crime_index['ValueCount'].iloc[i] / 45
    
    if crime_index['ValueCount'].iloc[i] > 2000:
        color = "#FF4500"
    else:
        color = "#008080"
    
    popup_text = """Latitude : {}<br>
                Longitude : {}<br>
                Criminal Incidents : {}<br>"""
    popup_text = popup_text.format(lat,
                               long,
                               crime_index['ValueCount'].iloc[i]
                               )
    folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(chicago_map_crime)
    
chicago_map_crime


# COMMAND ----------


