## Project title
Crime patterns among major metropolitan in U.S 
Deep dive into the cities: Chicago, Boston, New York City, San Francisco

## Team members
Nick Bergeland
Molly Chao
Alesia Stewart


## Project description
Comparing crime trends and patterns among four major cities. Variables analayzed include types of crime, the times of occurrence (month, time), areas of distributions (to inform the police patrol and civilians repercussions). In addition, to explore if there is any common behaviors among different cities regarding crimes. 

## Research questions to answer
What cities occurs most/least crimes?
What type of crimes occurs the most?
What time/seasons of crimes occurs the most?
What area of crimes occurs the most?
What are the demographics breakdown of the arrests?
How many cases are closed/open in the given year?


## Dataset to be used
Chicago Crime data: https://data.cityofchicago.org/Public-Safety/Crimes-2019/w98m-zvie
New York Crime data: https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
San Francisco: 
https://www.kaggle.com/psmavi104/san-francisco-crime-data
Boston Crime data: 
https://www.kaggle.com/AnalyzeBoston/crimes-in-boston

## Rough breakdown of the tasks
Clean up the each dataset, and merge all the meaning data into one dataset
Take each question and use jupyter notebook and pandas to analyze
Use matplotlib to create data visualizations of the trend
Use google map api to generate the heat map for certain findings
Create the report and presentation

## Code Notebook

import os
import csv
import pandas as pd
import numpy as np
import datetime
import requests
import json
import matplotlib.pyplot as plt
import gmaps
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
#plotly
import plotly.graph_objects as go
import plotly as pyo
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import plot, iplot, init_notebook_mode

# Make plotly work with Jupyter notebook
init_notebook_mode()
#google map api

from config import g_key
#print(g_key)
chi_crime = "Crimes_-_2019.csv"
#import and print dataset
df_chi_crime = pd.read_csv(chi_crime)
df_chi_crime
ID	Case Number	Date	Block	IUCR	Primary Type	Description	Location Description	Arrest	Domestic	...	Ward	Community Area	FBI Code	X Coordinate	Y Coordinate	Year	Updated On	Latitude	Longitude	Location
0	11933698	JC561524	12/27/2019 07:13:00 AM	013XX S SAWYER AVE	502P	OTHER OFFENSE	FALSE/STOLEN/ALTERED TRP	STREET	False	False	...	24.0	29.0	26	1154911.0	1893543.0	2019	01/03/2020 03:59:34 PM	41.863694	-87.706808	(41.86369439, -87.706807621)
1	11933991	JC562233	12/27/2019 04:22:00 PM	049XX W QUINCY ST	2826	OTHER OFFENSE	HARASSMENT BY ELECTRONIC MEANS	STREET	False	False	...	28.0	25.0	26	1143688.0	1898552.0	2019	01/03/2020 03:59:34 PM	41.877657	-87.747882	(41.877657084, -87.747881613)
2	11934247	JC562471	12/27/2019 09:25:00 PM	042XX W JACKSON BLVD	0486	BATTERY	DOMESTIC BATTERY SIMPLE	APARTMENT	False	False	...	28.0	26.0	08B	1148189.0	1898349.0	2019	01/03/2020 03:59:34 PM	41.877015	-87.731360	(41.877014589, -87.731360167)
3	11934527	JC561811	12/27/2019 03:00:00 AM	015XX N MASSASOIT AVE	0560	ASSAULT	SIMPLE	STREET	False	False	...	29.0	25.0	08A	1137722.0	1909754.0	2019	01/03/2020 03:59:34 PM	41.908506	-87.769517	(41.908506374, -87.76951727)
4	11934830	JC563144	12/27/2019 10:50:00 PM	107XX S EGGLESTON AVE	0820	THEFT	$500 AND UNDER	DRIVEWAY - RESIDENTIAL	False	False	...	34.0	49.0	06	1175192.0	1833678.0	2019	01/03/2020 03:59:34 PM	41.698988	-87.634145	(41.698988087, -87.634144583)
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
254437	24551	JC271105	05/20/2019 06:07:00 PM	078XX S STATE ST	0110	HOMICIDE	FIRST DEGREE MURDER	STREET	False	False	...	6.0	69.0	01A	1177650.0	1852817.0	2019	05/27/2019 04:12:56 PM	41.751453	-87.624568	(41.751452878, -87.624567581)
254438	24554	JC274572	05/23/2019 01:00:00 AM	069XX S MAPLEWOOD AVE	0110	HOMICIDE	FIRST DEGREE MURDER	PORCH	False	False	...	17.0	66.0	01A	1160573.0	1858532.0	2019	05/30/2019 04:12:42 PM	41.767505	-87.686989	(41.767504682, -87.686989416)
254439	24556	JC276141	05/24/2019 02:12:00 AM	013XX W 61ST ST	0110	HOMICIDE	FIRST DEGREE MURDER	GARAGE	False	False	...	16.0	67.0	01A	1168477.0	1864312.0	2019	05/31/2019 04:09:56 PM	41.783199	-87.657851	(41.783199083, -87.657851361)
254440	24557	JC278422	05/25/2019 07:01:00 PM	001XX W 109TH PL	0110	HOMICIDE	FIRST DEGREE MURDER	GANGWAY	False	False	...	34.0	49.0	01A	1177313.0	1832344.0	2019	06/01/2019 04:04:59 PM	41.695280	-87.626419	(41.695279885, -87.626418591)
254441	24558	JC278685	05/25/2019 11:43:00 PM	004XX W 77TH ST	0110	HOMICIDE	FIRST DEGREE MURDER	STREET	False	False	...	17.0	69.0	01A	1174853.0	1853877.0	2019	06/01/2019 04:04:59 PM	41.754424	-87.634786	(41.754424418, -87.634785654)
254442 rows × 22 columns

#drop columns we will not be using
#display data

chi_crime_df=df_chi_crime.drop(columns = ['ID','Case Number','Block','IUCR',
                             'Domestic','Beat','District','Ward','Community Area','FBI Code','X Coordinate',
                            'Y Coordinate','Updated On','Location','Year'])
chi_crime_df.sort_values(by='Date',ascending=False)
Date	Primary Type	Description	Location Description	Arrest	Latitude	Longitude
211	12/27/2019 12:59:00 PM	NARCOTICS	POSS: CANNABIS 30GMS OR LESS	ALLEY	True	41.680518	-87.620591
654	12/27/2019 12:59:00 PM	BATTERY	DOMESTIC BATTERY SIMPLE	RESIDENTIAL YARD (FRONT/BACK)	False	41.780421	-87.675146
263	12/27/2019 12:55:00 PM	CRIMINAL DAMAGE	TO VEHICLE	STREET	False	41.858373	-87.671117
650	12/27/2019 12:55:00 AM	INTERFERENCE WITH PUBLIC OFFICER	RESIST/OBSTRUCT/DISARM OFFICER	STREET	True	41.751392	-87.557734
196	12/27/2019 12:51:00 AM	BATTERY	SIMPLE	BAR OR TAVERN	False	41.945911	-87.655475
...	...	...	...	...	...	...	...
165159	01/01/2019 01:00:00 AM	CRIMINAL DAMAGE	TO PROPERTY	APARTMENT	False	41.879337	-87.758302
165088	01/01/2019 01:00:00 AM	THEFT	FROM BUILDING	HOTEL/MOTEL	False	41.893652	-87.622726
165552	01/01/2019 01:00:00 AM	THEFT	OVER $500	APARTMENT	False	41.949460	-87.651974
165835	01/01/2019 01:00:00 AM	THEFT	FROM BUILDING	BAR OR TAVERN	False	41.922751	-87.644994
165570	01/01/2019 01:00:00 AM	BATTERY	AGGRAVATED: OTHER DANG WEAPON	SIDEWALK	True	41.898003	-87.628771
254442 rows × 7 columns

#time of occurance
time_series=chi_crime_df["Date"]
time_df=time_series.to_frame(name='time')
test=pd.to_datetime(time_series.head())
print(test)
0   2019-12-27 07:13:00
1   2019-12-27 16:22:00
2   2019-12-27 21:25:00
3   2019-12-27 03:00:00
4   2019-12-27 22:50:00
Name: Date, dtype: datetime64[ns]
date_series=chi_crime_df["Date"]
datetime=pd.to_datetime(time_series.head())
datetime_df=datetime.to_frame(name="Date")
datetime_df_date = pd.to_datetime(datetime_df['Date']).dt.date
datetime_df_date
# datetime_df_time = pd.to_datetime(datetime_df['Date']).dt.time
# datetime_df_time
0    2019-12-27
1    2019-12-27
2    2019-12-27
3    2019-12-27
4    2019-12-27
Name: Date, dtype: object
#minor munging
date_series=chi_crime_df["Date"]
datetime=pd.to_datetime(time_series.head())
datetime_df=datetime.to_frame(name="Date")
datetime_df_date = pd.to_datetime(datetime_df['Date']).dt.date
datetime_df_date
datetime_df_time = pd.to_datetime(datetime_df['Date']).dt.time
datetime_df_time
0    07:13:00
1    16:22:00
2    21:25:00
3    03:00:00
4    22:50:00
Name: Date, dtype: object
datetime_df_date
0    2019-12-27
1    2019-12-27
2    2019-12-27
3    2019-12-27
4    2019-12-27
Name: Date, dtype: object
# time_series=chi_crime_df["Date"]
# datetime=pd.to_datetime(time_series.head())
# datetime_df=datetime.to_frame(name="Time")
# datetime_df_date = pd.to_datetime(datetime_df['Date']).dt.date
# datetime_df_date
# datetime_df_time = pd.to_datetime(datetime_df['Time']).dt.time
# datetime_df_time
pd.concat([datetime_df_time, datetime_df_date], axis = 1)
df_concat = pd.concat([datetime_df_time, datetime_df_date], axis = 1)
print(df_concat)
       Date        Date
0  07:13:00  2019-12-27
1  16:22:00  2019-12-27
2  21:25:00  2019-12-27
3  03:00:00  2019-12-27
4  22:50:00  2019-12-27
#format time series
time_series=chi_crime_df["Date"]
datetime=pd.to_datetime(time_series.head())
datetime_df=datetime.to_frame(name="time")
datetime_df_date = pd.to_datetime(datetime_df['time']).dt.date
datetime_df_time = pd.to_datetime(datetime_df['time']).dt.time
datetime_df_formatted=pd.concat([datetime_df_date, datetime_df_time], axis=1)
cols = []
count = 1
for column in datetime_df_formatted.columns:
    if column == 'time':
        cols.append(f'time_{count}')
        count+=1
        continue
    cols.append(column)
datetime_df_formatted.columns = cols
datetime_df_formatted
datetime_df_formatted = datetime_df_formatted.rename(columns={'time_1': 'Date', 'time_2': 'Time'})
datetime_df_formatted
Date	Time
0	2019-12-27	07:13:00
1	2019-12-27	16:22:00
2	2019-12-27	21:25:00
3	2019-12-27	03:00:00
4	2019-12-27	22:50:00
datetime_df_formatted.shape
(5, 2)
#hours of occurance
time_series=chi_crime_df["Date"]
datetime=pd.to_datetime(time_series)
datetime_df=datetime.to_frame(name="Time")
datetime_df_date = pd.to_datetime(chi_crime_df['Date']).dt.hour
hour_df=datetime_df_date.to_frame(name="hour")
hour_df.columns
bins = [0, 6, 12, 18, 24]
labels= ['Night','Morning','Afternoon','Evening']
hour_df["hour"] = pd.cut(hour_df["hour"], bins, labels=labels)
hour_df["hour"].value_counts()
Afternoon    82419
Morning      64436
Evening      62903
Night        32606
Name: hour, dtype: int64
#which crimes occur most frequently in our city?

chi_crime_df["Primary Type"].value_counts()
THEFT                                61324
BATTERY                              48888
CRIMINAL DAMAGE                      26368
ASSAULT                              20394
DECEPTIVE PRACTICE                   17278
OTHER OFFENSE                        16355
NARCOTICS                            13723
BURGLARY                              9469
MOTOR VEHICLE THEFT                   8871
ROBBERY                               7878
CRIMINAL TRESPASS                     6750
WEAPONS VIOLATION                     6241
OFFENSE INVOLVING CHILDREN            2277
CRIM SEXUAL ASSAULT                   1555
INTERFERENCE WITH PUBLIC OFFICER      1529
PUBLIC PEACE VIOLATION                1511
SEX OFFENSE                           1256
PROSTITUTION                           680
HOMICIDE                               498
ARSON                                  370
LIQUOR LAW VIOLATION                   229
CONCEALED CARRY LICENSE VIOLATION      214
STALKING                               214
KIDNAPPING                             175
INTIMIDATION                           163
GAMBLING                               142
OBSCENITY                               56
HUMAN TRAFFICKING                       12
PUBLIC INDECENCY                        10
OTHER NARCOTIC VIOLATION                 8
NON-CRIMINAL                             4
Name: Primary Type, dtype: int64
chi_crime_df["Arrest"].value_counts()
False    200897
True      53545
Name: Arrest, dtype: int64
chi_crime_df["Date"].value_counts()
01/01/2019 12:00:00 AM    62
01/01/2019 12:01:00 AM    54
05/01/2019 12:00:00 PM    36
01/01/2019 09:00:00 AM    35
05/31/2019 12:00:00 PM    34
                          ..
03/27/2019 06:54:00 AM     1
06/24/2019 10:49:00 PM     1
04/02/2019 12:19:00 PM     1
06/17/2019 12:47:00 PM     1
02/09/2019 04:45:00 AM     1
Name: Date, Length: 125155, dtype: int64
chi_crime_df["Location Description"].value_counts()
STREET                55809
RESIDENCE             41792
APARTMENT             33867
SIDEWALK              19912
OTHER                 10408
                      ...  
STAIRWELL                 1
RAILROAD PROPERTY         1
BASEMENT                  1
CTA SUBWAY STATION        1
CHA GROUNDS               1
Name: Location Description, Length: 126, dtype: int64
# plots
# crime_pie = chi_crime_df.plot.pie(subplots=True, figsize=(6, 3))
# crime_pie.set_ylabel("Occurences")

#df = pd.DataFrame({'mass': [.97, .87 , .3, .07, .3],
                   #'radius': [2439.7, 6051.8, 6378.1, 4, 5]},index=['THEFT', 'BATTERY', 'CRIMINAL DAMAGE','HOMICIDE', 'ASSAULT'])

#define plot
#plot = df.plot.pie(y='mass', figsize=(5, 5))
#print(plot)

top_crimes_chi =['THEFT', 'BATTERY','CRIMINAL DAMAGE', 'HOMICIDE', 'ASSAULT', 'DECEPTIVE PRACTICE', 'OTHER OFFENSE', 'NARCOTICS', 'BURGLARY', 'MOTOR VEHICLE THEFT','ROBBERY', 'CRIMINAL TRESPASS', 'WEAPONS VIOLATION', 'OFFENSE INVOLVING CHILDREN', 'CRIM SEXUAL ASSAULT', 'INTERFERENCE WITH PUBLIC OFFICER', 'PUBLIC PEACE VIOLATION', 'SEX OFFENSE', 'PROSTITUTION', 'ARSON', 'LIQUOR LAW VIOLATION', 'STALKING', 'CONCEALED CARRY LICENSE VIOLATION', 'KIDNAPPING', 'INTIMIDATION', 'GAMBLING', 'OBSCENITY', 'HUMAN TRAFFICKING', 'PUBLIC INDECENCY', 'OTHER NARCOTIC VIOLATION', 'NON-CRIMINAL']

top_df = chi_crime_df[chi_crime_df['Primary Type'].isin(top_crimes_chi)]

top_df = top_df['Primary Type'].value_counts().reset_index()

#top_df.head()

fig = px.pie(top_df, values='Primary Type', names='index', title='Top Crimes in Chicago')

fig.show()
24.1%
19.2%
10.4%
8.02%
6.79%
6.43%
5.39%
3.72%
3.49%
3.1%
2.65%
2.45%
0.895%
0.611%
0.601%
0.594%
0.494%
0.267%
0.196%
0.145%
0.09%
0.0841%
0.0841%
0.0688%
0.0641%
0.0558%
0.022%
0.00472%
0.00393%
0.00314%
0.00157%

<img width="1039" alt="image" src="https://github.com/nbergeland/Big_City_Crime/assets/55772476/0321ba70-1664-47c6-a5e3-6449d6eadc27">

<img width="1122" alt="image" src="https://github.com/nbergeland/Big_City_Crime/assets/55772476/e24aa645-690d-4712-9d8e-14cad3d5b773">

#count of crime descriptions
df_new_york_crime["OFNS_DESC"].value_counts()
PETIT LARCENY                       66010
HARRASSMENT 2                       53967
ASSAULT 3 & RELATED OFFENSES        40352
CRIMINAL MISCHIEF & RELATED OF      35577
GRAND LARCENY                       31565
                                    ...  
HOMICIDE-NEGLIGENT-VEHICLE              3
KIDNAPPING AND RELATED OFFENSES         1
OFFENSES AGAINST MARRIAGE UNCL          1
OTHER TRAFFIC INFRACTION                1
OTHER STATE LAWS (NON PENAL LAW)        1
Name: OFNS_DESC, Length: 63, dtype: int64
#count of crime descriptions
df_new_york_crime["PD_DESC"].value_counts()
HARASSMENT,SUBD 3,4,5             39129
ASSAULT 3                         32599
LARCENY,PETIT FROM STORE-SHOPL    25198
HARASSMENT,SUBD 1,CIVILIAN        14838
AGGRAVATED HARASSMENT 2           14111
                                  ...  
CIGARETTE,NO TAX STAMP,POSSESS        1
PROSTITUTION 3, PROMOTING UNDE        1
ALCOHOLIC BEVERAGES,PUBLIC CON        1
ROBBERY,DOCTOR/DENTIST OFFICE         1
SOLICITATION 4, CRIMINAL              1
Name: PD_DESC, Length: 347, dtype: int64
df_new_york_crime[(df_new_york_crime['PD_DESC'].notnull()) &
                  (df_new_york_crime['PD_DESC'].str.contains('HOMICIDE'))][['PD_DESC']]
PD_DESC
67822	HOMICIDE,NEGLIGENT,UNCLASSIFIE
77750	HOMICIDE, NEGLIGENT, VEHICLE,
85487	HOMICIDE,NEGLIGENT,UNCLASSIFIE
89360	HOMICIDE,NEGLIGENT,UNCLASSIFIE
97477	HOMICIDE,NEGLIGENT,UNCLASSIFIE
175135	HOMICIDE, NEGLIGENT, VEHICLE,
225000	HOMICIDE,NEGLIGENT,UNCLASSIFIE
252981	HOMICIDE, NEGLIGENT, VEHICLE,
 
#mix of crime severity

df_new_york_crime["LAW_CAT_CD"].value_counts()
MISDEMEANOR    186354
FELONY         106232
VIOLATION       54488
Name: LAW_CAT_CD, dtype: int64
# functions to call on race, age, etc.

df_new_york_crime["SUSP_RACE"].value_counts()
BLACK                             100287
UNKNOWN                            65756
WHITE HISPANIC                     46394
WHITE                              26566
BLACK HISPANIC                     14879
ASIAN / PACIFIC ISLANDER           10240
AMERICAN INDIAN/ALASKAN NATIVE       827
Name: SUSP_RACE, dtype: int64
#observing age groups suspected of crime most frequently in this cell
df_new_york_crime["SUSP_AGE_GROUP"].value_counts()
UNKNOWN    100079
25-44       89757
45-64       32830
18-24       29986
<18          9102
65+          3170
2019            7
-1              2
-966            2
-968            2
-973            2
929             2
930             1
947             1
924             1
1019            1
-64             1
-964            1
940             1
-80             1
Name: SUSP_AGE_GROUP, dtype: int64
#observing crime occurence across race groups
df_new_york_crime["VIC_RACE"].value_counts()
UNKNOWN                           103388
BLACK                              88021
WHITE HISPANIC                     59696
WHITE                              54212
ASIAN / PACIFIC ISLANDER           25634
BLACK HISPANIC                     14134
AMERICAN INDIAN/ALASKAN NATIVE      1989
Name: VIC_RACE, dtype: int64
#counting which age group has most crime happen to them
df_new_york_crime["VIC_AGE_GROUP"].value_counts()
25-44      122464
UNKNOWN     96757
45-64       64614
18-24       33290
<18         15060
65+         14856
936             3
-56             2
-970            2
-2              2
-943            2
951             1
-946            1
-39             1
-952            1
937             1
938             1
941             1
927             1
-954            1
-67             1
-961            1
1013            1
956             1
-974            1
-71             1
-942            1
-967            1
-934            1
960             1
-50             1
-69             1
-978            1
Name: VIC_AGE_GROUP, dtype: int64
#likelihood of police interaction last year for NY citizens

347074/8623000
0.04024979705438942
#count instances @ different hours of occurance
time_series=df_new_york_crime["CMPLNT_FR_TM"]
datetime=pd.to_datetime(time_series)
datetime_df=datetime.to_frame(name="time")
datetime_df_time = pd.to_datetime(datetime_df['time']).dt.hour
hour_df=datetime_df_time.to_frame(name="hour")
hour_df["hour"].value_counts()
15    20952
17    20943
16    20759
18    20429
12    20017
14    18837
19    18683
20    18377
13    17087
0     16992
21    16238
22    15178
11    14836
10    14645
23    13723
9     13504
8     12383
1     10799
2      9295
7      7949
3      7846
4      6854
6      5736
5      5012
Name: hour, dtype: int64
#binning hours to find to infer time of day correlation 

bins = [0, 6, 12, 18, 24]
labels= ['Night','Morning','Afternoon','Evening']
hour_df["hour"] = pd.cut(hour_df["hour"], bins, labels=labels)
hour_df["hour"].value_counts()
Afternoon    119007
Morning       83334
Evening       82199
Night         45542
Name: hour, dtype: int64
#month of occurance
time_series=df_new_york_crime["RPT_DT"]
datetime=pd.to_datetime(time_series)
datetime_df=datetime.to_frame(name="date")
datetime_df_date = pd.to_datetime(datetime_df['date']).dt.month
date_df=datetime_df_date.to_frame(name="month")
date_df["month"].value_counts()
7    42746
8    41216
9    40511
5    40478
6    40048
4    36907
3    36561
1    36389
2    32218
Name: month, dtype: int64
# seasonal bins
time_series=df_new_york_crime["RPT_DT"]
datetime=pd.to_datetime(time_series)
datetime_df=datetime.to_frame(name="Time")
datetime_df_date = pd.to_datetime(chi_crime_df['Date']).dt.month
season_df=datetime_df_date.to_frame(name="season")
season_df.columns
bins = [0, 3, 6, 9, 12]
labels= ['Winter','Spring','Summer','Fall']
season_df["season"] = pd.cut(season_df["season"], bins, labels=labels)
season_df["season"].value_counts()
Summer    70763
Spring    67641
Winter    58097
Fall      57941
Name: season, dtype: int64
 
#heat map
ny_homicide_df = df_new_york_crime[(df_new_york_crime["PD_DESC"].str.contains('HOMICIDE'))]

location_df = ny_homicide_df[(ny_homicide_df['Latitude'].notnull()) & 
                             (ny_homicide_df['Longitude'].notnull())]
location_df
locations=location_df[["Latitude", "Longitude"]]
figure_layout = {
    'width': '400px',
    'height': '300px',
    'border': '1px solid black',
    'padding': '1px',
    'margin': '0 auto 0 auto'
}
map_ny = gmaps.figure(layout=figure_layout)
# Assign the marker layer to a variable
markers = gmaps.marker_layer(locations)
map_ny.add_layer(markers)
map_ny
#plotly NY

top_crimes_ny =['PETIT LARCENY', 'HARRASSMENT 2', 'ASSAULT 3 & RELATED OFFENSES', 'HOMICIDE', 'GRAND LARCENY', 'AGGRAVATED HARASSMENT 2',]

top_df = df_new_york_crime[df_new_york_crime['OFNS_DESC'].isin(top_crimes_ny)]

top_df = top_df['OFNS_DESC'].value_counts().reset_index()

#top_df.head()

fig = px.pie(top_df, values='OFNS_DESC', names='index', title='Top Crimes in NYC')

fig.show()
34.4%
28.1%
21%
16.4%

<img width="1122" alt="image" src="https://github.com/nbergeland/Big_City_Crime/assets/55772476/4c6ebce3-5c78-40ab-8489-4a9a7387d22e">



