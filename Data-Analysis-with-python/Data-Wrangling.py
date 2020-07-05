import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
print("Python list headers containing name of headers")
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("Use the Pandas method read_csv() to load the data from the web address.")
df = pd.read_csv(filename, names = headers)
print("To see what the data set looks like, we'll use the head() method.")
print("----- head data with ? values ----")
print(df.head())
# replace "?" to NaN
print("----- replaced data ----")
df.replace("?", np.nan, inplace = True)
print(df.head(5))
print("----- missing data ----")
missing_data = df.isnull()
print(missing_data.head(5))
print("----- count missing data----")
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
print("--- calculate avg of column ------")
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
print("-----Replace NaN by mean value in normalized-losses column-----")
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
print("------Calculate the mean value for 'bore' column---------")
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
print("------Replace NaN by mean value--------")
df["bore"].replace(np.nan, avg_bore, inplace=True)
print("------replace NaN in stroke column by mean.----------")
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_bore, inplace=True)
print("---------Calculate the mean value for the 'horsepower' column:-------")
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
print("------Calculate the mean value for 'peak-rpm' column:-----")
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace=True)
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)
# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)
print(df.head())
print(df.dtypes)
print("--- Convert data types to proper format ------")
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print(df.dtypes)
print("----- *** Data Standardization *** ------")
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
# check your transformed data
print(df.head())
df['highway-L/100km'] = 235/df["highway-mpg"]
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
print(df.head())
print("------ **** Data Normalization **** ------")
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
print(df['length'])
print(df['width'])
df['height'] = df['height']/df['height'].max()
df[["length","width","height"]].head()
print("----- Data Binning -----")
df["horsepower"]=df["horsepower"].astype(float, copy=True)
plt.hist(df["horsepower"])

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))
print(df["horsepower-binned"].value_counts())
plt.bar(group_names, df["horsepower-binned"].value_counts())
# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()
a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()

df.columns
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
print(dummy_variable_1.head())
# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
print(df.head())

dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
print(dummy_variable_2.head())

#merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

df.to_csv('clean_df.csv')
