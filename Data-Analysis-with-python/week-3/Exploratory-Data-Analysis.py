import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()
print(df.dtypes)
print("----- correation values-----")
print(df.corr())
print("------- correlation between  bore, stroke,compression-ratio , and horsepower ------")
print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr() )

print("--- Positive linear relationship : scatterplot of 'engine-size' and 'price'---------")
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
#plt.show()
print("----- correlation between engine-size and price ----")
print(df[["engine-size", "price"]].corr())

print("--- Negative linear relationship : scatterplot of 'highway-mpg' and 'price' ------- ")
sns.regplot(x="highway-mpg", y="price", data=df)
#plt.show()
print("----- correlation between highway-mpg and price ----")
print(df[["highway-mpg", "price"]].corr())

print("--- Weak linear relationship : scatterplot of 'highway-mpg' and 'price' ------- ")
sns.regplot(x="peak-rpm", y="price", data=df)
#plt.show()
print("----- correlation between peak-rpm and price ----")
print(df[['peak-rpm','price']].corr())

print("---- correlation between x='stroke', y='price' ----- ")
print(df[["stroke","price"]].corr())
sns.regplot(x="stroke", y="price", data=df)
#plt.show()

print ("--- box plot for body-style and price ---")
sns.boxplot(x="body-style", y="price", data=df)
#plt.show()

print ("--- box plot for engine-location and price ---")
sns.boxplot(x="engine-location", y="price", data=df)
#plt.show()

print ("--- box plot for drive-wheels and price ---")
sns.boxplot(x="drive-wheels", y="price", data=df)
#plt.show()

print ("-------- *** Descriptive Statistical Analysis *** --------")

df.describe()
df.describe(include=['object'])
print(df['drive-wheels'].value_counts().to_frame())

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
print(drive_wheels_counts)

drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

#Basics of grouping
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)

# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
print(grouped_pivot)

# grouping results
df_gptest1 = df[['body-style','price']]
grouped_test2 = df_gptest1.groupby(['body-style'],as_index=False).mean()
print(grouped_test2)
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value )

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
print(grouped_test2.head(2))

print(df_gptest)

print(grouped_test2.get_group('4wd')['price'])

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)


