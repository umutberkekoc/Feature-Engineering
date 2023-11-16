import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as miss
import datetime as dt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
print(df.head())

# Feature Engineering - Pre-processing
# 1- Outliers
# 2- Missing Values
# 3- Encoding Scaling - Feature Scaling
# 4- Feature Extraction

##################################
# OUTLIERS:

# Finding Outliers Thresholds
# Check Outliers
# Show the Outlier Data
# Suppressed the Outliers
# Drop Outliers
# Grap Variables by Types


def outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    # !In literature 0.25 and 0.75 but we generally prefer to use 0.05 - 0.95 or 0.01 - 0.99
    quantile1 = dataframe[variable].quantile(q1)
    quantile3 = dataframe[variable].quantile(q3)
    iqr = quantile3 - quantile1
    lower_limit = quantile1 - 1.5 * iqr
    upper_limit = quantile3 + 1.5 * iqr
    return lower_limit, upper_limit

def check_outlier(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
    #if dataframe[(datarame[variable] < lower_limit) | (dataframe[variable] > upper_limit)].any(axis=None) == True:
        return True
    else:
        return False

def show_outliers(dataframe, variable, index=False, row=5):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    print(dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].head(row))
    if index:
        print(dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].index)

def change_with_threshold(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    print("Outliers Suppressed!")

def remove_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    df_without_outliers = dataframe[~(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)]
    return df_without_outliers

def grab_variables(dataframe, num_th=10, car_th=20, name=False):
    # Categoric Variables #
    cat_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["object", "category", "bool"]]
    num_but_cat = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"] and
                   dataframe[i].nunique() < num_th]
    cat_but_car = [i for i in dataframe.columns if dataframe[i].dtypes in ["object", "category"] and
                   dataframe[i].nunique() > car_th]
    cat_var = cat_var + num_but_cat
    cat_var = [i for i in cat_var if i not in cat_but_car]

    # Numeric Variables #
    num_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"] and
               dataframe[i].nunique() > num_th]

    if name:
        print("Catgoric Variables:", cat_var)
        print("Numeric Variables:", num_var)
        print("Cat But Cardinal Variables:", cat_but_car)
        print("Num But Cat Variables:", num_but_cat)

    print("Observations:", dataframe.shape[0])
    print("Total Variables:", dataframe.shape[1])
    #print("Total Variables:", len(cat_var + num_var + cat_but_car))
    print("Catgoric Variables:", len(cat_var))
    print("Numeric Variables:", len(num_var))
    print("Cat But Cardinal Variables:", len(cat_but_car))
    print("Num But Cat Variables:", len(num_but_cat))
    return cat_var, num_var, cat_but_car

print(df.head())
print(df.isnull().sum())
print(df.describe().T)
print(df.info())

cat_var, num_var, cat_but_car = grab_variables(df, name=True)
num_var = [i for i in num_var if "Id" not in i]

# Controlling Outliers For Numerics with Boxplot
import random
for i in num_var:
    sns.boxplot(data=df, x=i, color=random.choice(["blue", "green", "red", "pink", "gray", "purple"]))
    plt.grid()
    plt.title("Boxplot")
    plt.xlabel(i)
    print(plt.show())

# Finding Outlier Thresholds for each numeric variable
for i in num_var:
    print(i, ":", outlier_thresholds(df, i))

# Check are there any outlier for each variable
for i in num_var:
    print(i, ":", check_outlier(df, i))

# show the rows and index which includes outliers for each variable
for i in num_var:
    print(i, ":::", show_outliers(df, i, index=True))

# suppressed all outliers for each variable
for i in num_var:
    change_with_threshold(df, i)

# Check are there any outlier for each variable after suppressed
for i in num_var:
    print(i, ":", check_outlier(df, i))

# removing outliers and controlling any outlier for after removing
for i in num_var:
    new_df = remove_outliers(df, i)
    print(i, ":", check_outlier(new_df, i))



###############################
# MISSING VALUES

# Observing missing values
# Removing missing values from the data set
# Filling NaN in Numeric Variables
# Filling NaN in Categoric Variables
# Assigning values in categorical variable breakdown
# Filling with Predictive Assignment Process
# Standardization of variables
# examine the structure of missing data
# Analysis of Missing Values with Dependent Variable

# Veri Setinde Eksik Gözlem Kontrolü:
print(df.isnull().any())
print(df.isnull().sum().any())
print(df.isnull().values.any())

# Değişkenlere Göre Eksik Gözlem Sayısı:
print(df.isnull().sum())

# Değişkenlere Göre Eksik Gözlem Sayısını Sıralama:
print(df.isnull().sum().sort_values(ascending=False))

# Değişkenlere Göre Eksik Olmayan Gözlem Sayısı
print(df.notnull().sum())

# Veti Setindeki Toplam Eksik Değer Sayısı
print(df.isnull().sum().sum())

# En az 1 tane eksik değere sahip olan gözlem birimleri
print(df[df.isnull().any(axis=1)])
print(df[df.isnull().any(axis=1)].shape)

# Tam olan gözlem birimleri:
print(df[df.notnull().all(axis=1)])
print(df[df.notnull().all(axis=1)].shape)


# Değişkenlere göre eksik değer oranları

def missing_values_table(dataframe, na_name=False):
    nan_var = [i for i in dataframe.columns if dataframe[i].isnull().sum() > 0]
    # nan_var = [i for i in df.columns if df[i].isnull().sum().any() == True]  2. way
    number_of_miss = dataframe[nan_var].isnull().sum().sort_values(ascending=False)
    missing_ratio = (dataframe[nan_var].isnull().sum() * 100 / dataframe.shape[0]).sort_values(ascending=False)
    xxx = pd.concat([number_of_miss, missing_ratio], keys=["n_miss", "ratio"], axis=1)
    print(xxx, end="\n")

    if na_name:
        print("Nan Variables:\t", nan_var)

missing_values_table(df, True)


# dropna
#df.dropna(inplace=True)
df = df.drop()

# Filling NAN values for Numeric Variables
cat_var, num_var, cat_but_car = grab_variables(df)
num_var = [i for i in num_var if "Id" not in i]
print("Numeric Vasriables:", num_var)

df["Age"].fillna(df["Age"].mean(), inplace=True)  # filling nan with mean
df["Age"].fillna(df["Age"].median(), inplace=True)  # filling nan with median
df["Age"].fillna(1, inplace=True)  # filling nan with a specific number

df["Age"].fillna(df["Age"].std()).isnull().sum()  # check without permanent filling nan
df.loc[df["Age"].isnull()].head()

# filling all nan values in numeric variables with lambda
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype in ["int64", "float64"] else x, axis=0)
missing_values_table(dff)

# Filling NAN Values for Categoric Variables
print("Categoric Variables:", cat_var)
print(df.loc[df["Embarked"].isnull()])
df["Embarked"].fillna("this is nan", inplace=False)  # filling nan with specific character
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # filling nan with mode

# filling all nan values in categoric varaibles with lambda
dff = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype in ["category", "object"]
                                                  and x.nunique() <= 10 else x, axis=0)
missing_values_table(dff)


# Assigning values in categorical variable breakdown
print(df.loc[df["Age"].isnull()].head())
print(df.groupby("Sex").agg({"Age": "mean"}))
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"), inplace=True)



# tahmine dayalı atama işlemi ile doldurma
cat_var, num_var, cat_but_car = grab_variables(df)
num_var = [i for i in num_var if i not in "PassengerId"]
dff = pd.get_dummies(df[cat_var + num_var], drop_first=True)
dff.head()

# değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]

# eksik verinin yapısını incelemek
df = load()
miss.bar(df)
print(plt.show())
miss.heatmap(df)
print(plt.show())
miss.matrix(df)
print(plt.show())


#######################################
# Encoding Scaling

# Label / Binary Encoding
# ONE-HOT Encoding
# Rare Encoding

# 1- Label / Binary Encoding
df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_variable):
    label_encoder = LabelEncoder()
    dataframe[binary_variable] = label_encoder.fit_transform(dataframe[binary_variable])
    print("0, 1 --> ", label_encoder.inverse_transform([0, 1]))
    return dataframe[binary_variable]

bin_var = [i for i in df.columns if df[i].nunique() == 2
           and df[i].dtypes not in ["int64", "float64"]]
for i in bin_var:
    label_encoder(df, i)
print(df.head())

# 2- ONE-HOT ENCODING
df = load()
pd.get_dummies(data=df, columns=["Sex"]).head()  # return true false
pd.get_dummies(data=df, columns=["Sex"], dtype="int").head()  # return 0, 1
pd.get_dummies(data=df, columns=["Embarked"], dtype="int", drop_first=True).head()
pd.get_dummies(data=df, columns=["Embarked"], dtype="int", dummy_na=True).head()
pd.get_dummies(data=df, columns=["Embarked", "Sex"], dtype="int", drop_first=True).head()

def one_hot_encoding(dataframe, ohe_var, drop_first=True, dummy_na=False):
    dataframe = pd.get_dummies(data=dataframe, columns=ohe_var, dtype="int",
                               drop_first=drop_first, dummy_na=dummy_na)
    return dataframe
ohe_variable = [i for i in df.columns if 10 >= df[i].nunique() > 2]

df = one_hot_encoding(df, ohe_variable)
df.head()

# 3- Rare Encoding
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
df = load()
cat_var, num_var, cat_but_car = grab_variables(df)
print("Categoric Variables", cat_var)

def categoric_variable_summary(dataframe, variable, plot=False):
    print(pd.DataFrame({variable: dataframe[variable].value_counts(),
                        "Ratio": dataframe[variable].value_counts() * 100 / dataframe.shape[0]}))
    print("************************")
    if plot:
        sns.countplot(data=dataframe, x=dataframe[variable].value_counts() * 100 / dataframe.shape[0])
        plt.grid()
        plt.title("Ratio of" + variable)
        plt.xlabel(variable)
        print(plt.show())

for i in cat_var:
    categoric_variable_summary(df, i, plot=True)

def rare_target(dataframe, target, variable, plot=False):
    print(variable, ":", dataframe[variable].nunique())
    print(pd.DataFrame({variable: dataframe[variable].value_counts(),
                        "Ratio": dataframe[variable].value_counts() * 100 / len(dataframe),
                        "TARGET_MEAN": dataframe.groupby(variable)[target].mean()}), end="\n\n\n")
    if plot:
        sns.countplot(data=dataframe, x=dataframe.groupby(variable)[target].mean())
        plt.grid()
        plt.title("TARGET MEAN of" + variable)
        plt.xlabel(variable)
        print(plt.show())

for i in cat_var:
    rare_target(df, "Survived", i, True)

# 3. Rare encoder yazmak
def rare_encoder(dataframe, percentage):
    df2 = dataframe.copy()

    rare_columns = [i for i in df2.columns if df2[i].dtypes in ["object", "category"] and
                    ((df2[i].value_counts() / len(df2)) < percentage).any(axis=None)]

    for i in rare_columns:
        x = df2[i].value_counts() / len(df2)
        rare_labels = x[x < percentage].index
        df2[i] = np.where(df2[i].isin(rare_labels), "Rare", df2[i])

    return df2

new_df = rare_encoder(df, 0.01)
rare_target(new_df, "Survived", cat_var)

# FEATURE SCALING

# 1- Standard Scaler
df = load()
ss = StandardScaler()
df["Age_Standard_Scale"] = ss.fit_transform(df[["Age"]])

# 2- Robust Scaler
rs = RobustScaler()
df["Age_Robust_Scale"] = rs.fit_transform(df[["Age"]])

# 3- MinMax Scaler
mms = MinMaxScaler()
df["Age_MinMax_Scale"] = mms.fit_transform(df[["Age"]])
print(df.head())


def numeric_variable_summary(dataframe, num_var, plot=False):
    quantiles = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[num_var].describe(quantiles).T)
    if plot:
        sns.histplot(data=dataframe, x=num_var, bins=20, color=random.choice(["red", "blue", "green", "yellow", "gray", "pink"]))
        plt.grid()
        plt.xlabel(num_var)
        plt.title(num_var)
        print(plt.show(block=True))

numeric_variable_summary(df, "Age")

age_cols = [i for i in df.columns if "Age" in i]
print("Age Columns:", age_cols)

for i in age_cols:
    numeric_variable_summary(df, i, True)

# Numeric to Categorical
df["Age_qcut"] = pd.qcut(df["Age"], q=5)


################################
# Feature Extraction (Özellik Çıkarımı)

# Text Features
df = load()
df.head()

# Letter Count
df["NAME_LETTER_COUNT"] = df["Name"].str.len()

# Word Count
df["NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(x.split(" ")))

# Specific Word Count
df["Dr_WORD_COUNT"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))


# Date Features
import datetime as dt
dff = pd.read_csv("course_reviews.csv")
print(dff.head())
print(dff.info())

# change dtype:
dff["Timestamp"] = pd.to_datetime(dff["Timestamp"])  # 1. way
dff["Timestamp"] = dff["Timestamp"].astype("datetime64[ns]")

# Year
dff["Year"] = dff["Timestamp"].dt.year
# Month
dff["month"] = dff["Timestamp"].dt.month
# Day
dff["day"] = dff["Timestamp"].dt.day
# Year diff
dff["Year_Diff"] = date.today().year - dff["Timestamp"].dt.year
# Month diff
dff["Month_Diff"] = (date.today().month - dff["Timestamp"].dt.month) + (date.today().year - dff["Timestamp"].dt.year) * 12
# Day Name
dff["Day_Name"] = dff["Timestamp"].dt.day_name()
print(dff.head())

# Binary Features
df = load()
print(df.head())

#df["Caboin_Bool"] = df["Cabin"].isnull().astype("int64")  # set nan as 1
df["Cabin_Bool"] = df["Cabin"].notnull().astype("int64")  #sen nan as 0

print(df.groupby("Cabin_Bool").agg({"Survived": ["count", "mean"]}))

from statsmodels.stats.proportion import proportions_ztest

nobs_0 = df[df["Cabin_Bool"] == 0].shape[0]
nobs_1 = df[df["Cabin_Bool"] == 1].shape[0]
count_0 = df[df["Cabin_Bool"] == 0]["Survived"].sum()
count_1 = df[df["Cabin_Bool"] == 1]["Survived"].sum()

test_stat, p_value = proportions_ztest(count=[count_0, count_1],
                                       nobs=[nobs_0, nobs_1])

print("Test Stat: {}\tP-value: {}".format(test_stat, p_value))
if p_value < 0.05:
    print("HO Rejected. Thus there is statistically significant difference between two sample in their ratios")
else:
    print("HO Not Rejected. Thus there is not statistically significant difference between two sample in their ratios")


# Creating New Variable
df.groupby("SibSp").agg({"Survived": ["mean", "count", "sum"]})
df.groupby("Parch").agg({"Survived": ["mean", "count", "sum"]})

df.loc[df["Parch"] + df["SibSp"] > 0, "IS_ALONE"] = "NO"
df.loc[df["Parch"] + df["SibSp"] == 0, "IS_ALONE"] = "YES"

nobs_0 = df[df["IS_ALONE"] == "NO"].shape[0]
nobs_1 = df[df["IS_ALONE"] == "YES"].shape[0]
count_0 = df[df["IS_ALONE"] == "NO"]["Survived"].sum()
count_1 = df[df["IS_ALONE"] == "YES"]["Survived"].sum()

test_stat, p_value = proportions_ztest(count=[count_0, count_1],
                                       nobs=[nobs_0, nobs_1])

print("Test Stat: {}\tP-value: {}".format(test_stat, p_value))
if p_value < 0.05:
    print("HO Rejected. Thus there is statistically significant difference between two sample in their ratios")
else:
    print("HO Not Rejected. Thus there is not statistically significant difference between two sample in their ratios")

# Feature Interactions (Özellik Etkileşimleri)
df = load()
print(df.head())
df.loc[(df["Age"] <= 21) & (df["Sex"] == "female"), "NEW_SEX_CAT"] = "youngfemale"
df.loc[((df["Age"] > 21) & (df["Age"] <= 50)) & (df["Sex"] == "female"), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["Age"] > 50) & (df["Sex"] == "female"), "NEW_SEX_CAT"] = "seniorfemale"
df.loc[(df["Age"] <= 21) & (df["Sex"] == "male"), "NEW_SEX_CAT"] = "youngmale"
df.loc[((df["Age"] > 21) & (df["Age"] <= 50)) & (df["Sex"] == "male"), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["Age"] > 50) & (df["Sex"] == "male"), "NEW_SEX_CAT"] = "seniormale"

print(df.groupby("NEW_SEX_CAT").agg({"Survived": ["mean", "count"]}).sort_values(("Survived", "mean"), ascending=False))