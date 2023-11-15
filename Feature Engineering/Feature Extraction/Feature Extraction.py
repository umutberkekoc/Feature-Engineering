################################
# Feature Extraction
import pandas as pd
from datetime import date
from statsmodels.stats.proportion import proportions_ztest
pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
print(df.head())

#################################
# Binary Features
# creating a new variable that used by Cabin column. Return 1 for not nan values and return 0 for rest

df["Cabin_bool"] = df["Cabin"].notnull().astype("int")
print(df.groupby("Cabin_bool").agg({"Survived": ["mean", "count"]}))

# two sample proportion test (we check are there any stat. sig. diff. between the two groups of cabin)

nobs_0 = df[df["Cabin_bool"] == 0].shape[0]
nobs_1 = df[df["Cabin_bool"] == 1].shape[0]

count_0 = df[df["Cabin_bool"] == 0]["Survived"].sum()
count_1 = df.loc[df["Cabin_bool"] == 1]["Survived"].sum()

test_stat, p_value = proportions_ztest(count=[count_0, count_1],
                                       nobs=[nobs_0, nobs_1])

print("Test Stat: {}\tp-value: {}".format(test_stat, p_value))
if p_value < 0.05:
    print("p-value: {} < 0.05, So HO Is Rejected!".format(p_value))
else:
    print("p-value: {} >= 0.05, So HO Is Not Rejected!".format(p_value))

df.groupby("SibSp").agg({"Survived": ["mean", "count", "sum"]})
df.groupby("Parch").agg({"Survived": ["mean", "count", "sum"]})
print(df["SibSp"].value_counts())
print(df["Parch"].value_counts())

# Important Way of Creating New Variable!
df.loc[(df["SibSp"] + df["Parch"] > 0), "IS_ALONE"] = "NO"
df.loc[(df["SibSp"] + df["Parch"] == 0), "IS_ALONE"] = "YES"
# 2.way!
df["IS_ALONE"] = np.where((df["SibSp"] + df["Parch"] > 0), "NO", "YES")
print(df.head())

count_no = df[df["IS_ALONE"] == "NO"]["Survived"].sum()
count_yes = df[df["IS_ALONE"] == "YES"]["Survived"].sum()
nobs_no = df[df["IS_ALONE"] == "NO"].shape[0]
nobs_yes = df[df["IS_ALONE"] == "YES"]["Survived"].shape[0]

test_stat, p_value = proportions_ztest(count=[count_no, count_yes],
                                       nobs=[nobs_no, nobs_yes])

print("Test Stat: {}\tp-value: {}".format(test_stat, p_value))
if p_value < 0.05:
    print("p-value: {} < 0.05, So HO Is Rejected!".format(p_value))
else:
    print("p-value: {} >= 0.05, So HO Is Not Rejected!".format(p_value))

#################################
# Text Features:
df.head()

# Letter Count
df["Letter_Count"] = df["Name"].str.len()  # Includes . , / ""

# Word Count
df["Word_Count"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

# Specific Word Count (Dr for example)
df["Dr_Count"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.head()
print(df.groupby("Dr_Count").agg({"Survived": ["mean", "count"]}))

#################################
# Date Features
df = pd.read_csv("course_reviews.csv")
print(df.head())
print(df.info())

# -- set variable type as datetime
df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")  # 1.way
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d")  # 2.way

# -- Year - Month - Day:
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day

# -- Today  Variable Year Diff:
df["Year_Diff"] = date.today().year - df["Timestamp"].dt.year

# -- Today  Variable month Diff:
df["Month_Diff"] = (date.today().year - df["Timestamp"].dt.year) * 12 + (date.today().month - df["Timestamp"].dt.month)

# Day Name
df["Day_Name"] = df["Timestamp"].dt.day_name()

#################################
# Feature Interactions:
df = load()
print(df.head())

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & (df["Age"] <= 50)), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & (df["Age"] <= 50)), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"
print(df.head())
print(df.groupby("NEW_SEX_CAT").agg({"Survived": ["mean", "count"]}))