import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

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

# Label Encoding / Binary Encoding:
df = load()
print(df.head())

label_encoder = LabelEncoder()
label_encoder.fit_transform(df["Sex"])
print("0, 1 --> ", label_encoder.inverse_transform([0, 1]))

def label_encoding(dataframe, bin_var):
    label_encoder = LabelEncoder()
    dataframe[bin_var] = label_encoder.fit_transform(dataframe[bin_var])
    print("0, 1 --> ", label_encoder.inverse_transform([0, 1]))
    return dataframe[bin_var]

bin_var = [i for i in df.columns if df[i].dtypes not in ["int64", "float64"] and df[i].nunique() == 2]
print("Binary Variables:", bin_var)

for i in bin_var:
    label_encoding(df, i)

print(df.head())


df = load_application_train()

bin_var = [i for i in df.columns if df[i] not in ["int64", "float64"] and df[i].nunique() == 2]
print("Binary Variables:", bin_var)

for i in bin_var:
    label_encoding(df, i)
print(df.head())

# ONE-HOT Encoding:
df = load()
print(df.head())
print(df["Embarked"].value_counts())
pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], dtype="int").head()
pd.get_dummies(data=df, columns=["Embarked"], drop_first=True, dtype="int64").head()
pd.get_dummies(data=df, columns=["Embarked"], dummy_na=True, dtype="int64").head()
pd.get_dummies(data=df, columns=["Sex"], drop_first=True, dtype="int").head()  # same with binary encoding
pd.get_dummies(data=df, columns=["Sex", "Embarked"], drop_first=True, dtype="int").head()


def one_hot_encoding(dataframe, variable, df=True, dummy_na=False):
    dataframe = pd.get_dummies(data=dataframe, columns=variable, drop_first=df, dummy_na=dummy_na, dtype="int64")
    return dataframe

ohe_variable = [i for i in df.columns if 10 >= df[i].nunique() > 2]
print("Variables:", ohe_variable)

one_hot_encoding(df, ohe_variable).head()


# Rare Encoding: (Bonus İçerik)

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
# 3. Rare encoder yazmak

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi

df = load_application_train()
print(df["NAME_EDUCATION_TYPE"].value_counts())

print(df["NAME_EDUCATION_TYPE"].value_counts(), "\n", df["NAME_EDUCATION_TYPE"].value_counts() * 100 / len(df))

def grab_variable(dataframe, num_th=10, car_th=10):
    # Categoric Variables:
    cat_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["bool", "object", "category"]]
    num_but_cat = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"] and
                   dataframe[i].nunique() < num_th]
    cat_but_car = [i for i in dataframe.columns if dataframe[i].dtypes in ["object", "category"] and
                   dataframe[i].nunique() > car_th]

    cat_var = cat_var + num_but_cat
    cat_var = [i for i in cat_var if i not in cat_but_car]

    # Numeric Variables:
    num_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"] and
               dataframe[i].nunique() > num_th]

    print("Observations: {}".format(dataframe.shape[0]))
    print("Variables:", dataframe.shape[1])
    print("Categoric Variables:", len(cat_var))
    # print("Numeric But Categoric Variables:", len(num_but_cat))
    print("Categoric But Cardinal Variables:", len(cat_but_car))
    print("Numeric Variables:", len(num_var))
    return cat_var, num_var, cat_but_car


grab_variable(df)
cat_var, num_var, cat_but_car = grab_variable(df)
print("Categoric Variables:", cat_var)


def cat_summary(dataframe, categoric_variables, plot=False):
    print(pd.DataFrame({categoric_variables: dataframe[categoric_variables].value_counts(),
                        "Ratio": 100 * dataframe[categoric_variables].value_counts() / len(dataframe)}))
    print("***********************************************************")
    if plot:
        sns.countplot(data=dataframe, x=dataframe[categoric_variables])
        plt.grid()
        print(plt.show())

for i in cat_var:
    cat_summary(df, i)

# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
print(df["NAME_INCOME_TYPE"].value_counts())
print(df.groupby("NAME_INCOME_TYPE").agg({"TARGET": "mean"}))
print(len(df["NAME_INCOME_TYPE"].value_counts()))

def rare_analyser(dataframe, categoric_var, target):
    print(i, ":", len(dataframe[categoric_var].value_counts()))
    print(pd.DataFrame({"Count": dataframe[categoric_var].value_counts(),
                        "Ratio": 100 * dataframe[categoric_var].value_counts() / len(dataframe),
                        "TARGET_MEAN": dataframe.groupby(categoric_var)[target].mean()}))

for i in cat_var:
    rare_analyser(df, i, "TARGET")


# 3. Rare encoder yazmak:
def rare_encoder(dataframe, percentage):
    df2 = dataframe.copy()

    rare_columns = [i for i in df2.columns if df2[i].dtypes in ["object", "category"] and
                    (df2[i].value_counts() / len(df2) < percentage).any(axis=None)]

    for i in rare_columns:
        x = df2[i].value_counts() / len(df2)
        rare_labels = x[x < percentage].index
        df2[i] = np.where(df2[i].isin(rare_labels), "Rare", df2[i])
    print("Rare Columns:\n", rare_columns)
    return df2

new_df = rare_encoder(df, 0.01)
print(new_df.head())
rare_analyser(new_df, "TARGET", cat_var)


rare_columns = [i for i in df.columns if df[i].dtypes in ["category", "object"] and
                ((df[i].value_counts / len(df)) < 0.01).any(axis=None)]


# Feature Scaling (Özellik Ölçeklendirme):

# Değişkenleri standartlaştırma (normalleştirme) işlemi için,
# Gradient descent kullanan algoritmaların train sürelerini / eğitim sürelerini kısaltmak için ve,
# Uzaklık temelli yöntemlerde yanlılığın önüne geçmek için feature scaling yapılır.

df = load()
print(df.head())

# 1- ::: Standard Scaling :::
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
print(df.head())

# 2- ::: Robust Scaling :::
rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])

# 3- ::: MinMax Scaling :::
mms = MinMaxScaler()
df["Age_minmax_scaler"] = mms.fit_transform(df[["Age"]])
print(df.head())
print(df.describe().T)


age_columns = [i for i in df.columns if "Age" in i]
print("Age Columns:", age_columns)

def num_summary(dataframe,cols, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[cols].describe(quantiles).T)
    if plot:
        sns.histplot(data=dataframe, x=cols, bins=20)
        plt.xlabel(cols)
        plt.title(cols)
        plt.grid()
        print(plt.show(block=True))

for i in age_columns:
    num_summary(df, i, plot=True)

