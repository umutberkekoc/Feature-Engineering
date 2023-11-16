##############################
# Diabete Feature Engineering
##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import random

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 700)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("diabetes.csv")
df.head()

# GÖREV 1: KEŞİFCİ VERİ ANALİZİ

# Adım 1: Genel resmi inceleyiniz.
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Size #####################")
    print(dataframe.size)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]).T)
    print("##################### Descriptive Statistics #####################")
    print(dataframe.describe().T)

check_df(df, head=3)


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_variables(dataframe, num_th=10, car_th=20, name=False):
    # Categoric Variables #
    cat_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["bool", "object", "category"]]
    num_but_cat = [i for i in dataframe.columns if dataframe[i].dtypes in ["float64", "int64"]
                   and dataframe[i].nunique() < num_th]
    cat_but_car = [i for i in dataframe.columns if dataframe[i].dtypes in ["object", "category"]
                   and dataframe[i].nunique() > car_th]
    cat_var = cat_var + num_but_cat
    cat_var = [i for i in cat_var if i not in cat_but_car]

    # Numeric Variables #
    num_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["float64", "int64"]
               and dataframe[i].nunique() > num_th]

    print("Number of Observation:", len(dataframe))
    print("Number of Variable:", len(dataframe.columns))
    print("Number fo Categoric Variables:", len(cat_var))
    print("Number of Numeric Variables:", len(num_var))
    print("Number of Cat But Cardinal Variables:", len(cat_but_car))
    print("Number of Num But Categoric Variables:", len(num_but_cat))
    if name:
        print("Categoric Variables: {}\nNumeric Variables: {}\nCategoric But Cardinal Variables: {}".
              format(cat_var, num_var, cat_but_car))

    return cat_var, num_var, cat_but_car

cat_var, num_var, cat_but_car = grab_variables(df, name=True)



# Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
def categoric_variable_summary(dataframe, variable, plot=False):
    print("************************")
    print(pd.DataFrame({variable: dataframe[variable].value_counts(),
                        "Ratio": dataframe[variable].value_counts() * 100 / len(dataframe)}))
    if plot:
        sns.countplot(data=dataframe, x=variable, color=random.choice(["red", "green", "blue", "pink"]))
        plt.grid()
        plt.title("CountPlot for:" + variable)
        plt.xlabel(variable)
        print(plt.show())

for i in cat_var:
    categoric_variable_summary(df, i, plot=True)

def numeric_variable_summary(dataframe, variable, plot=False):
    print("*********************")
    quantiles = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[variable].describe(quantiles).T)
    if plot:
        sns.histplot(data=dataframe, x=variable, bins=20, color=random.choice(["red", "green", "blue", "pink"]))
        plt.grid()
        plt.title("Histogram for:" + variable)
        plt.xlabel(variable)
        print(plt.show())

for i in num_var:
    numeric_variable_summary(df, i, plot=True)

# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for i in num_var:
    target_summary_with_num(df, "Outcome", i)

def target_summary_with_cat(dataframe, target, cat_cols):
    print(dataframe.groupby(cat_cols).agg({target: "mean"}))

for i in cat_var:
    target_summary_with_cat(df, "Outcome", i)


# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, variable, qu1=0.05, qu3=0.95):
    q1 = dataframe[variable].quantile(qu1)
    q3 = dataframe[variable].quantile(qu3)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return lower_limit, upper_limit


def suppression_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit


def remove_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    df_without_outliers = dataframe[~(dataframe[variable] < lower_limit) |
                                    (dataframe[variable] > upper_limit)]
    return df_without_outliers

def show_outliers(dataframe, variable, head=5):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) |
                 (dataframe[variable] > upper_limit)].isnull().sum().any() == True:
        print(dataframe[(dataframe[variable] < lower_limit) |
                                    (dataframe[variable] > upper_limit)].head(head), end="\n")
    else:
        print("There is no any outlier in this variable!")

def check_outlier(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) |
                 (dataframe[variable] > upper_limit)].isnull().sum().any() == True:
        return True
    else:
        return False

for i in num_var:
    print(i, ":", outlier_thresholds(df, i))

for i in num_var:
    df_new = remove_outliers(df, i)

for i in num_var:
    print(i, ":", check_outlier(df, i))

for i in num_var:
    print(i, ":", show_outliers(df, i))


# Adım 6: Eksik gözlem analizi yapınız.
def missing_values_table(dataframe, na_name=False):
    nan_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    nan_number = dataframe[nan_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[nan_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([nan_number, np.round(ratio, 2)], axis=1, keys=['nan_number', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return nan_columns

na_columns = missing_values_table(df, na_name=True)

# Bir insanda Pregnancies ve Outcome dışındaki değişken değerlerinin 0 olamayacağı bilinmektedir.
# Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir.
for i in df.columns:
    print(i, ":", df[i].min())


# Gözlem birimlerinde 0 olan degiskenlerin belirlenmesi

zero_columns = [i for i in df.columns if (df[i].min() == 0 and
                                              i not in ["Pregnancies", "Outcome"])]
print("Zero Columns:\n", zero_columns)

for i in zero_columns:
    print(i, ":", df[i].mean())

# Gözlem birimlerinde 0 olan degiskenlerin her birine gidip 0 iceren
# gozlem degerlerini NaN ile değiştirdik.

for i in zero_columns:
    df[i] = df[i].replace(0, np.nan)

for i in zero_columns:
    print(i, ":", df[i].mean())

# Eksik Gözlem Analizi
df.isnull().sum()

# Eksik Değerlerin Doldurulması

for i in zero_columns:
    df[i].fillna(df[i].median(), inplace=True)

df.isnull().sum()

# Adım 7: Korelasyon analizi yapınız.

print(df.corr())

# Korelasyon Matrisi

plt.figure(figsize=[12, 12])
sns.heatmap(data=df.corr(), annot=True)
plt.xticks(rotation=45)
plt.title("Correlation Matrix", fontsize=24)
print(plt.show())

# "Outcome" ile ilişkilerine göre "diyabetik" ve "sağlıklı" olarak
# iki kategoride incelemek istersek;
healthy = df[df["Outcome"] == 0]
diabetic = df[df["Outcome"] == 1]
plt.scatter(x=healthy["Age"], y=healthy["Insulin"], color="blue", label="Healthy", alpha=0.4)
plt.scatter(x=diabetic["Age"], y=diabetic["Insulin"], color="red", label="Diabetic", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Insulin")
plt.legend()
print(plt.show())


# Adım 2: Yeni değişkenler oluşturunuz.

##################################
# ÖZELLİK ÇIKARIMI
##################################

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
print(df["Age"].describe().T)
df.loc[(df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"
print(df.head())

# BMI 18,5 aşağısı underweight,
# 18.5 ile 24.9 arası normal,
# 24.9 ile 29.9 arası Overweight ve
# 30 üstü obez
print(df["BMI"].describe().T)
df.loc[(df["BMI"] <= 18.5), "NEW_BMI"] = "Underweight"
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NEW_BMI"] = "Normal"
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NEW_BMI"] = "Overweight"
df.loc[(df["BMI"] > 29.9), "NEW_BMI"] = "Obese"

print(df.head())

# 2. way
df["NEW_BMI_2"] = pd.cut(x=df["BMI"], bins=[0, 18.5, 24.9, 29.9, df["BMI"].max()],
                         labels=["Underweight", "Normal", "Overweight", "Obese"])

df.drop("NEW_BMI_2", axis=1, inplace=True)

# Glukoz degerini kategorik değişkene çevirme
df["Glucose"].head()
df["Glucose"].info()
df["Glucose"].describe().T

df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"],
                           bins=[0, 120, 141, df["Glucose"].max()],
                           labels=["Normal", "Prediabetes", "Diabetes"])
print(df.head())

# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı
# age < 50 >= 50
# bmı 0 - 18,5 -  24.9 - 29.9 - max

df.loc[(df["Age"] < 50) & (df["BMI"] <= 18.5), "NEW_AGE_BMI"] = "UnderweightMature"
df.loc[(df["Age"] < 50) & ((df["BMI"] > 18.5) & (df["BMI"] <= 24.9)), "NEW_AGE_BMI"] = "NormalMature"
df.loc[(df["Age"] < 50) & ((df["BMI"] > 24.9) & (df["BMI"] <= 29.9)), "NEW_AGE_BMI"] = "OverweightMature"
df.loc[(df["Age"] < 50) & (df["BMI"] > 29.9), "NEW_AGE_BMI"] = "ObeseMature"

df.loc[(df["Age"] >= 50) & (df["BMI"] <= 18.5), "NEW_AGE_BMI"] = "UnderweightSenior"
df.loc[(df["Age"] >= 50) & ((df["BMI"] > 18.5) & (df["BMI"] <= 24.9)), "NEW_AGE_BMI"] = "NormalSenior"
df.loc[(df["Age"] >= 50) & ((df["BMI"] > 24.9) & (df["BMI"] <= 29.9)), "NEW_AGE_BMI"] = "OverweightSenior"
df.loc[(df["Age"] >= 50) & (df["BMI"] > 29.9), "NEW_AGE_BMI"] = "ObeseSenior"

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Glucose"] < 70) & (df["Age"] < 50), "NEW_AGE_GLUCOSE"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] < 50), "NEW_AGE_GLUCOSE"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] < 50), "NEW_AGE_GLUCOSE"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & (df["Age"] < 50), "NEW_AGE_GLUCOSE"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE"] = "highsenior"
df.head()

# değişken isimelerini büyüterek okunulurluğunu arttıralım
df.columns = [i.upper() for i in df.columns]

# Adım 3:  Encoding işlemlerini gerçekleştiriniz.
##################################
# ENCODING
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_var, num_var, cat_but_car = grab_variables(df)

# Label / Binary Encoding:
def label_encoder(dataframe, bin_var):
    label_encoder = LabelEncoder()
    dataframe[bin_var] = label_encoder.fit_transform(df[bin_var])
    print("0, 1 --> ", label_encoder.inverse_transform([0, 1]))
    return dataframe[bin_var]

binary_variables = [i for i in df.columns if df[i].dtypes in ["object", "category", "bool"]
                    and df[i].nunique() == 2]
print("Binary Variables:", binary_variables)

for i in binary_variables:
    label_encoder(df, i)
print(df.head())

# ONE-HOT ENCODING:
def one_hot_encoder(dataframe, ohe_var, dummy_na=False, drop_first=True):
    datafarme = pd.get_dummies(data=dataframe, columns=ohe_var, dummy_na=dummy_na, drop_first=drop_first, dtype="int64")
    return datafarme

ohe_var = [i for i in df.columns if df[i].nunique() > 2 and df[i].dtypes in ["category", "object"]]
ohe_var.remove("NEW_BMI_2")
df.info()
print( "One Hot Encoder Variables:\n", ohe_var)

df = one_hot_encoder(df, ohe_var)
print(df.head())

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

##################################
# STANDARTLAŞTIRMA
##################################

print("Numeric Varaibles:", num_var)
scaler = StandardScaler()
df[num_var] = scaler.fit_transform(df[num_var])

mms = MinMaxScaler()
df[num_var] = mms.fit_transform(df[num_var])

rs = RobustScaler()
df[num_var] = rs.fit_transform(df[num_var])
print(df.head())

# Adım 5: Model Oluşturma
##################################
# MODELLEME
##################################
y = df["OUTCOME"]
x = df.drop([i for i in df.columns if "NEW" in i or i == "OUTCOME"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 3)}")
print(f"F1: {round(f1_score(y_pred,y_test), 3)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 3)}")

##################################
# FEATURE IMPORTANCE
##################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)