import pandas as pd
import numpy as np
import seaborn as sns
from fairlearn.metrics import MetricFrame, false_positive_rate, true_positive_rate, selection_rate, count

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier

# link to dataset: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

pd.set_option('display.max_columns', 500)

# load data
df = pd.read_excel("data/default of credit card clients.xls", header = 1)
print(f"Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}")

# helper function
def fit_predict_score(X, y, estimator):
    # pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('estimator', estimator)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # Fit models
    pipe.fit(X_train, y_train)
    
    # Generate predictions
    expected = y_test
    predicted = pipe.predict(X_test)
    return accuracy_score(y_true = expected, y_pred = predicted), expected, predicted, X_test


#---------------------------------------
# FEATURE ENGINEERING
#---------------------------------------

# data munging
df["male"] = np.where(df.SEX == 1, 1, 0)
df["under30"] = np.where(df.AGE < 30, 1, 0)
df["unmarried"] = np.where(df.MARRIAGE ==2, 1, 0)
df.rename(columns = {'default payment next month':'y'}, inplace = True)

educ_dict = {"EDUCATION": ["1", "2", "3", "4"], "EDUC_LEVEL": ["graduate_school", "university", "high_school", "other"]}
df["EDUCATION"] = df.EDUCATION.astype(str)
df = pd.merge(df, pd.DataFrame(educ_dict), on = "EDUCATION")

for col in ['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
    df.loc[df[col] < 0, col] = 0

# change subset of int rows to numeric so they don't get one-hot encoded :/
for col in ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
       'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
       'PAY_AMT6']:
    df[col] = df[col].astype(float)

df = pd.get_dummies(df, prefix = "educ_", columns = ["EDUC_LEVEL"], drop_first = True)


#---------------------------------------
# EXPLORATION
#---------------------------------------

# View pairwise correlations
sns.heatmap(df)

# View missings
[(col, df[col].isnull().mean()) for col in df.columns.tolist()] # we good!

# plot the distribution of the sensitive attributes and the target attribute
for col in ["SEX", "under30", "unmarried"]:
    sns.catplot(x = col, y = "y",  kind="bar", data = df)


#---------------------------------------
# PREP
#---------------------------------------

X = df.drop(columns = ["SEX", "AGE", "MARRIAGE", "y", "ID", "EDUCATION", "EDUC_LEVEL"], axis = 1)
y = df.y

#---------------------------------------
# MODELLING
#---------------------------------------

# define the models we want to test
models = [
    SVC(gamma='auto'), LinearSVC(),
    SGDClassifier(max_iter=100, tol=1e-3), KNeighborsClassifier(),
    LogisticRegression(solver='lbfgs'), LogisticRegressionCV(cv=3),
    BaggingClassifier(), ExtraTreesClassifier(n_estimators=300),
    RandomForestClassifier(n_estimators=300)
]

results = []

# run models
for estimator in models:
    
    score, y_true, y_pred, testdata = fit_predict_score(X, y, estimator)

    # Compute and return F1 (harmonic mean of precision and recall)
    results.append((estimator.__class__.__name__, score) )

results_sorted = sorted(results, key = lambda s: s[1], reverse= True)

# export file
with open("data/modeling_results.txt", "w") as out:
    for item in results_sorted:
        out.writelines(f"{item[0]} : {item[1]} ")

#---------------------------------------
# ASSESS BIAS
#---------------------------------------

# Load in the results
results = open("data/modeling_results.txt", "r")


# let's take a closer look at the random forest model (so we can get permutation importance later on)
score, y_true, y_pred, testdata = fit_predict_score(X, y, RandomForestClassifier(n_estimators=300))

metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'false positive rate': false_positive_rate,
    'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': count}

# assess demographic parity for women and those under 30 (are these groups more likely to receive a prediction of default?)
gender_frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=testdata["male"])
print(gender_frame.by_group)

gender_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

age_frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=testdata["under30"])
print(age_frame.by_group)

age_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

# Looks like our model is more likely to predict males default than women, so no bias there. 
# on the other hand, under 30 is erroneously predicted as likely to default (see the false positive rate).
# we also have way less observations for this group. what can we do to mitigate this outcome?

#---------------------------------------
# MITIGATE BIAS
#---------------------------------------

# Okay, so what can we do to improve outcomes for the under-30 cohort?

df.under30.mean()

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Redo the pipeline with the SMOTE parameter added in and reassess the results

pipe = make_pipeline(SMOTE(random_state=0), StandardScaler(with_mean=False), RandomForestClassifier(n_estimators=300) )

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
pipe.fit(X_train, y_train)

y_true = y_test
y_pred = pipe.predict(X_test)

age_frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=testdata["under30"])

age_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)
