import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

import joblib

path_data = r'C:\Users\biaka\OneDrive\Documents\JupyterNotebooks\ExpensesClassification\data'
path_results = r'C:\Users\biaka\OneDrive\Documents\JupyterNotebooks\ExpensesClassification\results'
file_name = r'Gastos_Australia_PythonProject.xlsx'

df = pd.read_excel(os.path.join(path_data, file_name))

# Cleaning data: select expenses only; separate data points to be predicted
df_expenses = df[df['E/S'] == 'S'] 
df_expenses.loc[60, 'Cidade'] = 'Jabiru'
df_jabiru = df_expenses[df_expenses['Cidade'] == 'Jabiru']
df_expenses.drop(df_expenses[df_expenses['Cidade'] == 'Jabiru'].index, inplace=True)

# City analysis: keep only cities where stay was longer than 3 months
time_in_city = df_expenses.groupby(by='Cidade', dropna=True)['Data'].agg(['min', 'max','count','nunique'])
time_in_city['total_time'] = time_in_city['max'] - time_in_city['min'] + pd.Timedelta(days=1)
time_in_city['avg_transaction_per_day'] = time_in_city['count'] / time_in_city['total_time'].dt.days
cities_90days = time_in_city[time_in_city['nunique']>=90].index.tolist()
df_expenses = df_expenses[df_expenses.Cidade.isin(cities_90days)]

# Cleaning data: keep relevant variables
filter = df_expenses[(df_expenses['Variável'].isna()) | (df_expenses['Variável'].isin(['-', 'Viagens']))].index
df_expenses.drop(filter, inplace = True)

df_expenses.reset_index(drop=True, inplace=True)

# Split data into training and testing sets
train_set, test_set = train_test_split(df_expenses, test_size=0.3, random_state=42)

# Adding features to training set: day of the week, weekday or weekend
X_full = train_set.copy()
X_full['day_of_week'] = X_full['Data'].dt.dayofweek
X_full['weekday'] = np.where(X_full['day_of_week'] < 5, 1, 0)

# Dropping irrelevant features on training set
drop_columns = ['E/S', 'Data', 'Cidade']
X_full.drop(drop_columns, axis=1, inplace=True)

# Separating features from target on training set 
y_train = X_full['Variável']
y_train = y_train.values.reshape(-1,1)
X_train = X_full.drop('Variável', axis=1)

# Converting target on training set: categorical data into numerical format 
oe_target = OrdinalEncoder()
y_train = oe_target.fit_transform(y_train)
y_train = pd.DataFrame(y_train, columns=['Variável'])

# Adding features to training set
Xt_full = test_set.copy()
Xt_full['day_of_week'] = Xt_full['Data'].dt.dayofweek
Xt_full['weekday'] = np.where(Xt_full['day_of_week'] < 5, 1, 0)

# Dropping irrelevant features
Xt_full.drop(drop_columns, axis=1, inplace=True)

# Separating features from target on test set
y_test = Xt_full['Variável']
y_test = y_test.values.reshape(-1,1)
X_test = Xt_full.drop('Variável', axis=1)

# Converting target on test set
y_test = oe_target.transform(y_test)
y_test = pd.DataFrame(y_test, columns=['Variável'])


# Pre-processing data using pipelines for categorical (TF-ID and OrdinalEncoder) and numerical features (Standar Scaler)
bin_cat_columns = ['Quem pagou/recebeu?', 'Como pagou/recebeu?']
multi_cat_columns = ['Variável']
free_text_columns = ['O quê?']
num_columns = ['Valor']

# Individual pipelines
bin_cat_pp = Pipeline([
    ('ordinal_enc', OrdinalEncoder())
])

free_text_pp = Pipeline([
    ('tfidf', TfidfVectorizer())
])

num_pp = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    [
        ('binary_categorical', bin_cat_pp, bin_cat_columns),
        ('free_text', free_text_pp, 'O quê?'),
        ('numerical', num_pp, num_columns)
    ]
)

# Models to be tested
dict_classifiers = {
    "DecisionTreeClassifier" : DecisionTreeClassifier(max_depth = 10, random_state=0),
    "SVC" : SVC(kernel = 'rbf', C = 5, random_state=0),
    "KNN" : KNeighborsClassifier(n_neighbors = 5),
    "GBC": GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
}

df_predictions = pd.DataFrame()
cm_dict = {}
accuracy_dict = {}
macro_precision_dict = {}

warnings.filterwarnings("ignore")

# Testing models
for model, model_instantiation in dict_classifiers.items():
    
    pipeline = make_pipeline(preprocessor, model_instantiation)
    print ('\033[1m' + 'Fitting ' + model + '\033[0m')
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    df_predictions[model] = predictions 
    print ('Making predictions for model', model)
    
    cm = confusion_matrix(y_test, predictions)
    cm_dict.update({model:cm})
    print ('Confusion matrix calculated')

    accuracy = accuracy_score(y_test, predictions)
    accuracy_dict.update({model:accuracy})
    
    macro_precision = precision_score(y_test, predictions, average='macro')
    macro_precision_dict.update({model:macro_precision})
    print ('Accuracy and macro precision calculated')

    print (f"Accuracy: {accuracy:.4f} \nMacro precision: {macro_precision: .4f}")
    print ("Finished model", model)


# Plotting confusion matrix for each model to better assess performance

for model, cm in cm_dict.items():
    cm_name = f"confusion_matrix_{model}.png"
    sns.heatmap(cm, annot =True, fmt="d",cmap=plt.get_cmap('Blues'), vmax=50,
                xticklabels=oe_target.categories_[0], yticklabels=oe_target.categories_[0])
    acc = "{:.4f}".format(accuracy_dict.get(model))
    macro_p = "{:.4f}".format(macro_precision_dict.get(model))
    plt.title("Confusion Matrix - " + model + ' - Accuracy: ' + acc + '/ Macro Precision: ' + macro_p)
    plt.savefig(os.path.join(path_results, cm_name), bbox_inches="tight")
    plt.close()


# Hyperparameter tuning for best model (SVC) using GridSearchCV
SVC_parameters = {
    'classifier__C': [1, 2, 3, 4, 5], 
	'classifier__gamma': ['scale', 'auto'], 
	'classifier__kernel': ['rbf', 'linear'],
    'classifier__tol' : [0.0001, 0.001],
    'classifier__decision_function_shape': ['ovo', 'ovr']
    }

svc_model = SVC(random_state=42)
SVC_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', svc_model)
                              ]
                       )

SVC_gridsearch = GridSearchCV(SVC_pipeline,SVC_parameters,cv=2)
SVC_gridsearch.fit(X_train,y_train)

joblib.dump(SVC_gridsearch, 'SVC_model.pkl')

SVC_GS_predictions = SVC_gridsearch.best_estimator_.predict(X_test)
cm_SVC = confusion_matrix(y_test, SVC_GS_predictions)
sns.heatmap(cm_SVC, annot =True, fmt="d",cmap=plt.get_cmap('Blues'), vmax=50,
                xticklabels=oe_target.categories_[0], yticklabels=oe_target.categories_[0])
plt.title("Confusion Matrix - SVC")
plt.savefig(os.path.join(path_results, "FinalModel_SVC.png"), bbox_inches="tight")


X_jabiru = df_jabiru.copy()

X_jabiru['day_of_week'] = X_jabiru['Data'].dt.dayofweek
X_jabiru['weekday'] = np.where(X_jabiru['day_of_week'] < 5, 1, 0)

X_jabiru = X_jabiru[(X_jabiru['O quê?'] != 'Saque') & (X_jabiru['O quê?'] != 'Transferência')] 

X_jabiru.drop(drop_columns, axis=1, inplace=True)
X_jabiru.drop('Variável', axis=1, inplace=True)

SVC_result = SVC_gridsearch.best_estimator_.predict(X_jabiru)
SVC_result = oe_target.inverse_transform(SVC_result.reshape(-1, 1))


X_jabiru.reset_index(inplace=True)
X_jabiru['SVC'] = pd.DataFrame(SVC_result)
X_jabiru.set_index('index', inplace=True)

teste = X_jabiru[['SVC']]
teste.index.name = None

final_result = df_jabiru.join(teste, how='left')
final_result.drop('Variável', axis=1, inplace=True)
final_result.rename({'SVC':'Variável'}, axis=1, inplace=True)
final_result_name = 'Results_' + file_name
final_result.to_excel(os.path.join(path_results, final_result_name), index=False)