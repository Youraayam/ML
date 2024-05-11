# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:36:07 2024

@author: aayam
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:52:12 2024

@author: aayam
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
import re
import os
from datetime import datetime, timedelta
from collections import Counter
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


import warnings
warnings.filterwarnings("ignore")

for i in range (0,200):

    data_input = pd.read_csv('machinelearning_141.csv')
    
    data_input.columns
    
    variable_scale = MinMaxScaler()  #StandardScaler()
    train_percent = 60         #how many percent is the training dataset you wish from big dataset you've uploaded?
    target_col    = ['Average']
    
    
    data_come = data_input[['feel_air','feel_health','temp_o','rh_o','winds_o','temp_coldhot','Average']].copy()
    
    xb = data_come.drop(target_col, axis=1)
    x = variable_scale.fit_transform(xb)
    y = data_input.loc[:, target_col]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = (train_percent/100), shuffle = True)
    xb.columns #showing what kind of inputs you have
    
    model = ExtraTreesRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth'    : [3],
    'random_state' : [1,2,5,10,25,50,75,100]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    ET_Reg = grid_search.best_estimator_
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    y_train_pred_ETR = ET_Reg.predict(x_train)
    y_test_pred_ETR = ET_Reg.predict(x_test)
    
    print("Accuracy score (training): {:.3f}".format(ET_Reg.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(ET_Reg.score(x_test, y_test)))
    
    f_importance_ETR = pd.DataFrame({'features':xb.columns,'feature_importances ETR':ET_Reg.feature_importances_}).sort_values(by = 'feature_importances ETR',ascending = False)
    
    print(f_importance_ETR)
    
    r2_ETR = r2_score(y_test,y_test_pred_ETR).round(3)
    mse_ETR = mean_squared_error(y_test,y_test_pred_ETR).round(3)
    mae_ETR = mean_absolute_error(y_test,y_test_pred_ETR).round(3)
    rmse_ETR = np.sqrt(mean_squared_error(y_test,y_test_pred_ETR)).round(3)
    CV_value_ETR = explained_variance_score(y_test,y_test_pred_ETR).round(3)
    
    print('Extra Tree Performance')
    print('R2 : ', r2_ETR)
    print('MSE : ', mse_ETR)
    print('MAE : ', mae_ETR)
    print('RMSE : ', rmse_ETR)
    print('CV : ', CV_value_ETR)
    
    
    
    
    
    model =RandomForestRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth': [3],
    'random_state': [1,2,5,10,25,50,100,200]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    
    model_RFR = grid_search.best_estimator_
    
    y_train_pred_RFR = model_RFR.predict(x_train)
    y_test_pred_RFR = (model_RFR.predict(x_test)).reshape(-1,1)
    
    
    print("Accuracy score (training): {:.3f}".format(model_RFR.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(model_RFR.score(x_test, y_test)))
    
    f_importance_RFR = pd.DataFrame({'features':xb.columns,
                                'feature_importances':model_RFR.feature_importances_}).sort_values(by = 'feature_importances',ascending = False)
    print (f_importance_RFR)
    
    r2_RFR = r2_score(y_test,y_test_pred_RFR).round(3)
    mse_RFR = mean_squared_error(y_test,y_test_pred_RFR).round(3)
    mae_RFR = mean_absolute_error(y_test,y_test_pred_RFR).round(3)
    rmse_RFR = np.sqrt(mean_squared_error(y_test,y_test_pred_RFR)).round(3)
    CV_value_RFR = explained_variance_score(y_test,y_test_pred_RFR).round(3)
    
    print('Random Forest Regression Performance')
    print('R2 : ', r2_RFR)
    print('MSE : ', mse_RFR)
    print('MAE : ', mae_RFR)
    print('RMSE : ', rmse_RFR)
    print('CV : ', CV_value_RFR)
    
    
    data_result_reg = np.array([['Algorithm', 'R2', 'MSE', 'MAE', 'RMSE', 'CV Value'],
                ['Extra Trees Regression', r2_ETR, mse_ETR, mae_ETR, rmse_ETR, CV_value_ETR],
                ['Random Forest Regression', r2_RFR, mse_RFR, mae_RFR, rmse_RFR, CV_value_RFR]
                 ])
    
    table_regression = pd.DataFrame(data=data_result_reg[1:, 1:],
                     index = data_result_reg[1:,0],
                     columns=(data_result_reg[0,1:])).sort_values('RMSE', ascending = True)
    
    
    print(table_regression)
    
    plt.figure(figsize=(22,6))
    plt.plot(pd.DataFrame(y_test_pred_ETR), label = 'Prediction : ET Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test_pred_RFR), label = 'Prediction : RF Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test).reset_index(level=0, drop=True), label = 'Real' , linewidth= 0.7 )
    plt.legend(fontsize=30)
    plt.title('Comparision between Real and Predicted value',fontsize=30)
    plt.ylabel('Scale (0-4)',fontsize=30)
    plt.xlabel('Count',fontsize=30)
    plt.ylim(90, 100)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.show()
    
    
    x1 = pd.DataFrame(y_test_pred_ETR)
    x2 = pd.DataFrame(pd.DataFrame(y_test).reset_index(level=0))
    x3 = pd.DataFrame(pd.DataFrame(y_test_pred_RFR))
    
    x22 = x2.iloc[:, 1]

 
    fig, ax = plt.subplots(figsize = (16, 16))
    
    a,b = np.polyfit(x22,x1,1)
    c,d = np.polyfit(x22,x3,1)

    plt.scatter(x22,x1,s=120, alpha=0.9, edgecolors="k", color='purple')
    plt.plot(x22, a*x22+b, color='steelblue', linestyle='--', linewidth=2, label=f" Y = {a}X + {b}")
    plt.title('Relationship between predicted (Relative performance) and reported (Relative performance)',fontsize=30)
    plt.text((max(x22)-0.5),98, f" $R^2$ = {r2_ETR}", size=30, ha='center', va='center')
    plt.ylabel('Predicted (Relative performance)',fontsize=30)
    plt.xlabel('Reported (Relative performance)',fontsize=30)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.show()
    
    fig, ax = plt.subplots(figsize = (16, 16))
    plt.scatter(x22,x3,s=120, alpha=0.9, edgecolors="k", color='green')
    plt.plot(x22, c*x22+d, color='steelblue', linestyle='--', linewidth=2, label=f" Y = {c}X + {d}")
    plt.title('Relationship between predicted (Relative performance) and reported (Relative performance)',fontsize=30)
    plt.text((max(x22)-0.5),98 , f" $R^2$ = {r2_RFR}", size=30, ha='center', va='center')
    plt.ylabel('Predicted (Relative performance)',fontsize=30)
    plt.xlabel('Reported (Relative performance)',fontsize=30)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.show()


# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:12:49 2024

@author: aayam
"""




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
import re
import os
from datetime import datetime, timedelta
from collections import Counter
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


import warnings
warnings.filterwarnings("ignore")

for i in range (0,200):

    data_input = pd.read_csv('machinelearning_302.csv')
    
    data_input.columns
    
    variable_scale = MinMaxScaler()  #StandardScaler()
    train_percent = 80         #how many percent is the training dataset you wish from big dataset you've uploaded?
    target_col    = ['feel_health']
    
    
    data_come = data_input[['temp_a','rh_a','co2_a','voc_a','temp_o','rh_o','winds_o','feel_health']].copy()
    
    xb = data_come.drop(target_col, axis=1)
    x = variable_scale.fit_transform(xb)
    y = data_input.loc[:, target_col]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = (train_percent/100), shuffle = True)
    xb.columns #showing what kind of inputs you have
    
    model = ExtraTreesRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth'    : [3],
    'random_state' : [1,2,5,10,25,50,75,100]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    ET_Reg = grid_search.best_estimator_
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    y_train_pred_ETR = ET_Reg.predict(x_train)
    y_test_pred_ETR = ET_Reg.predict(x_test)
    
    print("Accuracy score (training): {:.3f}".format(ET_Reg.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(ET_Reg.score(x_test, y_test)))
    
    f_importance_ETR = pd.DataFrame({'features':xb.columns,'feature_importances ETR':ET_Reg.feature_importances_}).sort_values(by = 'feature_importances ETR',ascending = False)
    
    print(f_importance_ETR)
    
    r2_ETR = r2_score(y_test,y_test_pred_ETR).round(3)
    mse_ETR = mean_squared_error(y_test,y_test_pred_ETR).round(3)
    mae_ETR = mean_absolute_error(y_test,y_test_pred_ETR).round(3)
    rmse_ETR = np.sqrt(mean_squared_error(y_test,y_test_pred_ETR)).round(3)
    CV_value_ETR = explained_variance_score(y_test,y_test_pred_ETR).round(3)
    
    print('Extra Tree Performance')
    print('R2 : ', r2_ETR)
    print('MSE : ', mse_ETR)
    print('MAE : ', mae_ETR)
    print('RMSE : ', rmse_ETR)
    print('CV : ', CV_value_ETR)
    
    
    
    
    
    model =RandomForestRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth': [3],
    'random_state': [1,2,5,10,25,50,100,200]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    
    model_RFR = grid_search.best_estimator_
    
    y_train_pred_RFR = model_RFR.predict(x_train)
    y_test_pred_RFR = (model_RFR.predict(x_test)).reshape(-1,1)
    
    
    print("Accuracy score (training): {:.3f}".format(model_RFR.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(model_RFR.score(x_test, y_test)))
    
    f_importance_RFR = pd.DataFrame({'features':xb.columns,
                                'feature_importances':model_RFR.feature_importances_}).sort_values(by = 'feature_importances',ascending = False)
    print (f_importance_RFR)
    
    r2_RFR = r2_score(y_test,y_test_pred_RFR).round(3)
    mse_RFR = mean_squared_error(y_test,y_test_pred_RFR).round(3)
    mae_RFR = mean_absolute_error(y_test,y_test_pred_RFR).round(3)
    rmse_RFR = np.sqrt(mean_squared_error(y_test,y_test_pred_RFR)).round(3)
    CV_value_RFR = explained_variance_score(y_test,y_test_pred_RFR).round(3)
    
    print('Random Forest Regression Performance')
    print('R2 : ', r2_RFR)
    print('MSE : ', mse_RFR)
    print('MAE : ', mae_RFR)
    print('RMSE : ', rmse_RFR)
    print('CV : ', CV_value_RFR)
    
    
    data_result_reg = np.array([['Algorithm', 'R2', 'MSE', 'MAE', 'RMSE', 'CV Value'],
                ['Extra Trees Regression', r2_ETR, mse_ETR, mae_ETR, rmse_ETR, CV_value_ETR],
                ['Random Forest Regression', r2_RFR, mse_RFR, mae_RFR, rmse_RFR, CV_value_RFR]
                 ])
    
    table_regression = pd.DataFrame(data=data_result_reg[1:, 1:],
                     index = data_result_reg[1:,0],
                     columns=(data_result_reg[0,1:])).sort_values('RMSE', ascending = True)
    
    
    print(table_regression)
    
    plt.figure(figsize=(22,6))
    plt.plot(pd.DataFrame(y_test_pred_ETR), label = 'Prediction : ET Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test_pred_RFR), label = 'Prediction : RF Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test).reset_index(level=0, drop=True), label = 'Real' , linewidth= 0.7 )
    plt.legend(fontsize=30)
    plt.title('Comparision between Real and Predicted value',fontsize=30)
    plt.ylabel('Scale (0-7)',fontsize=30)
    plt.xlabel('Count',fontsize=30)
    plt.ylim(0, 7)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.show()




# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:12:49 2024

@author: aayam
"""




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
import re
import os
from datetime import datetime, timedelta
from collections import Counter
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


import warnings
warnings.filterwarnings("ignore")

for i in range (0,200):

    data_input = pd.read_csv('machinelearning_302.csv')
    
    data_input.columns
    
    variable_scale = MinMaxScaler()  #StandardScaler()
    train_percent = 80         #how many percent is the training dataset you wish from big dataset you've uploaded?
    target_col    = ['feel_health']
    
    
    data_come = data_input[['temp_a','rh_a','co2_a','voc_a','temp_o','rh_o','winds_o','feel_health']].copy()
    
    xb = data_come.drop(target_col, axis=1)
    x = variable_scale.fit_transform(xb)
    y = data_input.loc[:, target_col]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = (train_percent/100), shuffle = True)
    xb.columns #showing what kind of inputs you have
    
    model = ExtraTreesRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth'    : [3],
    'random_state' : [1,2,5,10,25,50,75,100]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    ET_Reg = grid_search.best_estimator_
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    y_train_pred_ETR = ET_Reg.predict(x_train)
    y_test_pred_ETR = ET_Reg.predict(x_test)
    
    print("Accuracy score (training): {:.3f}".format(ET_Reg.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(ET_Reg.score(x_test, y_test)))
    
    f_importance_ETR = pd.DataFrame({'features':xb.columns,'feature_importances ETR':ET_Reg.feature_importances_}).sort_values(by = 'feature_importances ETR',ascending = False)
    
    print(f_importance_ETR)
    
    r2_ETR = r2_score(y_test,y_test_pred_ETR).round(3)
    mse_ETR = mean_squared_error(y_test,y_test_pred_ETR).round(3)
    mae_ETR = mean_absolute_error(y_test,y_test_pred_ETR).round(3)
    rmse_ETR = np.sqrt(mean_squared_error(y_test,y_test_pred_ETR)).round(3)
    CV_value_ETR = explained_variance_score(y_test,y_test_pred_ETR).round(3)
    
    print('Extra Tree Performance')
    print('R2 : ', r2_ETR)
    print('MSE : ', mse_ETR)
    print('MAE : ', mae_ETR)
    print('RMSE : ', rmse_ETR)
    print('CV : ', CV_value_ETR)
    
    
    
    
    
    model =RandomForestRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth': [3],
    'random_state': [1,2,5,10,25,50,100,200]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    
    model_RFR = grid_search.best_estimator_
    
    y_train_pred_RFR = model_RFR.predict(x_train)
    y_test_pred_RFR = (model_RFR.predict(x_test)).reshape(-1,1)
    
    
    print("Accuracy score (training): {:.3f}".format(model_RFR.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(model_RFR.score(x_test, y_test)))
    
    f_importance_RFR = pd.DataFrame({'features':xb.columns,
                                'feature_importances':model_RFR.feature_importances_}).sort_values(by = 'feature_importances',ascending = False)
    print (f_importance_RFR)
    
    r2_RFR = r2_score(y_test,y_test_pred_RFR).round(3)
    mse_RFR = mean_squared_error(y_test,y_test_pred_RFR).round(3)
    mae_RFR = mean_absolute_error(y_test,y_test_pred_RFR).round(3)
    rmse_RFR = np.sqrt(mean_squared_error(y_test,y_test_pred_RFR)).round(3)
    CV_value_RFR = explained_variance_score(y_test,y_test_pred_RFR).round(3)
    
    print('Random Forest Regression Performance')
    print('R2 : ', r2_RFR)
    print('MSE : ', mse_RFR)
    print('MAE : ', mae_RFR)
    print('RMSE : ', rmse_RFR)
    print('CV : ', CV_value_RFR)
    
    
    data_result_reg = np.array([['Algorithm', 'R2', 'MSE', 'MAE', 'RMSE', 'CV Value'],
                ['Extra Trees Regression', r2_ETR, mse_ETR, mae_ETR, rmse_ETR, CV_value_ETR],
                ['Random Forest Regression', r2_RFR, mse_RFR, mae_RFR, rmse_RFR, CV_value_RFR]
                 ])
    
    table_regression = pd.DataFrame(data=data_result_reg[1:, 1:],
                     index = data_result_reg[1:,0],
                     columns=(data_result_reg[0,1:])).sort_values('RMSE', ascending = True)
    
    
    print(table_regression)
    
    plt.figure(figsize=(22,6))
    plt.plot(pd.DataFrame(y_test_pred_ETR), label = 'Prediction : ET Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test_pred_RFR), label = 'Prediction : RF Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test).reset_index(level=0, drop=True), label = 'Real' , linewidth= 0.7 )
    plt.legend(fontsize=30)
    plt.title('Comparision between Real and Predicted value',fontsize=30)
    plt.ylabel('Scale (0-7)',fontsize=30)
    plt.xlabel('Count',fontsize=30)
    plt.ylim(0, 7)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.show()




# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:12:09 2024

@author: aayam
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:12:49 2024

@author: aayam
"""




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics
import re
import os
from datetime import datetime, timedelta
from collections import Counter
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


import warnings
warnings.filterwarnings("ignore")

for i in range (0,200):

    data_input = pd.read_csv('machinelearning_23.csv')
    
    data_input.columns
    
    variable_scale = MinMaxScaler()  #StandardScaler()
    train_percent = 80         #how many percent is the training dataset you wish from big dataset you've uploaded?
    target_col    = ['Average']
    
    
    data_come = data_input[['feel_air','temp_coldhot','temp_o','rh_o','winds_o','feel_health','Average']].copy()
    
    xb = data_come.drop(target_col, axis=1)
    x = variable_scale.fit_transform(xb)
    y = data_input.loc[:, target_col]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = (train_percent/100), shuffle = True)
    xb.columns #showing what kind of inputs you have
    
    model = ExtraTreesRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth'    : [3],
    'random_state' : [1,2,5,10,25,50,75,100]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    ET_Reg = grid_search.best_estimator_
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    y_train_pred_ETR = ET_Reg.predict(x_train)
    y_test_pred_ETR = ET_Reg.predict(x_test)
    
    print("Accuracy score (training): {:.3f}".format(ET_Reg.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(ET_Reg.score(x_test, y_test)))
    
    f_importance_ETR = pd.DataFrame({'features':xb.columns,'feature_importances ETR':ET_Reg.feature_importances_}).sort_values(by = 'feature_importances ETR',ascending = False)
    
    print(f_importance_ETR)
    
    r2_ETR = r2_score(y_test,y_test_pred_ETR).round(3)
    mse_ETR = mean_squared_error(y_test,y_test_pred_ETR).round(3)
    mae_ETR = mean_absolute_error(y_test,y_test_pred_ETR).round(3)
    rmse_ETR = np.sqrt(mean_squared_error(y_test,y_test_pred_ETR)).round(3)
    CV_value_ETR = explained_variance_score(y_test,y_test_pred_ETR).round(3)
    
    print('Extra Tree Performance')
    print('R2 : ', r2_ETR)
    print('MSE : ', mse_ETR)
    print('MAE : ', mae_ETR)
    print('RMSE : ', rmse_ETR)
    print('CV : ', CV_value_ETR)
    
    
    
    
    
    model =RandomForestRegressor()
    param_grid = {
    'criterion'    : ['squared_error'],
    'max_depth': [3],
    'random_state': [1,2,5,10,25,50,100,200]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train )
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    
    model_RFR = grid_search.best_estimator_
    
    y_train_pred_RFR = model_RFR.predict(x_train)
    y_test_pred_RFR = (model_RFR.predict(x_test)).reshape(-1,1)
    
    
    print("Accuracy score (training): {:.3f}".format(model_RFR.score(x_train, y_train)))
    print("Accuracy score (validation): {:.3f}".format(model_RFR.score(x_test, y_test)))
    
    f_importance_RFR = pd.DataFrame({'features':xb.columns,
                                'feature_importances':model_RFR.feature_importances_}).sort_values(by = 'feature_importances',ascending = False)
    print (f_importance_RFR)
    
    r2_RFR = r2_score(y_test,y_test_pred_RFR).round(3)
    mse_RFR = mean_squared_error(y_test,y_test_pred_RFR).round(3)
    mae_RFR = mean_absolute_error(y_test,y_test_pred_RFR).round(3)
    rmse_RFR = np.sqrt(mean_squared_error(y_test,y_test_pred_RFR)).round(3)
    CV_value_RFR = explained_variance_score(y_test,y_test_pred_RFR).round(3)
    
    print('Random Forest Regression Performance')
    print('R2 : ', r2_RFR)
    print('MSE : ', mse_RFR)
    print('MAE : ', mae_RFR)
    print('RMSE : ', rmse_RFR)
    print('CV : ', CV_value_RFR)
    
    
    data_result_reg = np.array([['Algorithm', 'R2', 'MSE', 'MAE', 'RMSE', 'CV Value'],
                ['Extra Trees Regression', r2_ETR, mse_ETR, mae_ETR, rmse_ETR, CV_value_ETR],
                ['Random Forest Regression', r2_RFR, mse_RFR, mae_RFR, rmse_RFR, CV_value_RFR]
                 ])
    
    table_regression = pd.DataFrame(data=data_result_reg[1:, 1:],
                     index = data_result_reg[1:,0],
                     columns=(data_result_reg[0,1:])).sort_values('RMSE', ascending = True)
    
    
    print(table_regression)
    
    plt.figure(figsize=(22,6))
    plt.plot(pd.DataFrame(y_test_pred_ETR), label = 'Prediction : ET Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test_pred_RFR), label = 'Prediction : RF Reg' , linewidth= 0.9 )
    plt.plot(pd.DataFrame(y_test).reset_index(level=0, drop=True), label = 'Real' , linewidth= 0.7 )
    plt.legend(fontsize=30)
    plt.title('Comparision between Real and Predicted value',fontsize=30)
    plt.ylabel('Scale (90-100)',fontsize=30)
    plt.xlabel('Count',fontsize=30)
    plt.ylim(90, 100)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.show()




