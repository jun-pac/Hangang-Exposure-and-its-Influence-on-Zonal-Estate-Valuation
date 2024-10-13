import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import statsmodels.api as sm
import numpy as np
import csv
from tqdm import tqdm
from sklearn.inspection import permutation_importance
import shap

data=np.load("./all_datas.npy")
print(f"data.shape : {data.shape}")
# Price, area, transaction_t, floor, t_build

def get_performance_error(predicted_price, X_test, y_test):
    error=0
    baseline_error=0
    for i in range(len(y_test)):
        error+=abs(predicted_price[i]-y_test[i])/y_test[i]
        baseline_error+=abs(X_test[i][0]-y_test[i])/y_test[i]
    print(f"Average error: {float(error/len(y_test)*100):.2f}")



def get_performance(predicted_price, X_test, y_test):
    error=0
    baseline_error=0
    for i in range(len(y_test)):
        error+=abs(predicted_price[i]-y_test[i])/y_test[i]
        baseline_error+=abs(X_test[i][0]-y_test[i])/y_test[i]
    print(f"Average error: {float(error/len(y_test)*100):.2f}")


def get_performance_train(predicted_price,X_train, y_train):
    error=0
    baseline_error=0
    for i in range(len(y_train)):
        error+=abs(predicted_price[i]-y_train[i])/y_train[i]
        baseline_error+=abs(X_train[i][0]-y_train[i])/y_train[i]
    print(f"Average train error: {float(error/len(y_train)*100):.2f}")


def OLS1(printflag):
    print(f"OLS_model1: area + transaction_t + floor + t_build")
    X = data[:, 1:] 
    print(f"X.shape: {X.shape}")
    Y = data[:, 0]   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

    X_OLS=sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_OLS, formula = 'Price ~ area + transaction_t + floor + t_build')
    results = ols_model.fit()
    if(printflag):
        print(results.summary())
    X_OLS=sm.add_constant(X_test)
    predicted_price = results.predict(X_OLS)
    X_OLS=sm.add_constant(X_train)
    predicted_train_price = results.predict(X_OLS)

    get_performance(predicted_price, X_test, y_test)
    get_performance_train(predicted_train_price,X_train, y_train)
    print()


def OLS2(printflag):
    # Exclude transaction_t
    print(f"OLS_model2: area + floor + t_build")
    datas=data.T
    datas=np.concatenate([datas[1:2],datas[3:]])
    X = datas.T  
    print(f"X.shape: {X.shape}")
    Y = data[:, 0]  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
    X_OLS=sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_OLS, formula = 'Price ~ area + floor + t_build')
    results = ols_model.fit()
    if(printflag):
        print(results.summary())
    X_OLS=sm.add_constant(X_test)
    predicted_price = results.predict(X_OLS)
    X_OLS=sm.add_constant(X_train)
    predicted_train_price = results.predict(X_OLS)

    get_performance(predicted_price, X_test, y_test)
    get_performance_train(predicted_train_price,X_train, y_train)
    print()

def OLS3(printflag):
    # Add transaction_t2
    print(f"OLS_model3: area + transaction_t + transaction_t2 + floor + t_build")
    datas=data.T
    transaction_t=datas[2:3]
    transaction_t2=transaction_t*transaction_t
    datas=np.concatenate([datas[1:2],transaction_t,transaction_t2,datas[3:]])
    X = datas.T  
    print(f"X.shape: {X.shape}")
    Y = data[:, 0]  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
    X_OLS=sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_OLS, formula = 'Price ~ area + transaction_t + transaction_t2 + floor + t_build')
    results = ols_model.fit()
    if(printflag):
        print(results.summary())
    X_OLS=sm.add_constant(X_test)
    predicted_price = results.predict(X_OLS)
    X_OLS=sm.add_constant(X_train)
    predicted_train_price = results.predict(X_OLS)

    get_performance(predicted_price, X_test, y_test)
    get_performance_train(predicted_train_price,X_train, y_train)
    print()




# OLS
OLS1(True)
OLS2(True)
OLS3(True)

import matplotlib.pyplot as plt

def run_random_forest(X_train, X_test, y_train, y_test):
    # Random Forest Regressor
    print(f"Random Forest Regressor")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_train_pred_rf = rf_model.predict(X_train)

    # Performance
    get_performance(y_pred_rf, X_test, y_test)
    get_performance_train(y_train_pred_rf, X_train, y_train)
    print()
    perm_importance = permutation_importance(rf_model, X_train, y_train, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(range(X_train.shape[1]), perm_importance.importances_mean[sorted_idx])
    plt.yticks(range(X_train.shape[1]), [f"Feature {i}" for i in sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()


    # Feature importances
    # importances = rf_model.feature_importances_

    # # Plotting Gini importance
    # indices = np.argsort(importances)
    # plt.barh(range(len(importances)), importances[indices])
    # plt.yticks(range(len(importances)), [f"Feature {i}" for i in indices])
    # plt.xlabel("Gini Importance")
    # plt.show()





def run_gradient_boosting(X_train, X_test, y_train, y_test):
    # Gradient Boosting Regressor
    print(f"Gradient Boosting Regressor")
    gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbm_model.fit(X_train, y_train)

    # Predictions
    y_pred_gbm = gbm_model.predict(X_test)
    y_train_pred_gbm = gbm_model.predict(X_train)

    # Performance
    get_performance(y_pred_gbm, X_test, y_test)
    get_performance_train(y_train_pred_gbm, X_train, y_train)
    print()



def run_dnn(X_train, X_test, y_train, y_test):
    print(f"DNN")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    class DNN(nn.Module):
        def __init__(self):
            super(DNN, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    model = DNN()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch} finished")

    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        y_pred_dnn = model(X_test_tensor)
        y_train_pred_dnn = model(X_train_tensor)

    # Performance
    get_performance(y_pred_dnn, X_test, y_test)
    get_performance_train(y_train_pred_dnn, X_train, y_train)
    print()

    # # SHAP explanation
    # explainer = shap.DeepExplainer(model, X_train_tensor)
    # shap_values = explainer.shap_values(X_train_tensor)

    # # SHAP Summary plot
    # shap.summary_plot(shap_values, X_train)





# Selectdata configure
print()
print(f"=========== OLS1: area + transaction_t + floor + t_build ===========")
print()
X = data[:, 1:] 
print(f"X.shape: {X.shape}")
Y = data[:, 0]   
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
run_random_forest(X_train, X_test, y_train, y_test)
run_gradient_boosting(X_train, X_test, y_train, y_test)
# run_dnn(X_train, X_test, y_train, y_test)



print()
print(f"=========== OLS2: area + floor + t_build ===========")
print()
datas=data.T
datas=np.concatenate([datas[1:3],datas[4:5],datas[6:7]])
X = datas.T  
Y = data[:, 0]  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
run_random_forest(X_train, X_test, y_train, y_test)
run_gradient_boosting(X_train, X_test, y_train, y_test)
# run_dnn(X_train, X_test, y_train, y_test)



print()
print(f"=========== OLS3: area + transaction_t + transaction_t2 + floor + t_build ===========")
print()
datas=data.T
transaction_t=datas[2:3]
transaction_t2=transaction_t*transaction_t
datas=np.concatenate([datas[1:2],transaction_t,transaction_t2,datas[3:]])
X = datas.T  
Y = data[:, 0]  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
run_random_forest(X_train, X_test, y_train, y_test)
run_gradient_boosting(X_train, X_test, y_train, y_test)
# run_dnn(X_train, X_test, y_train, y_test)
