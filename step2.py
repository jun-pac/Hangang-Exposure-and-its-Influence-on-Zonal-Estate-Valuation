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

data=np.load("./step2_data_wo_time.npy")
print(f"data.shape : {data.shape}")
# Price, Baseline_price, area, transaction_t, floor, t_build, hangang_angle, distance, obscure_angle, sound_barrier, highway, north_region


def get_performance_error(predicted_price):
    error=0
    baseline_error=0
    for i in range(len(y_test)):
        # print(f"Actual: {y_test[i]}, Predicted: {predicted_price[i]:.2f}, Baseline: {X_test[i][0]:.2f}")
        error+=abs(predicted_price[i]-y_test[i])/y_test[i]
        baseline_error+=abs(X_test[i][0]-y_test[i])/y_test[i]
    print(f"Average error: {float(error/len(y_test)*100):.2f}")
    print(f"Baseline error: {baseline_error/len(y_test)*100:.2f}")



# OLS
X = data[:, 1:]  # Explanatory variables (columns 1 to 11)
Y = data[:, 0]   # Actual prices (column 0)

X_OLS = sm.add_constant(X)
ols_model = sm.OLS(Y, X_OLS)
results = ols_model.fit()
predicted_price = results.predict(X_OLS)
# print(results.summary())



# Train, Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
# print(f"X.shape: {X.shape}") 
# print(f"Y.shape: {Y.shape}")
# print(f"X_train.shape: {X_train.shape}")
# print(f"y_train.shape: {y_train.shape}")
# print(f"X_test.shape: {X_test.shape}")
# print(f"y_test.shape: {y_test.shape}")
'''
X.shape: (212, 11)
Y.shape: (212,)
X_train.shape: (169, 11)
y_train.shape: (169,)
X_test.shape: (43, 11)
y_test.shape: (43,)
'''


def get_performance(predicted_price):
    error=0
    baseline_error=0
    for i in range(len(y_test)):
        # print(f"Actual: {y_test[i]}, Predicted: {predicted_price[i]:.2f}, Baseline: {X_test[i][0]:.2f}")
        error+=(predicted_price[i]-y_test[i])**2
        baseline_error+=(X_test[i][0]-y_test[i])**2
    print(f"Average error: {np.sqrt(float(error/len(y_test))):.2f}")
    print(f"Baseline error: {np.sqrt(baseline_error/len(y_test)):.2f}")


def get_performance_train(predicted_price):
    error=0
    baseline_error=0
    for i in range(len(y_train)):
        error+=(predicted_price[i]-y_train[i])**2
        baseline_error+=(X_train[i][0]-y_train[i])**2
    print(f"Average train error: {np.sqrt(float(error/len(y_train))):.2f}")
    print(f"Baseline train error: {np.sqrt(baseline_error/len(y_train)):.2f}")





print(f"OLS")
X_OLS=sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_OLS)
results = ols_model.fit()
print(results.summary())
X_OLS=sm.add_constant(X_test)
predicted_price = results.predict(X_OLS)
X_OLS=sm.add_constant(X_train)
predicted_train_price = results.predict(X_OLS)


get_performance(predicted_price)
get_performance_train(predicted_train_price)

print()



# Random Forest Regressor
print(f"Random Forest Regressor")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_train_pred_rf = rf_model.predict(X_train)

# Performance
get_performance(y_pred_rf)
get_performance_train(y_train_pred_rf)
print()





# Gradient Boosting Regressor
print(f"Gradient Boosting Regressor")
gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm_model.fit(X_train, y_train)

# Predictions
y_pred_gbm = gbm_model.predict(X_test)
y_train_pred_gbm = gbm_model.predict(X_train)



# Performance
get_performance(y_pred_gbm)
get_performance_train(y_train_pred_gbm)
print()




# DNN
print(f"DNN")
# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Define a simple DNN model
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

# Instantiate the model
model = DNN()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in tqdm(range(epochs)):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Switch to evaluation mode
with torch.no_grad():
    y_pred_dnn = model(X_test_tensor)
    y_train_pred_dnn = model(X_train_tensor)

# Performance
get_performance(y_pred_dnn)
get_performance_train(y_train_pred_dnn)
print()