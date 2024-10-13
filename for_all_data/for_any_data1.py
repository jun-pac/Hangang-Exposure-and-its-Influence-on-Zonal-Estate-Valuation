import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
import csv
import tqdm

apartment_num={}
apartment_name=[]

def get_identifier_from_line(line):
    identifier=line[0][6:]+' / '+line[4]
    return identifier

num=0
step1_datas=[] # Step1 regression data 
all_datas=[]
error_cnt=0
with open("./any_data.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        try:
            line=line[0].split(',')[1:]
            if(i==0):
                print(line)
                continue

            identifier=get_identifier_from_line(line)
            if not (identifier in apartment_num):
                apartment_num[identifier]=num
                apartment_name.append(identifier)
                num+=1
                step1_datas.append([])
            
            building_num=line[9]
            area=eval(line[5])
            transaction_t=(eval(line[6][:4])-2020)*12+int(line[6][4:6])-1
            price=eval(line[8])
            floor=eval(line[10])
            t_build=eval(line[13])
            step1_datas[apartment_num[identifier]].append([price,area,transaction_t,floor,t_build])
        except:
            error_cnt+=1
            continue
        
        all_datas.append([price,area,transaction_t,floor,t_build])


ts1=0
# Step 1 Regression
excluded_apart=[]
print(f"error_cnt: {error_cnt}")

all_datas=np.array(all_datas)
print(f"all_datas.shape: {all_datas.shape}")
np.save("./all_datas",all_datas)
print(f"num: {num}")
lens=[]
for i in range(num):
    lens.append(len(step1_datas[i]))
lens.sort()

# print(lens[-99])
thres=6
if(num>100):
    # print(lens[-100:])
    print(f"thres: lens[-99]")    
    thres=lens[-99]

print(f"Step 1 OLS")
for i in range(num):
    if(len(step1_datas[i])<thres):
        excluded_apart.append(i)
        continue
    data=np.array(step1_datas[i])
    data=data.T
    Y=data[0]
    area=data[1]
    floor=data[3]
    transaction_t=data[3]
    transaction_t2=data[3]**2    

    if(area.min()==area.max() or floor.min()==floor.max()):
        excluded_apart.append(i)
        continue
    print(f"For apartment {apartment_name[i]}: {len(step1_datas[i])}")
    ts1+=len(step1_datas[i])
    
    

    # X=area # Only area
    X=np.column_stack((area,floor))
    # X=np.column_stack((area,floor,transaction_t,transaction_t2))
    X=sm.add_constant(X)

    ols_model = sm.OLS(Y, X)
    results = ols_model.fit()
    predicted_price = results.predict(X)

    print(f"results.params : ",end='')
    for item in results.params:
        print(f"{item:.2f}",end=' ')
    print()
    print(f"results.rsquared : {results.rsquared:.4f}, results.rsquared_adj : {results.rsquared_adj:.4f}")
    # print(f"results.ssr: {results.ssr:.4f}, Average ssr: {results.ssr/len(step1_datas[i]):.4f}, std: {np.sqrt(results.ssr/len(step1_datas[i])):.4f}")
    
    print(f"==================================================================")


for i in range(num):
    if not (i in excluded_apart):
        print(f"{i}th apartment name; {apartment_name[i]}, # step1: {len(step1_datas[i])}")

plots=11
plote=20

plt.figure(figsize=(10, 6))
for i in range(num):
    if (i in excluded_apart):
        continue
    data=np.array(step1_datas[i])
    data=data.T
    Y=data[0] # Price
    area=data[1]
    floor=data[3]
    transaction_t=data[3]
    plt.scatter(area,Y,label=f'Complex {i}',alpha=0.2)
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend(title="Apartment Complex")

plt.figure(figsize=(10, 6))
data=all_datas
data=data.T
Y=data[1] # area
X=data[4] # t_build
plt.scatter(X,Y,label=f'Complex {i}',alpha=0.2)
plt.xlabel('T-build(month)')
plt.ylabel('Area')
plt.legend(title="Apartment Complex")


plt.figure(figsize=(10, 6))
for i in range(num):
    if (i in excluded_apart):
        continue
    data=np.array(step1_datas[i])
    data=data.T
    Y=data[0] # Price
    area=data[1]
    floor=data[3]
    transaction_t=data[3]
    plt.scatter(floor,Y,label=f'Complex {i}',alpha=0.2)
plt.xlabel('Floor')
plt.ylabel('Price')
plt.legend(title="Apartment Complex")


# plt.figure(figsize=(10, 6))
# for i in range(num):
#     if (i in excluded_apart):
#         continue
#     data=np.array(step1_datas[i])
#     data=data.T
#     Y=data[0] # Price
#     area=data[1]
#     floor=data[3]
#     transaction_t=data[3]
#     plt.scatter(floor,Y,label=f'Complex {i}',alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend(title="Apartment Complex")

plt.show()