import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
import csv
import tqdm

# data = pd.read_csv('./Research_data.csv')
hanriver = pd.read_csv('./hanriver.csv')
apartment_num={}
apartment_name=[]

def get_identifier_from_line(line):
    identifier=line[0][6:]+' / '+line[4]
    return identifier

num=0
hanriver_vars=[]
with open("./hanriver.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        line=line[0].split(',')

        if(i==0):
            print(line)
            continue
        identifier=get_identifier_from_line(line)

        if not identifier in apartment_num: 
            apartment_num[identifier]=num
            hanriver_vars.append({})
            apartment_name.append(identifier)
            num+=1
        
        building_num=line[5]
        if building_num in hanriver_vars[apartment_num[identifier]]:
            # print(f"Duplicate entry! {identifier} - {building_num}")
            continue
        hanriver_vars[apartment_num[identifier]][building_num]=line[6:19]

# For debugging purpose
# for i in range(num):
#     print(f"New apartment: {apartment_name[i]}")
#     for k in hanriver_vars[i]:
#         print(f"{k}: {hanriver_vars[i][k]}")


step1_datas=[[] for i in range(num)] # Step1 regression data (Datas without han-river view)
step2_datas=[[] for i in range(num)] # Step2 regression data (Datas with han-river view)
step2_datas_buildnum=[[] for i in range(num)]

with open("./data.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        line=line[0].split(',')[1:]
        
        if(i==0):
            print(line)
            continue

        identifier=get_identifier_from_line(line)
        if not (identifier in apartment_num):
            continue
        
        building_num=line[9]
        area=eval(line[5])
        transaction_t=(int(line[6][:4])-2023)*12+int(line[6][4:6])-1
        price=int(line[8])
        floor=int(line[10])
        t_build=int(line[13])
        
        if  building_num in hanriver_vars[apartment_num[identifier]]:
            step2_datas[apartment_num[identifier]].append([price,area,transaction_t,floor,t_build])
            step2_datas_buildnum[apartment_num[identifier]].append(building_num)
        else:
            step1_datas[apartment_num[identifier]].append([price,area,transaction_t,floor,t_build])

ts1=0
ts2=0
# Step 1 Regression
excluded_apart=[]
total_step2_datas=[]

print(f"Step 1 OLS")
for i in range(num):
    print(f"For apartment {i}: Step1 {len(step1_datas[i])}, Step2 {len(step2_datas[i])}")
    ts1+=len(step1_datas[i])
    
    data=np.array(step1_datas[i])
    data=data.T
    Y=data[0] # Price
    area=data[1]
    floor=data[3]
    transaction_t=data[3]
    transaction_t2=data[3]**2    
    if(area.min()==area.max() or floor.min()==floor.max() or len(Y)<6):
        print(f"EXCLUDED")
        print(f"==================================================================")
        excluded_apart.append(i)
        continue
    
    ts2+=len(step2_datas[i])

    
    # X=area # Only area
    X=np.column_stack((area,floor))
    # X=np.column_stack((area,floor,transaction_t,transaction_t2))
    X=sm.add_constant(X)
    print(f"X.shape: {X.shape}")

    ols_model = sm.OLS(Y, X)
    results = ols_model.fit()
    predicted_price = results.predict(X)

    print(f"results.params : ",end='')
    for item in results.params:
        print(f"{item:.2f}",end=' ')
    print()
    print(f"results.rsquared : {results.rsquared:.4f}, results.rsquared_adj : {results.rsquared_adj:.4f}")
    # print(f"results.ssr: {results.ssr:.4f}, Average ssr: {results.ssr/len(step1_datas[i]):.4f}, std: {np.sqrt(results.ssr/len(step1_datas[i])):.4f}")
    avg_error=0
    for j in range(len(step1_datas[i])):
        apart=step1_datas[i][j]
        avg_error+=abs(predicted_price[j]-apart[0])/apart[0]
    
    print(f"avg_error: {100*avg_error/len(step1_datas[i]):.2f}")

    hanriver_data=np.array(step2_datas[i])
    hanriver_data=hanriver_data.T
    hanriver_price=hanriver_data[0]
    hanriver_area=hanriver_data[1]
    hanriver_floor=hanriver_data[3]
    hanriver_transaction_t=hanriver_data[3]
    hanriver_transaction_t2=hanriver_data[3]**2
    hanriver_t_build=hanriver_data[4]

    constant=np.ones(len(hanriver_price))
    # X=np.column_stack((constant, hanriver_area))
    X=np.column_stack((constant, hanriver_area,hanriver_floor))
    # X=np.column_stack((constant, hanriver_area,hanriver_floor,hanriver_transaction_t,hanriver_transaction_t2))
    baseline_price=results.predict(X)
     
    for j in range(len(step2_datas[i])):
        current_data=list(step2_datas[i][j])
        current_data=current_data[0:1]+[baseline_price[j]]+current_data[1:]
        building_num=step2_datas_buildnum[i][j]
        var_b=[eval(temp) for temp in hanriver_vars[i][building_num]]
        current_data=current_data+var_b
        # print(f"current_data; {current_data}")
        total_step2_datas.append(current_data)

    # For intuition
    avgdiff=0
    for j in range(len(baseline_price)):
        # print(f"Price: {hanriver_price[j]}, Baseline price: {baseline_price[j]:.2f}")
        avgdiff+=hanriver_price[j]-baseline_price[j]
    print(f"hanpremium: {avgdiff/len(baseline_price):.2f}")
    print(f"==================================================================")

# Price, Baseline_price, area, transaction_t, floor, t_build, hangang_angle, distance, obscure_angle, sound_barrier, highway, north_region

print()
print(f"ts1:{ts1}, ts2:{ts2}") # ts1:972, ts2:282 (Parkrio included), ts1:757, ts2:218 (excluded)

total_step2_datas=np.array(total_step2_datas)
print(f"total_step2_datas.shape: {total_step2_datas.shape}")
np.save("./step2_data",total_step2_datas)
print()

for i in range(num):
    print(f"{i}th apartment name; {apartment_name[i]}, # step1: {len(step1_datas[i])}, # step2: {len(step2_datas[i])}")


plots=11
plote=20

plt.figure(figsize=(10, 6))
for i in range(plots,plote):
    if (i in excluded_apart):
        continue
    data=np.array(step2_datas[i])
    data=data.T
    Y=data[0] # Price
    area=data[1]
    floor=data[3]
    transaction_t=data[3]
    plt.scatter(area,Y,label=f'Complex {i}',alpha=0.7)
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend(title="Apartment Complex")

plt.figure(figsize=(10, 6))
for i in range(plots,plote):
    if (i in excluded_apart):
        continue
    data=np.array(step2_datas[i])
    data=data.T
    Y=data[0] # Price
    area=data[1]
    floor=data[3]
    transaction_t=data[3]
    plt.scatter(floor,Y,label=f'Complex {i}',alpha=0.7)
plt.xlabel('Floor')
plt.ylabel('Price')
plt.legend(title="Apartment Complex")


plt.figure(figsize=(10, 6))
for i in range(plots,plote):
    if (i in excluded_apart):
        continue
    data=np.array(step2_datas[i])
    data=data.T
    Y=data[0] # Price
    area=data[1]
    floor=data[3]
    transaction_t=data[3]
    plt.scatter(floor,Y,label=f'Complex {i}',alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(title="Apartment Complex")

plt.show()