import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['PROPDE10802', 'PROPDE10840', 'PROPDE10880', 'PROPDE11200','PROPDE16200','PROPDE16301',
	'PROPDE16401','PROPDE16501','PROPDE16601','PROPDE16701','PROPDE16801','PROPDE16901','PROPDE17001',
	'PROPDE17101','PROPDE17201','PROPDE17301','PROPDE18301','PROPDE18351','PROPDE18501','PROPDE18551',
	'PROPDE18701','PROPDE18901','PROPDE19101','PROPDE19301','PROPDE19501','PROPDE19701','PROPDE19901',
	'PROPDE20101','PROPDE20301','PROPDE20501','PROPDE20701','PROPDE20901','PROPDE21101','PROPDE21301',
	'PROPDE21501','PROPDE21701','PROPDE21901','PROPDE22101','PROPDE22301','PROPDE22501','PROSSO10200',
	'PROSSO10400','CD_PDE_CLUTCH','PROPDE10651','PROPDE10820','PROPDE12500','PROPDE12700','PROPDE13680',
	'PROPDE31100', 'PROPDE31200', 'CD_PSC1330', 'Target']

# dataX = pd.read_csv("C:\\Users\\sehgals\\Documents\\Cathy's Projects\\new navy scriptsKEEP\\Random Forest rerun\\normalized_train_file.csv", names=labels[:-1])
# test_dataX = pd.read_csv("C:\\Users\\sehgals\\Documents\\Cathy's Projects\\new navy scriptsKEEP\\Random Forest rerun\\pseudonormalized_test_file.csv", names=labels[:-1])
# dataY = pd.read_csv("C:\\Users\\sehgals\\Documents\\Cathy's Projects\\new navy scriptsKEEP\\Random Forest rerun\\Y_train_file.csv", names=[labels[-1]])
# test_dataY = pd.read_csv("C:\\Users\\sehgals\\Documents\\Cathy's Projects\\new navy scriptsKEEP\\Random Forest rerun\\Y_test_file.csv", names=[labels[-1]])
# data = pd.concat([
# 	pd.concat([dataX, dataY], axis=1), 
# 	pd.concat([test_dataX, test_dataY], axis=1)], 
# 	ignore_index=True, axis=0)

data = pd.read_csv("R:\\SMM-Structures\\A1-010391 (Navy IPMS data analytics)\\Technical\\Data\\datafiles\\ship datasets\\New folder\\normalized_combined_data.csv", names = labels)

print("Sum of null values in each feature:\n")
print(f"{data.isnull().sum()}")

#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(40,40))

#plot heat map
ax=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="YlGnBu",linewidths=.1)
fig = plt.gcf()
plt.savefig('Correlation_heatmap3.png')
plt.show()