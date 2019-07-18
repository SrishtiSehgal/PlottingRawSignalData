import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

labels = ['PROPDE10802', 'PROPDE10840', 'PROPDE10880', 'PROPDE11200','PROPDE16200','PROPDE16301',
	'PROPDE16401','PROPDE16501','PROPDE16601','PROPDE16701','PROPDE16801','PROPDE16901','PROPDE17001',
	'PROPDE17101','PROPDE17201','PROPDE17301','PROPDE18301','PROPDE18351','PROPDE18501','PROPDE18551',
	'PROPDE18701','PROPDE18901','PROPDE19101','PROPDE19301','PROPDE19501','PROPDE19701','PROPDE19901',
	'PROPDE20101','PROPDE20301','PROPDE20501','PROPDE20701','PROPDE20901','PROPDE21101','PROPDE21301',
	'PROPDE21501','PROPDE21701','PROPDE21901','PROPDE22101','PROPDE22301','PROPDE22501','PROSSO10200',
	'PROSSO10400','CD_PDE_CLUTCH','PROPDE10651','PROPDE10820','PROPDE12500','PROPDE12700','PROPDE13680',
	'PROPDE31100', 'PROPDE31200', 'CD_PSC1330', 'Target']

df = pd.read_csv('R:\\SMM-Structures\\A1-010391 (Navy IPMS data analytics)\\Technical\\Data\\datafiles\\ship datasets\\combination of all ship datafiles\\normalized_combined_data.csv', names=labels)
df['Target'].replace([0,1], ["Healthy", "Failed"], inplace=True)

# df.head()
# df.describe()
# df.info()
#https://github.com/mwaskom/seaborn/issues/1627
# sns.pairplot(df, 
# 	vars=['CD_PDE_CLUTCH','PROPDE10651','PROPDE10820','PROPDE12500','PROPDE12700','PROPDE13680','PROPDE31100', 'PROPDE31200', 'CD_PSC1330'], 
# 	hue='Target', 
# 	palette={'Healthy':'g', 'Failed':'r'},
# 	height=1.5, 
# 	diag_kind='hist')

# fig1 = plt.gcf()
# plt.show()
# quit()
# df['Target'].value_counts()

# df.hist(column='sepal_length_cm',bins=20, figsize=(10,5))
df['Target'].replace(["Healthy", "Failed"], [0,1], inplace=True)

df.head()
inp_df = df.drop(df.columns[[-1]], axis=1)
out_df = df['Target']

# scaler = StandardScaler()
# inp_df = scaler.fit_transform(inp_df)

X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)

X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()

print('Train Shape', (X_tr_arr.shape))
print('Test Shape', X_test.shape)


def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b

def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result

def model_optimize(w, b, X, Y):
    m = X.shape[0]
    
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
   
    return {"dw": dw, "db": db}, cost

def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w,b,X,Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
                
        if (i % 100 == 0):
            costs.append(cost)
            print("Cost after %i iteration is %f" %(i, cost))
    
    return {"w": w, "b": b}, {"dw": dw, "db": db}, costs

def predict(final_pred, m, threshold):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > threshold:
            y_pred[0][i] = 1
    return y_pred

#Get number of features
n_features = X_tr_arr.shape[1]
print('Number of Features', n_features)
# w, b = weightInitialization(n_features)

# #Gradient Descent
# coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=4500)

# #Final prediction
# w = coeff["w"]
# b = coeff["b"]
# print('Optimized weights', w)
# print('Optimized intercept',b)

# final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
# final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)

# m_tr =  X_tr_arr.shape[0]
# m_ts =  X_ts_arr.shape[0]

# y_tr_pred = predict(final_train_pred, m_tr)
# print('Training Accuracy',accuracy_score(y_tr_pred.T, y_tr_arr))

# y_ts_pred = predict(final_test_pred, m_ts)
# print('Test Accuracy',accuracy_score(y_ts_pred.T, y_ts_arr))

# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title('Cost reduction over time')
# plt.show()

clf = LogisticRegression()
clf.fit(X_tr_arr, y_tr_arr)
print(clf.intercept_, clf.coef_)

tra_pred = clf.predict(X_tr_arr)
print("tra", confusion_matrix(y_tr_arr, tra_pred))
pred = clf.predict(X_ts_arr)
print("test", confusion_matrix(y_ts_arr, pred))


print(clf.get_params())
print('Accuracy from sk-learn: {0}'.format(clf.score(X_ts_arr, y_ts_arr)))
print('Accuracy from sk-learn: {0}'.format(clf.score(X_tr_arr, y_tr_arr)))
# class_weight={0:1, 1:100}