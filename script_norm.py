from statistics import mean,stdev,median
import csv
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
# from sklearn.covariance import EmpericalCovariance, MinCovDet
def cumulative_sum(arr):
    cum_sum=[]
    cur_sum=0
    for i in arr:
        cur_sum+=i
        cum_sum.append(cur_sum)
    return cum_sum

def moving_average(arr,window_size):
    l=len(arr)
    ans=[]
    for i in range(l - window_size + 1):
        ans.append(mean(arr[i:i+window_size]))
    return ans

# index of lowest 5
# min of list 5 times deleting values from m.copy()
# m.index(min())

def remove_outliers(gamma, all_data):
    robust_cov = MinCovDet.fit(all_data)
    m = robust_cov.mahalanobis(all_data)
    for i in range(int(0.05)*len(all_data)):
        # minimum.append(min(m))
        del all_data[m.index(min(m))]


with open('attack.csv', 'r') as f:
    reader = list(csv.reader(f))
    reader.pop(0)
    malicious_ids=[]
    for i in reader:
        malicious_ids.append(i[1])

positives=[]
negatives=[]
all_data =[]
with open('message.csv', 'r') as f:
    reader = list(csv.reader(f))
    for row in range(1,len(reader)):
        entry=reader[row]
        time_stamp=entry[1][:-7]
        entry_type=entry[2]
        order_id=entry[3]
        price=float(entry[4])
        volume=float(entry[5])
        direction=entry[6]
        trader_id=entry[7]
        stock_id=entry[8]
        order_level=entry[9]
        matched_order_trader_id=entry[10]
        match_price=entry[11]
        match_volume=entry[12]
        match_timestamp=entry[13]
        # print(time_stamp,direction,trader_id)
        all_data.append([price,volume])
        if int(direction)==-1 and int(entry_type) == 1:
            price*=-1
          
        if order_id in malicious_ids:
            negatives.append([price, volume])
        else:
            positives.append([price, volume])
# print(positives,negatives)
## Preprocessing
positives_mat = np.matrix(positives)
negatives_mat = np.matrix(negatives)
all_data_mat = np.matrix(all_data)

positives_mat = positives_mat - positives_mat.mean(axis =0)
negatives_mat = negatives_mat - negatives_mat.mean(axis =0)
all_data_mat = all_data_mat - all_data_mat.mean(axis=0)

positives_mat = positives_mat/positives_mat.std(axis =0)
positives2 = positives_mat.tolist()

negatives_mat = negatives_mat/negatives_mat.std(axis =0)
negatives2 = negatives_mat.tolist()

all_data_mat = all_data_mat/all_data_mat.std(axis =0)
all_data = all_data_mat.tolist()

## Robust PCA
## Remove outliers
## calculate distance of all 
## gamma value of 0.005

  
indices_positives=list(range(len(positives)))
indices_negatives=list(range(len(negatives)))
shuffle(indices_negatives)
shuffle(indices_positives)

train_len_positive=int(len(positives)*0.8)
test_len_positive=len(positives)-train_len_positive
train_len_negative=int(len(negatives)*0.8)
test_len_negative=len(negatives)-train_len_negative

train_set=[]
pred_train_set=[]
test_set=[]
pred_test_set=[]

for i in range(train_len_positive):
    train_set.append(positives2[indices_positives[i]])
    pred_train_set.append(0)
for i in range(train_len_positive,len(positives)):
    test_set.append(positives2[indices_positives[i]])
    pred_test_set.append(0)

for i in range(train_len_negative):
    train_set.append(negatives2[indices_negatives[i]])
    pred_train_set.append(1)
for i in range(train_len_negative,len(negatives)):
    test_set.append(negatives2[indices_negatives[i]])
    pred_test_set.append(1)
# print(len(train_set),len(test_set))
import numpy as np
train_set=np.array(train_set)
test_set=np.array(test_set)

from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
# from pyod.models.mcd import MCD

clf1=PCA(standardization = True,contamination=0.2)
# clf1 = MCD(assume_centered = True)
clf2=OCSVM(kernel = 'poly',nu = 0.25,degree =2,contamination =0.2)
# clf2 = OCSVM(kernel = 'linear',nu =0.02)
clf1.fit(train_set)
clf2.fit(train_set)

y_pred_train_pca=clf1.predict(train_set)
y_pred_test_pca=clf1.predict(test_set)

y_pred_train_ocsvm=clf2.predict(train_set)
y_pred_test_ocsvm=clf2.predict(test_set)
print(clf1.explained_variance_)
# print(y_pred_test_pca,y_pred_test_ocsvm)
train_pca_correct=0
train_ocsvm_correct=0
print("TRAIN SET")
for i in range(len(pred_train_set)):
    # print("Actual:",pred_train_set[i],"PCA",y_pred_train_pca[i],"OCSVM",y_pred_train_ocsvm[i])
    if pred_train_set[i]==y_pred_train_pca[i] and pred_train_set[i]==1:
        train_pca_correct+=1
    if pred_train_set[i]==y_pred_train_ocsvm[i] and y_pred_train_ocsvm[i]==1:
        train_ocsvm_correct+=1

test_pca_correct=0
test_ocsvm_correct=0
print("TEST SET")
for i in range(len(pred_test_set)):
    # print("Actual:",pred_test_set[i],"PCA",y_pred_test_pca[i],"OCSVM",y_pred_test_ocsvm[i])
    if(pred_test_set[i]==y_pred_test_pca[i] and y_pred_test_pca[i]==1):
        test_pca_correct+=1
    if(pred_test_set[i]==y_pred_test_ocsvm[i] and y_pred_test_ocsvm[i]==1):
        test_ocsvm_correct+=1
print(train_len_negative,train_pca_correct,train_ocsvm_correct,test_len_negative,test_pca_correct,test_ocsvm_correct)
print('PCA train accuracy: '+str(train_pca_correct/train_len_negative*100))
print('PCA test accuracy: '+str(test_pca_correct/test_len_negative*100))
print('OCSVM train accuracy: '+str(train_ocsvm_correct/train_len_negative*100))
print('OCSVM test accuracy: '+str(test_ocsvm_correct/test_len_negative*100))
fig,ax1 = plt.subplots(1,1)
positives2 = np.asarray(positives2)
negatives2 = np.asarray(negatives2)
for i in range(len(positives2)):
    positives2[i] = positives2[i]**2
for i in range(len(negatives2)):
    negatives2[i] = negatives2[i]**2    
ax1.scatter(positives2[:,0],positives2[:,1],c='r')
ax1.scatter(negatives2[:,0],negatives2[:,1],c = 'b')
plt.show()