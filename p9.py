import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
    m,n=np.shape(xmat) #size of matrix m
    weights=np.mat(np.eye(m)) #np.eye returns mat with 1 in the diagonal
    for j in range(m):
        diff=point-xmat[j]
        weights[j,j]=np.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localWeight(point,xmat,ymat,k):
    wei=kernel(point,xmat,k)
    W=(xmat.T*(wei*xmat)).I*(xmat.T*(wei*ymat.T))
    return W

def localWeightRegression(xmat,ymat,k):
    row,col=np.shape(xmat) #return 244 rows and 2 columns
    ypred=np.zeros(row)
    for i in range(row):
        ypred[i]=xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

data=pd.read_csv('p9.csv')
bill=np.array(data.total_bill)
tip=np.array(data.tip)

mbill=np.mat(bill)
mtip=np.mat(tip)

mbillMatCol=np.shape(mbill)[1] 
onesArray=np.mat(np.ones(mbillMatCol))
xmat=np.hstack((onesArray.T,mbill.T)) 
print(xmat)

ypred=localWeightRegression(xmat,mtip,2)
SortIndex=xmat[ :,1].argsort(0) 
xsort=xmat[SortIndex][:,0]

fig= plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(bill,tip,color='blue')
ax.plot(xsort[:,1],ypred[SortIndex],color='red',linewidth=1)
plt.xlabel('Total bill')
plt.ylabel('tip')
plt.show()
