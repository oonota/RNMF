import numpy as np
import RNMF

N = 10
X = np.random.randint(1,10,(N,N))
k = 5
lamda = 2
maxiter = 10000

rnmf = RNMF.RobustNMF(X,k,lamda,maxiter)
rnmf.rnmf()
rmse = rnmf.rmse()
print(rmse)



