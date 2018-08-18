import numpy as np


class RobustNMF:



    def initialize_WH(self,X_shape,k):

        W = np.random.random((X_shape[0],k))
        H = np.random.random((k,X_shape[1]))

        return W,H


    def __init__(self,X,k,lamda,maxiter):

        self.X = X
        self.X_shape = X.shape
        self.k
        self.lamda = lamda
        self.maxiter = maxiter

        self.W , self.H = self.initialize_WH(X.shape,k)
        self.S = X - np.dot(W,H)


    def update_S(self,S,X_shape,lamda):

        for i in range(X_shape[0]):
            for j in range(X_shape[1]):

                if S[i,j] > (lamda/2):
                    S[i,j] -= (lamda/2)
                elif S[i,j] < -(lamda/2):
                    S[i,j] += (lamda/2)
                else:
                    S[i,j] = 0.0
    
        return S

    def update_W(self,X,W,H,S):

        numerator = np.abs(np.dot(S-X,H.T)) - np.dot(S-X,H.T)
        denominator = 2 * np.dot(np.dot(W,H),H.T)

        return (numerator/denominator)*W


    def rnmf(self):

        X = self.X
        W = self.W
        H = self.H

        maxiter = self.maxiter
        lamda = self.lamda


        for iter in range(maxiter):

            S = X - np.dot(W,H)
            S = self.update_S(S,X.shape,lamda)


            W = self.update_W(X,W,H,S)



        
