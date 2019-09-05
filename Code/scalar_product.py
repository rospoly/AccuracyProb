import numpy as np
import matplotlib.pyplot as plt
from error_model import ErrorModel

class ScalarProduct:

    def __init__(self, X,Y):
        #Test if X,Y are scalars or arrays
        if np.isscalar(X):
            self.X=[]
            self.X.append(X)
        else:
            self.X=X
        if np.isscalar(Y):
            self.Y=[]
            self.Y.append(Y)
        else:
            self.Y=Y
        # Test if X,Y have the same dimension
        if len(self.X)!=len(self.Y):
            raise Exception('The input vectors must have the same dimensions')
        self.dimension=len(self.X)

    def get_pushforward(self):
        #Compute pushforward of input distributions
        self.pushforward=self.X[0]*self.Y[0]
        for i in range(1,self.dimension):
            self.pushforward+=(self.X[i]*self.Y[i])

    def plot_pushforward(self,strFile):
    # Quick and dirty plotting function
        x=np.linspace(-10,10,201)
        y=self.pushforward.pdf(x)
        plt.plot(x, y)
        plt.savefig(strFile)
        plt.clf()

    def get_errorPushforward(self, precision, minexp, maxexp, poly_precision):
        #Compute pushforward through noisy arithmetic
        eps=2**-precision
        Q1=self.X[0]*self.Y[0]
        E1=ErrorModel(Q1,precision, minexp, maxexp, poly_precision)
        Q1=Q1*(1+eps*E1.distribution)

        for i in range(1,self.dimension):
            print(i)
            Q2=self.X[i]*self.Y[i]
            print(Q2.summary())
            E2=ErrorModel(Q2,precision, minexp, maxexp, poly_precision)
            Q2=Q2*(1+eps*E2.distribution)
            Q1+=Q2
            print(Q1.summary())
            E1=ErrorModel(Q1,precision, minexp, maxexp, poly_precision)
            Q1=Q1*(1+eps*E1.distribution)

        self.errorPushforward=Q1

    def plot_all(self,strFile):
        x=np.linspace(-1,1,201)
        y=self.pushforward.pdf(x)
        plt.plot(x, y, 'r')
        z=self.errorPushforward.pdf(x)
        plt.plot(x, z, 'b')
        plt.savefig(strfile)
        plt.clf()
