import simple_tests
import numpy as np
import matplotlib.pyplot as plt
from error_model import ErrorModel
from stats import plot_error
from scalar_product import ScalarProduct
import pacal
import time


def test_scalar_products():
    X=[]
    Y=[]
    for i in range(1,4):
        X.append(pacal.UniformDistr(-1,1))
        Y.append(pacal.NormalDistr())
    SP=ScalarProduct(X,Y)
    SP.get_pushforward()
    #SP.plot_pushforward('pics/pushfwd')
    SP.get_errorPushforward(10,-15,16,32)
    SP.plot_all('pics/pushfwd')

def test_error_model(distribution):
    error=ErrorModel(distribution,10,-15,16,32)
    error.plot('pics/test0')


def test_plot_error(distribution):
    plot_error(distribution,10,100000)


def test_simple_tests():
    test1=simple_tests.TestUniformVariable(0,1,0.25,10)
    test1.plot_against_precision(4,32)
    test1.precision=10
    test1.plot_against_threshold()

def test_operations():
    prec=10
    emin=-15
    emax=16
    poly_prec=64
    eps=2**-prec
    X=pacal.BetaDistr(1,10)
    Y=pacal.NormalDistr(0,0.2)
    Z=pacal.BetaDistr(1,10)
    U=X*Y
    Uerr=ErrorModel(U, prec, emin, emax, poly_prec)
    print('error(U) error:  '+repr(Uerr.distribution.int_error()))
    strFile='pics/test1'
    #strFile ='pics/TH_'+repr(U.getName()).replace("'",'')+'_'+repr(prec)
    Uerr.plot(strFile)
    Ucor=U*(1+eps*Uerr.distribution)
    strFile='pics/test2'
    V=U/Z
    Verr=ErrorModel(V, prec, emin, emax, poly_prec)
    print('error(V) error:  '+repr(Verr.distribution.int_error()))
    Verr.plot(strFile)
    Vcor=V*(1+eps*Uerr.distribution)





#main:
start = time.time()
#test_scalar_products()
dist1 = pacal.UniformDistr(-10 ,10)
dist2=pacal.UniformDistr(0,1)
dist3=pacal.NormalDistr()
dist4=pacal.NormalDistr(2,10)
error1=ErrorModel(dist1,10,-15,16,32)
error2=ErrorModel(dist2,10,-15,16,32)
error3=ErrorModel(dist3,10,-15,16,32)
error4=ErrorModel(dist4,10,-15,16,32)
x=np.linspace(-1,1,201)
y1=error1.pdf(x)
y2=error2.pdf(x)
y3=error3.pdf(x)
y4=error4.pdf(x)
plt.subplot(2,2,1)
plt.plot(x,y1)
plt.xlabel('U(-10,10)')
plt.subplot(2,2,2)
plt.plot(x,y2)
plt.xlabel('U(0,1)')
plt.subplot(2,2,3)
plt.plot(x,y3)
plt.xlabel('N(0,1)')
plt.subplot(2,2,4)
plt.plot(x,y4)
plt.xlabel('N(2,10)')
plt.subplots_adjust(hspace=0.3)
plt.savefig("pics/several_examples")
#test_error_model(U)
#test_plot_error(dist)
#test_operations()
end = time.time()
print('Elapsed time:'+repr(end - start)+'s')




#test1=simple_tests.TestUniformVariable(0,2,0.75,10)
#test1.compute()
#test2=simple_tests.TestSumUniformVariable(0,1,0,1,0.75,10)
#test2.compute()
#print(test1.error_prob)
#print(test2.error_prob)
#test1.plot_against_precision(4,32)
#test2.plot_against_precision(4,32)
