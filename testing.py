import findspark
findspark.init()
from random import random
import numpy as np
from pyspark import SparkContext
import ParallelRegression as PR

def rand_vector():
    x1 = np.random.rand(100)
    beta = np.random.rand(100)
    y1 = random()
    return x1, y1, beta

def gradient_test(rddout):
    d = 0.0001 # this is delta 
    x1, y1, beta = rand_vector()
    gradvalue = PR.estimateGrad(lambda beta: PR.f(x1,y1,beta),beta, d)
    localgradvalue = PR.localGradient(x1, y1, beta)
    print("Local Gradient = ", localgradvalue)
    print("Estimated Gradient = ", gradvalue)
    for i in range(len(gradvalue)):
        sub= abs(localgradvalue[i] - gradvalue[i])
        print("Difference = ", sub)
        assert sub < d, f"Assert failed"

def gradient_file_test(rddout):
     d = 0.0001 # this is delta 
     beta = np.random.rand(9)
     lam = random() * 5.0 + 0.1
     gradientactualvalue = PR.gradient(rddout, beta, lam)
     gradientcalculatedvalue = PR.estimateGrad(lambda beta: PR.F(rddout, beta, lam), beta, 0.0000001)
     print("Actual Gradient = ", gradientactualvalue)
     print("Estimated Gradient = ", gradientcalculatedvalue)
     for i in range(len(gradientactualvalue)):
        sub = abs(gradientcalculatedvalue[i] - gradientactualvalue[i])
        print("Difference = ", sub)
        assert sub < d, f"Assert failed"

def test_hcoeff(rddout):
    beta1=np.random.rand(9)
    beta2=np.random.rand(9)
    gamma = np.random.randn()
    tol = 1e-1
    lamda = 0.01
    a, b, c = PR.hcoeff(rddout,beta1,beta2,lamda)
    f1 = PR.F(rddout, beta1 + gamma*beta2, lamda)
    f2 = a*((gamma)**2)+ b*(gamma) + c
    print("f1",f1)
    print("f2",f2)
    assert abs(f1 - f2) < tol ,f"Assert failed"

if __name__ == "__main__":
    sc = SparkContext(appName='test code')
    sc.setLogLevel('warn')
    input="data/small.test"
    rddout=PR.readData(input,sc)
    print(rddout.take(10))
    #gradient_test(rddout)
    #print("All tests passed 1!")
    #gradient_file_test(rddout)
    #print("All tests passed 2!")
    test_hcoeff(rddout)
    print("All the test passed 3!")

