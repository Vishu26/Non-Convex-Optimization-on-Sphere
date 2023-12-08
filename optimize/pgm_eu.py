import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

random.seed(42)
np.random.seed(42)
def grad(x):
    grad = (2*a.dot(x)) # - 2*b)
    return grad

a = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        a[i,j] = min(i+1, j+1) / 50


b = np.random.uniform(size=(50,))
xinit = np.concatenate((np.array([1]),np.zeros((49,))))

from scipy.optimize import LinearConstraint,NonlinearConstraint

def cons_f(x):
    return np.linalg.norm(x)
def jac(x):
    return [2*x]
def cons_H(x, v):
    return v[0]*2*np.eye(50)

nonlinear = NonlinearConstraint(cons_f, 1, 1, jac=jac, hess=cons_H)

def cost(x):
    return (100*x.dot(a)).dot(x) - 20*b.T.dot(x)

def func3(x,xt,lr=0.0003):
    return 1/2 * np.linalg.norm(x-xt)**2 - lr*2*b.dot(x) #grad.T.dot(x) + 0.5 / lr * np.linalg.norm(x-xt)

ll = np.array([0])
ul = np.array([1])
linear_constraint = LinearConstraint(np.eye(1),ll,ul)

def line_search(gamma, x, grad):
    return cost(x - gamma*grad)

grad_list = [np.inf]
lr = 0.007
x = [np.inf, xinit]
i=0
norms = [np.linalg.norm((x[-1]))]

while np.linalg.norm(grad(x[-1]))>1e-6:
    grad_list.append(grad(x[-1]))
    #print(grad_list)
    #lr = minimize(line_search, x0=lr_in, args=(x[-1],grad_list[-1]), constraints=linear_constraint).x
    zt = x[-1] - lr*grad_list[-1]
    x.append(minimize(func3, x0=x[-1], args=(zt,lr),constraints=nonlinear).x)
    i+=1
    #lr*=1/(i+1)
    norms.append(np.linalg.norm(x[-1]))
    if i%100==0:
        print(i)
    if i==2000:
        break

val = []
for i in range(len(x[1:])):
    coords = 10*x[i+1]
    val.append((coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords))
print(f"norm = {norms[-1]}")

plt.plot(val)
np.save('eu_pgm.npy',val)
np.save('norms_eu_pgm.npy',norms)
plt.savefig("eu_pgm.png")
plt.figure()
plt.plot(norms)
plt.savefig('norms_eu_pgm.png')