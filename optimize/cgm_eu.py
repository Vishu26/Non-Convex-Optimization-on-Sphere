import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

def grad(x):
    grad = (2*a.dot(x) - 2*b)
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

def func3(x,grad):
    return grad.T.dot(x)# + 0.5 / lr * np.linalg.norm(x-xt)

ll = np.array([0])
ul = np.array([1])
linear_constraint = LinearConstraint(np.eye(1),ll,ul)

def line_search(gamma, x, st, grad):
    return cost(x + gamma*(st-x))


grad_list = [np.inf]
lr = 0.001
x = [np.inf, xinit]
i=0
norms = [np.linalg.norm((x[-1]))]
while np.linalg.norm(grad(x[-1]))>1e-6:
    grad_list.append(grad(x[-1]))
    #print(grad_list)
    st = minimize(func3, x0=x[-1], args=(grad_list[-1]),constraints=nonlinear).x
    lr = minimize(line_search, x0=0.000001, args=(x[-1],st,grad_list[-1]), constraints=linear_constraint).x
    x.append(x[-1] + lr*(st-x[-1]))
    norms.append(np.linalg.norm((x[-1])))
    i+=1
    #lr = 2 / (i + 2)
    if i%100==0:
        print(i)
        print((100*x[-1].T.dot(a)).dot(x[-1]) - 20 * b.T.dot(x[-1]))
        print(np.linalg.norm(x[-1]))
    if i==100:
        break

val = []
for i in range(len(x[1:])):
    coords = 10*x[i+1]
    val.append((coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords))

plt.plot(val)
plt.savefig("cgm_eu.png")
np.save('cgm_eu.npy',val)
np.save('cgm_eu_norm.npy',norms)
np.save('cgm_eu_norms.npy',norms)
plt.figure()
plt.plot(norms)
plt.ylim([-2, 2])
plt.savefig("norms_eu_cgm.png")