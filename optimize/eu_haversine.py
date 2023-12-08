import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

random.seed(42)
np.random.seed(42)

def get_sines(x, i, n):
    if i > n:
        return 1
    else:
        return np.multiply(np.sin(x[i]), get_sines(x, i + 1, n))

def euclidean_distance_polar(x, y):
    # Euclidean Distance in N dimensional polar coordinates
    dist = 0
    dist += x[0]**2 * (get_sines(x[1:], 0, len(x[1:]) - 1) - get_sines(y[1:], 0, len(y[1:]) - 1)) ** 2
    for i in range(len(x[1:])-1):
        dist += x[0]**2 * (np.cos(x[1:][i]) * get_sines(x, i + 1, len(x[1:]) - 1) - np.cos(y[1:][i]) * get_sines(y[1:], i + 1, len(y[1:]) - 1)) ** 2
    dist += x[0]**2 * (np.cos(x[1:][len(x[1:]) - 1]) - np.cos(y[1:][len(y[1:]) - 1])) ** 2
    return 1 / 2 * dist

def coords_to_polar(x):
    polar = np.zeros(len(x))
    polar[0] = np.linalg.norm(x)
    polar[1] = np.arctan2(x[0], x[1])
    for i in range(2, len(x)):
        polar[i] = np.arctan2(np.linalg.norm(x[:i]), x[i])
    return polar

def polar_to_coords(x):
    coords = np.zeros(len(x))
    coords[0] = x[0] * (get_sines(x[1:], 0, len(x[1:]) - 1))
    for i in range(len(x[1:])-1):
        coords[i + 1] = x[0] * (np.cos(x[1:][i]) * get_sines(x, i + 1, len(x[1:]) - 1))
    coords[-1] = x[0] * (np.cos(x[1:][len(x[1:]) - 1]))
    return coords

def jacobian(X):
    r = X[0]
    x = X[1:]
    sines = 1
    for i in range(1, len(x)):
        sines *= np.sin(x[i]) ** i
    return ((1) ** len(x)) * (r ** len(x)) * sines

def grad(X):
    coords = polar_to_coords(X)
    grad = (2*a.dot(coords) - 2*b) * jacobian(X)
    return grad

a = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        a[i,j] = min(i+1, j+1) / 50

def cost(x):
    coords = polar_to_coords(x)
    coords = coords / np.linalg.norm(coords)
    return (coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords)

from scipy.optimize import LinearConstraint,NonlinearConstraint

ll = np.array([0])
ul = np.array([1])
linear_constraint_lr = LinearConstraint(np.eye(1),ll,ul)

def line_search(gamma, x, grad):
    return cost(x - gamma*grad)


b = np.random.uniform(size=(50,))
xinit = np.concatenate((np.array([1., np.pi/2]), np.ones((48,))*np.pi/2))#coords_to_polar(np.concatenate((np.array([1]),np.zeros((49,)))))#np.concatenate((np.array([1., np.pi/2]), np.ones((48,))*np.pi/2))

ll = np.concatenate((np.array([1]),np.zeros((49,))))
ul = np.concatenate((np.array([1., 2*np.pi]), np.ones((48,))*np.pi))
linear_constraint = LinearConstraint(np.eye(50),ll,ul)

def func(x,xt,grad,lr=0.00001):
    coords = polar_to_coords(x)
    coords_t = polar_to_coords(xt)
    return grad.T.dot(coords) + 1 / lr * (np.linalg.norm(np.sin(x[1:]/4))**2-np.linalg.norm(np.sin(xt[1:]/4))**2-(1/4)*(np.sin(xt[1:]/2).dot(x[1:]-xt[1:])))

grad_list = [np.inf]
lr = 0.00001
x = [np.inf, xinit]
i=0
norms = [np.linalg.norm(polar_to_coords(x[-1]))]
while np.linalg.norm(grad(x[-1]))>1e-6:
    grad_list.append(grad(x[-1]))
    #print(grad_list)
    #lr = minimize(line_search, x0=lr_in, args=(x[-1],grad_list[-1]), constraints=linear_constraint_lr).x
    x.append(minimize(func, x0=x[-1], args=(x[-1],grad_list[-1],lr),constraints=linear_constraint).x)
    coords = 10*polar_to_coords(x[-1])
    norms.append(np.linalg.norm(coords/10))
    print((coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords))
    print(norms[-1])
    #print(np.linalg.norm(grad(x[-1])))
    #lr_in*=1/np.sqrt(i+1)
    i+=1
    if i%10==0:
        print(i)
        #lr_in*=0.1
    if i==20:
        break
#print(norms[-1])

val = []
for i in range(len(x[1:])):
    coords = 10*polar_to_coords(x[i+1])
    val.append((coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords))

plt.plot(val)
np.save('eu_haversine.npy',val)
np.save('norms_haversine.npy', norms)
plt.savefig("eu_haversine.png")