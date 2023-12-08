import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def polar_to_coords(x):
    coords = np.zeros(len(x))
    coords[0] = x[0] * (get_sines(x[1:], 0, len(x[1:]) - 1))
    for i in range(len(x[1:])-1):
        coords[i + 1] = x[0] * (np.cos(x[1:][i]) * get_sines(x, i + 1, len(x[1:]) - 1))
    coords[-1] = x[0] * (np.cos(x[1:][len(x[1:]) - 1]))
    return coords

def coords_to_polar(x):
    polar = np.zeros(len(x))
    polar[0] = np.linalg.norm(x)
    polar[1] = np.arctan2(x[0], x[1])
    for i in range(2, len(x)):
        polar[i] = np.arctan2(np.linalg.norm(x[:i]), x[i])
    return polar

def jacobian(X):
    r = X[0]
    x = X[1:]
    sines = 1
    for i in range(1, len(x)):
        sines *= np.sin(x[i]) ** i
    return (-1 ** len(x)) * (r ** len(x)) * sines

def grad(X):
    coords = polar_to_coords(X)
    grad = (2*a.dot(coords) - 2*b) * jacobian(X)
    return grad

a = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        a[i,j] = min(i+1, j+1) / 50


b = np.random.uniform(size=(50,))
xinit = coords_to_polar(np.concatenate((np.array([1]),np.zeros((49,))))) #np.concatenate((np.array([1., np.pi/2]), np.ones((48,))*np.pi/2))

def cost(x):
    coords = polar_to_coords(x)
    coords = coords / np.linalg.norm(coords)
    return (coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords)

def line_search(gamma, x, st, grad):
    return cost(x + gamma*(st-x))

from scipy.optimize import LinearConstraint,NonlinearConstraint

ll = np.concatenate((np.array([1]),np.zeros((49,))))
ul = np.concatenate((np.array([1., 2*np.pi]), np.ones((48,))*np.pi))
linear_constraint = LinearConstraint(np.eye(50),ll,ul)

ll = np.array([0])
ul = np.array([1])
linear_constraint_lr = LinearConstraint(np.eye(1),ll,ul)

def func(x,grad):
    coords = polar_to_coords(x)
    return grad.T.dot(coords)# + 0.5 / lr * np.linalg.norm(coords-coords_t)

grad_list = [np.inf]
lr_in = 0.0003
x = [np.inf, xinit]
i=0
norm = [np.linalg.norm(polar_to_coords(x[-1]))]
while np.linalg.norm(grad(x[-1]))>1e-6:
    grad_list.append(grad(x[-1]))
    st = minimize(func, x0=x[-1], args=(grad_list[-1]),constraints=linear_constraint).x
    lr = minimize(line_search, x0=lr_in, args=(x[-1],st,grad_list[-1]), constraints=linear_constraint_lr).x
    x.append(x[-1] + lr*(st-x[-1]))
    coords = 10*polar_to_coords(x[-1])
    print((coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords))
    norm.append(np.linalg.norm(polar_to_coords(x[-1])))
    #print(np.linalg.norm(grad(x[-1])))
    i+=1
    if i%10==0:
        print(i)
        lr_in*=0.1
    if i==20:
        break

val = []
for i in range(len(x[1:])):
    coords = 10*polar_to_coords(x[i+1])
    val.append((coords.T.dot(a)).dot(coords) - 2 * b.T.dot(coords))

plt.plot(val)
np.save('cgm_polar.npy',val)
plt.savefig("cgm_polar.png")

plt.figure()
plt.plot(norm)
np.save('cgm_polar_norm.npy',norm)
plt.savefig("cgm_polar_norm.png")