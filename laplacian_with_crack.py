# This code implements a Physics-Informed Neural Network (PINN) to solve a 2D linear elasticity problem using TensorFlow and Keras.
# Important feature of this version: 


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import time
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# np.random.seed(1234)
# tf.random.set_seed(1234)

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, layers, lb, ub):
        super(PhysicsInformedNN, self).__init__()
        self.lb = lb
        self.ub = ub
        
        # build dense layers
        self.hidden = []
        for width in layers[:-1]:
            self.hidden.append(tf.keras.layers.Dense(width, activation="tanh"))
        self.out = tf.keras.layers.Dense(layers[-1])

        #custom trainable parameters
        self.left_end = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='left_end')
        self.right_end = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='right_end')


    def call(self, X):
        Z = X
        for layer in self.hidden:
            Z = layer(Z)
        return self.out(Z)
    
#initialize neural network
lb = tf.constant([0.0, 0.0], dtype=tf.float32)  # lower bounds
ub = tf.constant([1.0, 1.0], dtype=tf.float32)  # upper bounds
layers = [20, 20, 20, 1]  # hidden sizes and output dim

pinn = PhysicsInformedNN(layers, lb, ub)

# Training step
optimizer = tf.keras.optimizers.Adam()

@ tf.function
def pde_derivs_2d(X):
    with tf.GradientTape(persistent = True) as t2:
        t2.watch(X)
        with tf.GradientTape(persistent = True) as t1:
            t1.watch(X)
            u = pinn(X)
        u_x = t1.gradient(u, X)[:, 0:1]
        u_y = t1.gradient(u, X)[:, 1:2]
    u_xx = t2.gradient(u_x,X)[:, 0:1]
    u_yy = t2.gradient(u_y,X)[:, 1:2]
    del t1, t2
    return u, u_x, u_y, u_xx, u_yy

def f_forcing(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    f = -2.0 * np.pi**2 * tf.sin(np.pi * x) * tf.sin(np.pi * y)
    return f

def true_solution(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    u = tf.sin(np.pi * x) * tf.sin(np.pi * y)
    return u

@tf.function
def residual_poisson(X):
    u, u_x, u_y, u_xx, u_yy = pde_derivs_2d(X)
    f = f_forcing(X)
    r = u_xx + u_yy - f
    return r

def boundary_condition_Dirichlet(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return tf.zeros((tf.shape(X)[0],1), dtype= X.dtype)

def normal_flux_on_boundary(X, normal):
    x = X[:, 0:1]
    y = X[:, 1:2]
    n1 = normal[0]
    n2 = normal[1]
    with tf.GradientTape() as tape:
        tape.watch(X)
        u = pinn(X)
    grad_u = tape.gradient(u, X)
    flux = grad_u[:, 0:1]*n1 + grad_u[:, 1:2]*n2
    return flux 

def true_normal_flux_top_side(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    flux =-np.pi * tf.sin(np.pi * x)
    return flux


def sample_interior(n, lb, ub):
    eps = 1e-6
    X = tf.random.uniform((n, 2), lb, ub, dtype=tf.float32)
    return tf.clip_by_value(X, lb+eps, ub-eps)

def sample_boundary_Dirichlet(n_per_side, lb, ub):
    s = tf.random.uniform((n_per_side, 1), 0.0, 1.0)
    # x in [lb_x, ub_x], y in [lb_y, ub_y]
    x0, x1 = lb[0], ub[0]
    y0, y1 = lb[1], ub[1]

    left   = tf.concat([tf.fill((n_per_side,1), x0), y0 + (y1-y0)*s], axis=1)
    right  = tf.concat([tf.fill((n_per_side,1), x1), y0 + (y1-y0)*s], axis=1)
    bottom = tf.concat([x0 + (x1-x0)*s, tf.fill((n_per_side,1), y0)], axis=1)
    return tf.concat([left, right, bottom], axis=0)

def sample_boundary_Neumann(n_sample_point, lb, ub):
    s = tf.random.uniform((n_sample_point, 1), 0.0, 1.0)
    # x in [lb_x, ub_x], y in [lb_y, ub_y]
    x0, x1 = lb[0], ub[0]
    y0, y1 = lb[1], ub[1]

    top    = tf.concat([x0 + (x1-x0)*s, tf.fill((n_sample_point,1), y1)], axis=1)
    return top

optimizer = tf.keras.optimizers.Adam(1e-3)

Xf = sample_interior(nf, lb, ub)
Xb_Dirichlet = sample_boundary_Dirichlet(nb, lb, ub)
Xb_Neumann = sample_boundary_Neumann(nb, lb, ub)


@tf.function
def train_step_pinn(Xf, Xb_Dirichlet, Xb_Neumann, w_phys=1.0, w_bnd_D=1.0, w_bnd_N=1.0, Xu=None, Yu=None, w_data=0.0):

    with tf.GradientTape() as tape:
        # physics loss (interior)
        r = residual_poisson(Xf)
        loss_phys = tf.reduce_mean(tf.square(r))

        # boundary loss (Dirichlet)
        loss_bnd_Dirichlet = tf.reduce_mean(tf.square(pinn(Xb_Dirichlet) - boundary_condition_Dirichlet(Xb_Dirichlet)))

        # boundary loss (Neumann) 
        normal_vector = [0.0,1.0]
        loss_bnd_Neumann = tf.reduce_mean(tf.square(normal_flux_on_boundary(Xb_Neumann,normal_vector) - true_normal_flux_top_side(Xb_Neumann)))

        # optional supervised/data loss
        if (Xu is not None) and (Yu is not None) and (w_data > 0.0):
            u_d = pinn(Xu, training=True)
            loss_data = tf.reduce_mean(tf.square(u_d - Yu))
        else:
            loss_data = tf.constant(0.0, dtype=tf.float32)

        loss = w_phys*loss_phys + w_bnd_D*loss_bnd_Dirichlet + w_data*loss_data + w_bnd_N*loss_bnd_Neumann

    grads = tape.gradient(loss, pinn.trainable_variables)
    optimizer.apply_gradients(zip(grads, pinn.trainable_variables))
    return loss, loss_phys, loss_bnd_Dirichlet, loss_bnd_Neumann ,loss_data

for epoch in range(20000):
    loss, lp, lbDirichlet, lbNeumann, ld = train_step_pinn()
    if (epoch+1) % 200 == 0:
        print(f"epoch {epoch+1:04d} | total {loss:.3e} | phys {lp:.3e} | bndDir {lbDirichlet:.3e} | bndNeu {lbNeumann:.3e} | data {ld:.3e}")

X_test = np.linspace(0.0, 1.0, 100)
Y_test = np.linspace(0.0, 1.0, 100)
X, Y = np.meshgrid(X_test, Y_test)
X_star = np.column_stack((X.ravel(), Y.ravel())).astype(np.float32)
u_start = true_solution(X_star)
u_pred = pinn(X_star)
u_diff = np.abs(u_start - u_pred)

testing_error = np.linalg.norm(u_start - u_pred,2) / np.linalg.norm(u_start,2)
print(f"Testing Error: {testing_error:.3e}")


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X_star[:,0],X_star[:,1], Y, u_diff, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax.set_title("|u_true - u_pred| surface")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("|diff|")
plt.show()


