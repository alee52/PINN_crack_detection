# This code implements a Physics-Informed Neural Network (PINN) to solve a 2D linear elasticity problem using TensorFlow and Keras.
# Important feature of this version: 
# 1)residue points are inside the training loop, so they are re-sampled at each iteration.
# 2) There is no crack yet in this verision
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

        # #custom trainable parameters
        # self.left_end = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='left_end')
        # self.right_end = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='right_end')

    def call(self, X):
        Z = X
        for layer in self.hidden:
            Z = layer(Z)
        return self.out(Z)
    
#initialize neural network
lb = tf.constant([0.0, 0.0], dtype=tf.float32)  # lower bounds
ub = tf.constant([1.0, 1.0], dtype=tf.float32)  # upper bounds
layers = [20, 20, 20, 2]  # hidden sizes and output dim

pinn = PhysicsInformedNN(layers, lb, ub)

# Training step
optimizer = tf.keras.optimizers.Adam()


@tf.function
def divCGradU(X, lambda_=0.7, mu=0.5):
    # X: (N,2)
    # returns: u (N,2), du_dx (N,2,2), sigma components (N,1) and div_sigma (N,2)
    with tf.GradientTape(persistent=True) as g:
        g.watch(X)
        u = pinn(X)                 # (N,2)
        # jacobian: shape (N, out_dim, in_dim) -> (N,2,2)
        du = g.jacobian(u, X)       # du[i,j] = d(u_i)/d(x_j)
    # strains
    eps_xx = tf.expand_dims(du[:,0,0], -1)   # (N,1)
    eps_xy = 0.5*(du[:,0,1] + du[:,1,0])    # (N,)  shear
    eps_xy = tf.expand_dims(eps_xy, -1)
    eps_yy = tf.expand_dims(du[:,1,1], -1)

    trace = eps_xx + eps_yy                # (N,1)
    # stress components (plane strain / plane stress note: here classical linear elasticity, using 2D constitutive)
    sigma_xx = lambda_*trace + 2.0*mu*eps_xx  # (N,1)
    sigma_yy = lambda_*trace + 2.0*mu*eps_yy  # (N,1)
    sigma_xy = 2.0*mu*eps_xy                  # (N,1)

    # compute divergence: div_x = d(sigma_xx)/dx + d(sigma_xy)/dy
    # and div_y = d(sigma_xy)/dx + d(sigma_yy)/dy
    with tf.GradientTape(persistent=True) as g2:
        g2.watch(X)
        # re-evaluate sigma as tape2 needs the tensor that depends on X
        # To ensure g2 can compute gradients we rebuild them inside g2:
        u2 = pinn(X)
        du2 = g2.jacobian(u2, X)  # shape (N,2,2)
        eps_xx2 = tf.expand_dims(du2[:,0,0], -1)
        eps_xy2 = tf.expand_dims(0.5*(du2[:,0,1] + du2[:,1,0]), -1)
        eps_yy2 = tf.expand_dims(du2[:,1,1], -1)
        trace2 = eps_xx2 + eps_yy2
        sigma_xx2 = lambda_*trace2 + 2.0*mu*eps_xx2
        sigma_yy2 = lambda_*trace2 + 2.0*mu*eps_yy2
        sigma_xy2 = 2.0*mu*eps_xy2

    dsigma_xx = g2.jacobian(sigma_xx2, X)   # shape (N,1,2)
    dsigma_xy = g2.jacobian(sigma_xy2, X)
    dsigma_yy = g2.jacobian(sigma_yy2, X)

    # dsigma_...[:,0,0] -> d(...)/dx ; dsigma_...[:,0,1] -> d(...)/dy
    div_x = dsigma_xx[:,0,0:1] + dsigma_xy[:,0,1:2]  # (N,1)
    div_y = dsigma_xy[:,0,0:1] + dsigma_yy[:,0,1:2]  # (N,1)

    del g, g2
    return u, du, sigma_xx, sigma_xy, sigma_yy, div_x, div_y

# =============
def true_solution(X):
    x = X[:,0:1]
    y = X[:,1:2]
    # choose smooth vector solution
    u1 = tf.sin(np.pi*x) * tf.sin(np.pi*y)
    u2 = tf.cos(np.pi*x) * tf.sin(np.pi*y)
    return tf.concat([u1, u2], axis=1)


# =============

@tf.function
def forcing(X, lambda_=0.7, mu=0.5):
    # returns f = div(sigma_true) so equilibrium div(sigma) = f holds
    with tf.GradientTape(persistent=True) as g:
        g.watch(X)
        u_true = true_solution(X)
        du = g.jacobian(u_true, X)
    eps_xx = tf.expand_dims(du[:,0,0], -1)
    eps_xy = tf.expand_dims(0.5*(du[:,0,1] + du[:,1,0]), -1)
    eps_yy = tf.expand_dims(du[:,1,1], -1)
    trace = eps_xx + eps_yy
    sigma_xx = lambda_*trace + 2.0*mu*eps_xx
    sigma_yy = lambda_*trace + 2.0*mu*eps_yy
    sigma_xy = 2.0*mu*eps_xy

    # divergence of sigma_true
    with tf.GradientTape(persistent=True) as g2:
        g2.watch(X)
        # recompute to be safe for gradient
        u2 = true_solution(X)
        du2 = g2.jacobian(u2, X)
        eps_xx2 = tf.expand_dims(du2[:,0,0], -1)
        eps_xy2 = tf.expand_dims(0.5*(du2[:,0,1] + du2[:,1,0]), -1)
        eps_yy2 = tf.expand_dims(du2[:,1,1], -1)
        trace2 = eps_xx2 + eps_yy2
        sx = lambda_*trace2 + 2.0*mu*eps_xx2
        sy = lambda_*trace2 + 2.0*mu*eps_yy2
        sxy = 2.0*mu*eps_xy2

    dsx = g2.jacobian(sx, X)
    dsxy = g2.jacobian(sxy, X)
    dsy = g2.jacobian(sy, X)
    div_x = dsx[:,0,0:1] + dsxy[:,0,1:2]
    div_y = dsxy[:,0,0:1] + dsy[:,0,1:2]

    # div(sigma) = f
    f1 = div_x
    f2 = div_y
    del g, g2
    return tf.concat([f1, f2], axis=1)


@tf.function
def residual(X):
    _, _, _, _, _, div_x, div_y = divCGradU(X)
    f = forcing(X)                # (N,2)
    # r1 = div_x - tf.expand_dims(f[:,0], -1)
    # r2 = div_y - tf.expand_dims(f[:,1], -1)
    r1 = div_x - f[0]
    r2 = div_y - f[1]
    return tf.concat([r1, r2], axis=1)  # (N,2)

def boundary_condition_Dirichlet(X):
    # Non-zero Dirichlet data g(x,y):
    return true_solution(X)

# traction (true) on top side from true solution: t = sigma_true · n
@tf.function
def true_traction_top(Xb, lambda_=0.7, mu=0.5):
    # normal on top: n = (0,1)
    n1, n2 = 0.0, 1.0
    # compute sigma from true solution at Xb and produce traction vector
    with tf.GradientTape() as g:
        g.watch(Xb)
        u_t = true_solution(Xb)
        du = g.jacobian(u_t, Xb)
    eps_xx = tf.expand_dims(du[:,0,0], -1); eps_xy = tf.expand_dims(0.5*(du[:,0,1] + du[:,1,0]), -1); eps_yy = tf.expand_dims(du[:,1,1], -1)
    trace = eps_xx + eps_yy
    sx = lambda_*trace + 2.0*mu*eps_xx
    sy = lambda_*trace + 2.0*mu*eps_yy
    sxy = 2.0*mu*eps_xy
    # traction t = [sigma_xx*n1 + sigma_xy*n2, sigma_xy*n1 + sigma_yy*n2]
    t1 = sx * n1 + sxy * n2
    t2 = sxy * n1 + sy * n2
    del g
    return tf.concat([t1, t2], axis=1)

# traction from pinn
@tf.function
def traction_from_pinn(Xb, normal, lambda_=0.7, mu=0.5):
    # build sigma using pinn and compute t = sigma · n
    n1, n2 = normal[0], normal[1]
    with tf.GradientTape(persistent=True) as g:
        g.watch(Xb)
        u_pred = pinn(Xb)
        du = g.jacobian(u_pred, Xb)
    eps_xx = tf.expand_dims(du[:,0,0], -1); eps_xy = tf.expand_dims(0.5*(du[:,0,1] + du[:,1,0]), -1); eps_yy = tf.expand_dims(du[:,1,1], -1)
    trace = eps_xx + eps_yy
    sx = lambda_*trace + 2.0*mu*eps_xx
    sy = lambda_*trace + 2.0*mu*eps_yy
    sxy = 2.0*mu*eps_xy
    t1 = sx * n1 + sxy * n2
    t2 = sxy * n1 + sy * n2
    del g
    return tf.concat([t1, t2], axis=1)


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

# sample top side for traction (Neumann)
def sample_boundary_Neumann(n, lb, ub):
    s = tf.random.uniform((n,1), 0.0, 1.0)
    x0, x1 = lb[0], ub[0]
    y1 = ub[1]
    top = tf.concat([x0 + (x1-x0)*s, tf.fill((n,1), y1)], axis=1)
    return top

optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step_pinn(nf=100, nb=25, w_phys=1.0, w_bnd_D=1.0, w_bnd_N=1.0, Xu=None, Yu=None, w_data=0.0):
    Xf = sample_interior(nf, lb, ub)
    Xb_Dirichlet = sample_boundary_Dirichlet(nb, lb, ub)
    Xb_Neumann = sample_boundary_Neumann(nb, lb, ub)

    print("this is the shape of Xf:", Xf.shape)
    print("this is the shape of Xf:", Xb_Dirichlet.shape)
    print("this is the shape of Xf:", Xb_Neumann.shape)



    with tf.GradientTape() as tape:
        # physics loss (interior)
        r = residual(Xf)
        loss_phys = tf.reduce_mean(tf.reduce_sum(tf.square(r), axis=1))

        # boundary loss (Dirichlet)
        loss_bnd_Dirichlet = tf.reduce_mean(tf.reduce_sum (tf.square(pinn(Xb_Dirichlet) - boundary_condition_Dirichlet(Xb_Dirichlet)), axis = 1))

        # boundary loss (Neumann) 
        normal_vector = [0.0,1.0]
        loss_bnd_Neumann = tf.reduce_mean(tf.reduce_sum(tf.square(traction_from_pinn(Xb_Neumann,normal_vector) - true_traction_top(Xb_Neumann)), axis=1))

        # optional supervised/data loss
        if (Xu is not None) and (Yu is not None) and (w_data > 0.0):
            u_d = pinn(Xu, training=True)
            loss_data = tf.reduce_mean(tf.reduce_sum(tf.square(u_d - Yu), axis=1))
        else:
            loss_data = tf.constant(0.0, dtype=tf.float32)

        loss = w_phys*loss_phys + w_bnd_D*loss_bnd_Dirichlet + w_data*loss_data + w_bnd_N*loss_bnd_Neumann


    grads = tape.gradient(loss, pinn.trainable_variables)
    optimizer.apply_gradients(zip(grads, pinn.trainable_variables))
    return loss, loss_phys, loss_bnd_Dirichlet, loss_bnd_Neumann ,loss_data

for epoch in range(5000):
    loss, lp, lbDirichlet, lbNeumann, ld = train_step_pinn()
    if (epoch+1) % 2 == 0:
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


# ---- evaluation & plotting ----
# grid
nx = ny = 80
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
Xg, Yg = np.meshgrid(x,y)
XY = np.column_stack((Xg.ravel(), Yg.ravel())).astype(np.float32)
XY_tf = tf.convert_to_tensor(XY)

u_true = true_solution(XY_tf).numpy()              # (nx*ny, 2)
u_pred = pinn(XY_tf).numpy()
err = np.linalg.norm(u_true - u_pred, axis=1)
err_grid = err.reshape((ny,nx))

# displacement components
u1_true = u_true[:,0].reshape((ny,nx))
u1_pred = u_pred[:,0].reshape((ny,nx))
u2_true = u_true[:,1].reshape((ny,nx))
u2_pred = u_pred[:,1].reshape((ny,nx))

print("L2 relative error (displacement vector):",
      np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true))

# plot error magnitude
plt.figure(figsize=(6,5))
plt.contourf(Xg, Yg, err_grid, levels=50)
plt.colorbar(label='|u_true - u_pred|')
plt.title('Error magnitude')
plt.xlabel('x'); plt.ylabel('y')
plt.show()

# quick plots of u1_true vs pred along centerline, for debugging
center_idx = ny//2
plt.figure()
plt.plot(x, u1_true[center_idx,:], label='u1_true_line')
plt.plot(x, u1_pred[center_idx,:], '--', label='u1_pred_line')
plt.legend()
plt.show()

