# This code implements a Physics-Informed Neural Network (PINN) to solve a 2D linear elasticity problem using TensorFlow and Keras.
# Important feature of this version: 
# 
# 1) the crack is part of a closed surface and assumed to be known. Only the slip is unknown.
# 2) since the crack is known, we only need to sample the residue points once before training
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import time
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

from mpl_toolkits.mplot3d import Axes3D  
# np.random.seed(1234)
# tf.random.set_seed(1234)

#============================ SET CASE: ZERO SLIPPAGE OR NONZERO SLIPPAGE ============================#
# slippage_case = 0 #slippage is zero
slippage_case = 1 #slippage is non-zero


# ============================ Load MATLAB data ============================#
from scipy.io import loadmat

# path = "/Users/arumlee/Desktop/matlab_code_dislocations 2/Gsub.mat"
if slippage_case == 0:  
    path = "data/Gsub.mat"
else:
    path = "data/Gsub_nonzero.mat"
data = loadmat(path)
Test_data = data['Gsub']   #shape (N,3)
X_test = Test_data[:,0]    
Y_test = Test_data[:,1]
U1_test = Test_data[:,2]  
U2_test = Test_data[:,3]    

# Plot test data points
plt.figure(figsize=(5,5))
plt.scatter(X_test, Y_test, c='b', marker='o', s=1)
plt.title('Test data points')
plt.xlabel('x')
plt.ylabel('y')
plt.show(block = False)

# ---- 3D scatter plots of U1 and U2 ----
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test, Y_test, U1_test, c=U1_test, cmap='coolwarm', s=5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('U1')
ax1.set_title('3D map of Matlab U1(X, Y)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test, Y_test, U2_test, c=U2_test, cmap='coolwarm', s=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('U2')
ax2.set_title('3D map of Matlab U2(X, Y)')

plt.tight_layout()
plt.show(block = False)






#============================Domain description============================#
crack_left_end = [-0.4, 0.0]
crack_right_end = [0.4, 0.0]
domain_bottom_left = [-1.0, -1.0]
domain_top_right = [1.0, 1.0]


# =========================== Crack (closed) geometry and sampling ===========================#
class CrackLine:
    def __init__(self, left = -0.4, right = 0.4, up=0.0, down=-0.3, radius=0.04):
        #the end points of the crack position at (left, up) and (right, up)
        #and its opposite sides at (left, down) and (right, down)

        self.left   = float(left)
        self.right  = float(right)
        self.up     = float(up)
        self.down   = float(down)
        self.radius = float(radius)

        # center of the circular corners
        self.corners_tf = tf.constant([
            [self.left, self.down + radius],   # downLeftCorner
            [self.right, self.down + radius],   # downRightCorner
            [self.left, self.up - radius],     # upLeftCorner
            [self.right, self.up - radius],     # upRightCorner
        ], dtype=tf.float32)

    @tf.function
    def is_inside(self, x, y, overide = None):
        
        #force the neural net to use inside or outside the crack region
        if overide is True:
            return tf.ones_like(x, dtype=tf.bool)
        if overide is False:
            return tf.zeros_like(x, dtype=tf.bool)
        
        #if not forced by overide, use normal checking
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        # central rectangle
        inside_rect = tf.logical_and(
            tf.logical_and(self.left < x, x < self.right),
            tf.logical_and(self.down < y, y < self.up)
        )

        # middle band expanded left/right by radius
        inside_mid = tf.logical_and(
            tf.logical_and(self.left - self.radius < x, x < self.right + self.radius),
            tf.logical_and(self.down + self.radius < y, y < self.up - self.radius)
        )

        # corners_tf: (4,2). Broadcast x,y to (N,1) then subtract
        cx = self.corners_tf[:, 0]               # (4,)
        cy = self.corners_tf[:, 1]               # (4,)
        dx = tf.expand_dims(x, axis=-1) - cx     # (N,4)
        dy = tf.expand_dims(y, axis=-1) - cy     # (N,4)
        dist = tf.sqrt(dx*dx + dy*dy)            # (N,4)
        inside_corners = tf.reduce_any(dist < self.radius, axis=1)  # (N,)

        # union of regions
        inside = tf.logical_or(
            tf.logical_or(inside_rect, inside_mid),
            inside_corners
        )  # (N,)

        return inside

    def sample_crack(self,n_points_horizontal = 100, n_points_vertical = 25, n_points_corner = 20 ,dtype=tf.float32):
        #sample on horizontal upper edge
        edgeUpSample = tf.stack([tf.linspace(self.left, self.right, n_points_horizontal, name=None),
                      tf.fill([n_points_horizontal], tf.cast(self.up, dtype))], axis=1)
        normal_vec = tf.constant([0.0, 1.0], dtype=edgeUpSample.dtype)
        N = tf.shape(edgeUpSample)[0]
        normal_up = tf.tile(normal_vec[None, :], [N, 1])
        edgeUp_with_normals = tf.concat([edgeUpSample, normal_up], axis=1)

        #sample on horizontal lower edge
        edgeDownSample = tf.stack([tf.linspace(self.left, self.right, n_points_horizontal, name=None),
                      tf.fill([n_points_horizontal], tf.cast(self.down, dtype))], axis=1)
        normal_vec_down = tf.constant([0.0, -1.0], dtype=edgeDownSample.dtype)
        normal_down = tf.tile(normal_vec_down[None, :], [tf.shape(edgeDownSample)[0], 1])
        edgeDown_with_normals = tf.concat([edgeDownSample, normal_down], axis=1)
        
        #sample on vertical left edge
        edgeLeftSample = tf.stack([tf.fill([n_points_vertical], tf.cast(self.left - self.radius, dtype)),
                      tf.linspace(self.down + self.radius,self.up - self.radius,n_points_vertical, name=None)], axis=1)
        normal_vec_left = tf.constant([-1.0, 0.0], dtype=edgeLeftSample.dtype)
        normal_left = tf.tile(normal_vec_left[None, :], [tf.shape(edgeLeftSample)[0], 1])
        edgeLeft_with_normals = tf.concat([edgeLeftSample, normal_left], axis=1)


        #sample on vertical right edge
        edgeRightSample = tf.stack([tf.fill([n_points_vertical], tf.cast(self.right + self.radius, dtype)),
                      tf.linspace(self.down + self.radius,self.up - self.radius,n_points_vertical, name=None)], axis=1)
        normal_vec_right = tf.constant([1.0, 0.0], dtype=edgeRightSample.dtype)

        normal_right = tf.tile(normal_vec_right[None, :],
                       [tf.shape(edgeRightSample)[0], 1])
        edgeRight_with_normals = tf.concat([edgeRightSample, normal_right], axis=1)

        #sample on circular corners 
        #upper right corner
        theta = tf.random.uniform([n_points_corner], 0.0, tf.constant(np.pi/2, dtype), dtype=dtype, seed=None)
        cx, cy = tf.unstack(self.corners_tf[3])
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        r1 = tf.cos(theta)
        r2 = tf.sin(theta)
        upRightCornerSample = tf.stack([x, y, r1, r2], axis=1)

        #upper left corner 
        theta = tf.random.uniform([n_points_corner],  tf.constant(np.pi/2, dtype), tf.constant(np.pi, dtype), dtype=dtype, seed=None)
        cx, cy = tf.unstack(self.corners_tf[2])
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        r1 = tf.cos(theta)
        r2 = tf.sin(theta)
        upLeftCornerSample = tf.stack([x, y, r1, r2], axis=1)

        #lower left corner 
        theta = tf.random.uniform([n_points_corner],  tf.constant(np.pi, dtype), tf.constant(3*np.pi/2, dtype), dtype=dtype, seed=None)
        cx, cy = tf.unstack(self.corners_tf[0])
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        r1 = tf.cos(theta)
        r2 = tf.sin(theta)
        downLeftCornerSample = tf.stack([x, y, r1, r2], axis=1)

        #lower right corner 
        theta = tf.random.uniform([n_points_corner],  tf.constant(3*np.pi/2, dtype), tf.constant(2*np.pi, dtype), dtype=dtype, seed=None)
        cx, cy = tf.unstack(self.corners_tf[1])
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        r1 = tf.cos(theta)
        r2 = tf.sin(theta)
        downRightCornerSample = tf.stack([x, y, r1, r2], axis=1)

    
        return tf.concat([edgeUp_with_normals, edgeDown_with_normals, edgeLeft_with_normals, edgeRight_with_normals, upRightCornerSample,upLeftCornerSample, downLeftCornerSample, downRightCornerSample], axis=0)
    
# ===========================Define PINN structure ============================#

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, crack_left_end0, crack_right_end0, domain_bottom_left0, domain_top_right0):
        super(PhysicsInformedNN, self).__init__()
        self.domain_bottom_left0 = domain_bottom_left0
        self.domain_top_right0 = domain_top_right0
        self.crack_left_end0 = crack_left_end0
        self.crack_right_end0 = crack_right_end0
        self.crack_line = CrackLine(left=crack_left_end0[0], right=crack_right_end0[0], up=crack_left_end0[1], down=-0.3, radius=0.04)  

        zero_init = tf.keras.initializers.Zeros()

        self.subnet_in  = tf.keras.Sequential([
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(2),
        ])

        # self.subnet_in  = tf.keras.Sequential([
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(2,
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        # ])

        self.subnet_out = tf.keras.Sequential([
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(30, activation="tanh"),
            tf.keras.layers.Dense(2),
        ])

        # self.subnet_out = tf.keras.Sequential([
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(30, activation="tanh",
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        #     tf.keras.layers.Dense(2,
        #                           kernel_initializer=zero_init,
        #                           bias_initializer=zero_init),
        # ])

        # #custom trainable parameters
        # self.left_end = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='left_end')
        # self.right_end = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='right_end')
    def call(self, X, overide_call = None , training=False):
        x = X[:, 0]
        y = X[:, 1]

        # Boolean mask per example: True if point is inside the crack region
        inside = self.crack_line.is_inside(x, y, overide = overide_call)                 # shape: (batch,) of booleans

        # Evaluate both subnets once
        y_in  = self.subnet_in(X, training=training)             # shape: (batch, out_dim)
        y_out = self.subnet_out(X, training=training)            # shape: (batch, out_dim)

        # Select per row using the mask
        inside = tf.expand_dims(tf.cast(inside, tf.bool), axis=-1)  # (batch, 1)
        Z = tf.where(inside, y_in, y_out)                        # (batch, out_dim)
        return Z

pinn = PhysicsInformedNN(crack_left_end, crack_right_end, domain_bottom_left, domain_top_right)



# ============================================#Boundary data and forcing terms============================================================
#=========================================================================================================================================
@tf.function
def true_solution(X):
    x = X[:,0:1]
    y = X[:,1:2]
    # choose smooth vector solution
    u1 = tf.sin(np.pi*x) * tf.sin(np.pi*y)
    u2 = tf.cos(np.pi*x) * tf.sin(np.pi*y)
    return tf.concat([u1, u2], axis=1)


@tf.function
def forcing(X, lambda_=1, mu=1, homogeneous=True):
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    tf.ensure_shape(X, [None, 2])

    with tf.GradientTape(persistent=True) as g2:
        g2.watch(X)

        # true u and its per-sample Jacobian: (N,2,2)
        u = true_solution(X)
        du = g2.batch_jacobian(u, X)

        # small-strain components, shape (N,1)
        eps_xx = du[:, 0, 0:1]
        eps_yy = du[:, 1, 1:2]
        eps_xy = 0.5 * (du[:, 0, 1:2] + du[:, 1, 0:1])

        trace = eps_xx + eps_yy
        sigma_xx = lambda_ * trace + 2.0 * mu * eps_xx
        sigma_yy = lambda_ * trace + 2.0 * mu * eps_yy
        sigma_xy = 2.0 * mu * eps_xy

        # stack σ = [σxx, σxy, σyy] so one jacobian gets every needed partial
        sigma = tf.concat([sigma_xx, sigma_xy, sigma_yy], axis=1)  # (N,3)

    # J = d sigma / dX has shape (N,3,2): rows map to [σxx, σxy, σyy], cols to [x,y]
    J = g2.batch_jacobian(sigma, X)  # (N,3,2)
    del g2 

    # div σ = [∂σxx/∂x + ∂σxy/∂y,  ∂σxy/∂x + ∂σyy/∂y]
    div_x = J[:, 0, 0:1] + J[:, 1, 1:2]
    div_y = J[:, 1, 0:1] + J[:, 2, 1:2]
    div = tf.concat([div_x, div_y], axis=1)  # (N,2)

    multiplier = 1.0 - tf.cast(homogeneous, tf.float32)  # scalar 0. or 1.
    f = -div * multiplier  # broadcasted to (N,2)

    return f


@tf.function
def residual(X):
    _, _, _, _, _, div_x, div_y = divCGradU(X)
    f = forcing(X)                # (N,2)
    # r1 = div_x - tf.expand_dims(f[:,0], -1)
    # r2 = div_y - tf.expand_dims(f[:,1], -1)
    r1 = div_x - tf.expand_dims(f[:, 0], -1)
    r2 = div_y - tf.expand_dims(f[:, 1], -1)
    out = tf.concat([r1, r2], axis=1)  # (N,2)
    return out

def boundary_condition_Dirichlet(X, homogeneous=True):
    tf.ensure_shape(X, [None, 2])
    
    u_true = true_solution(X)     # (N,2)

    # homogeneous=True  -> u_bc = 0
    # homogeneous=False -> u_bc = true_solution(X)
    multiplier = 1.0 - tf.cast(homogeneous, tf.float32)  # scalar in {0,1}

    return u_true * multiplier   # broadcasts to (N,2)

# traction (true) on top side from true solution: t = sigma_true · n
@tf.function
def true_traction_top(Xb, lambda_=1, mu=1):
    # normal on top: n = (0,1)
    n1, n2 = 0.0, 1.0
    # compute sigma from true solution at Xb and produce traction vector
    with tf.GradientTape() as g:
        g.watch(Xb, persistent=False)
        u_t = true_solution(Xb)
        du = g.batch_jacobian(u_t, Xb)
    eps_xx = tf.expand_dims(du[:,0,0], -1); eps_xy = tf.expand_dims(0.5*(du[:,0,1] + du[:,1,0]), -1); eps_yy = tf.expand_dims(du[:,1,1], -1)
    trace = eps_xx + eps_yy
    sx = lambda_*trace + 2.0*mu*eps_xx
    sy = lambda_*trace + 2.0*mu*eps_yy
    sxy = 2.0*mu*eps_xy
    # traction t = [sigma_xx*n1 + sigma_xy*n2, sigma_xy*n1 + sigma_yy*n2]
    t1 = sx * n1 + sxy * n2
    t2 = sxy * n1 + sy * n2

    return tf.concat([t1, t2], axis=1)

# traction from pinn (used for both top boundary and crack interface)
@tf.function
def traction_from_pinn(Xb, normal, lambda_=1, mu=1, overide_traction=None):


    tf.ensure_shape(Xb, [None,2])
    tf.ensure_shape(normal, [None,2])
    dtype = Xb.dtype
    n1 = normal[:, 0:1]
    n2 = normal[:, 1:2]

    lambda_ = tf.cast(lambda_, dtype)
    mu      = tf.cast(mu, dtype)

    with tf.GradientTape(persistent=False) as g:
        g.watch(Xb)
        u_pred = pinn(Xb, overide_call=overide_traction)  # (N, 2)
        du = g.batch_jacobian(u_pred, Xb)  # (N, 2, 2)

    # strain components
    eps_xx = tf.expand_dims(du[:, 0, 0], -1)
    eps_yy = tf.expand_dims(du[:, 1, 1], -1)
    eps_xy = tf.expand_dims(0.5 * (du[:, 0, 1] + du[:, 1, 0]), -1)

    trace = eps_xx + eps_yy

    # stresses
    sx  = lambda_ * trace + 2.0 * mu * eps_xx
    sy  = lambda_ * trace + 2.0 * mu * eps_yy
    sxy = 2.0 * mu * eps_xy

    # traction t = sigma n
    t1 = sx * n1 + sxy * n2
    t2 = sxy * n1 + sy * n2

    return tf.concat([t1, t2], axis=1)  # (N, 2)

@tf.function
def divCGradU(X, lambda_=1, mu=1):

    # Outer tape for second derivatives wrt X
    with tf.GradientTape(persistent=True) as g2:
        g2.watch(X)
        # Inner tape for first derivatives wrt X
        with tf.GradientTape(persistent=True) as g1:
            g1.watch(X)
            u = pinn(X)             # (N,2)

        # ---- FIRST DERIVATIVES (per-sample) ----
        du = g1.batch_jacobian(u, X)   # (N,2,2)
        eps_xx = du[:, 0, 0:1]                 # (N,1)
        eps_yy = du[:, 1, 1:2]                 # (N,1)
        eps_xy = 0.5*(du[:, 0, 1:2] + du[:, 1, 0:1])  # (N,1)

        trace    = eps_xx + eps_yy
        sigma_xx = lambda_ * trace + 2.0 * mu * eps_xx   # (N,1,1)
        sigma_yy = lambda_ * trace + 2.0 * mu * eps_yy
        sigma_xy = 2.0 * mu * eps_xy

    # ---- SECOND DERIVATIVES (divergence per sample) ----
    dsxx = g2.batch_jacobian(sigma_xx, X)   # (N,1,2)
    dsxy = g2.batch_jacobian(sigma_xy, X)
    dsyy = g2.batch_jacobian(sigma_yy, X)

    div_x = dsxx[:, 0, 0:1] + dsxy[:, 0, 1:2]   # (N,1)
    div_y = dsxy[:, 0, 0:1] + dsyy[:, 0, 1:2]   # (N,1)

    del g1, g2

    return u, du, sigma_xx, sigma_xy, sigma_yy, div_x, div_y


# @tf.function
# def interface_jump_displacement(X, a = 100/39):
#     #true jump condition in u across the crack line
#     #used for forward problem only
#     x = X[:, 0:1]
#     a =tf.cast(a, tf.float32)

#     mask = tf.cast((x >= -39/100) & (x <= 39/100), tf.float32)

#     f1 = -10 * (-(a**12) * x**12 + 1) * ((x >= -39/100) & (x <= 39/100))
#     f2 =  20 * (-(a**12) * x**12 + 1) * ((x >= -39/100) & (x <= 39/100))

#     return tf.concat([f1, f2], axis=1)

@tf.function
def interface_jump_displacement(X, a=100/39):
    # ensure dtype & shape
    X = tf.convert_to_tensor(X)
    tf.ensure_shape(X, [None, 2])
    dtype = X.dtype if X.dtype != tf.string else tf.float32 
    X = tf.cast(X, dtype)

    x = X[:, 0:1]  # (N,1)
    y = X[:, 1:2]

    # make all scalars same dtype
    a = tf.cast(a, dtype)
    left  = tf.cast(-39.0/100.0, dtype)
    right = tf.cast( 39.0/100.0, dtype)
    m10   = tf.cast(-10.0, dtype)
    p20   = tf.cast( 20.0, dtype)
    one   = tf.cast(  1.0, dtype)

    # boolean mask in float dtype
    mask_interval = tf.cast((x >= left) & (x <= right), dtype)  # (N,1)
    mask_y_zero = tf.cast(tf.equal(y, 0.0), dtype)  # (N,1)
    mask = mask_interval * mask_y_zero 


    # poly = 1 - (a^12) * x^12, all tensor ops (no Python-side pow on mixed dtypes)
    poly = one - tf.pow(a, 12) * tf.pow(x, 12)  # (N,1)

    f1 =- m10 * poly * mask  # (N,1)
    f2 =- p20 * poly * mask  # (N,1)

    return tf.concat([f1, f2], axis=1)  # (N,2)



@tf.function
def interface_jump_traction(X, normal):
    # zero jump in traction across crack
    X = tf.convert_to_tensor(X)
    tf.ensure_shape(X, [None, 2])
    return tf.zeros_like(X, dtype=tf.float32)




#===========================================#Sampling points======================================================
#=================================================================================================================


def sample_uniform_points_in_rectangle(n, domain_bottom_left0, domain_top_right0, dtype=tf.float32):
    #Sample n points uniformly inside a rectangle provided bottom left and top right corners
    x_min, y_min = domain_bottom_left0
    x_max, y_max = domain_top_right0

    xs = tf.random.uniform(shape=(n,), minval=x_min, maxval=x_max, dtype=dtype)
    ys = tf.random.uniform(shape=(n,), minval=y_min, maxval=y_max, dtype=dtype)

    return tf.stack([xs, ys], axis=1)

def sample_near_crack_tips(n, crack_left_end0, crack_right_end0, radius=0.1, dtype=tf.float32):
    #Sample n points uniformly near the crack tips (within a circle of given radius)
    angles = tf.random.uniform(shape=(n,), minval=0.0, maxval=2.0*np.pi, dtype=dtype)
    radii = tf.random.normal(shape=(n,), mean=0.0, stddev=radius/3.0, dtype=dtype)
    radii = tf.abs(radii)  # ensure positive radius

    # Half points near left tip, half near right tip
    n_half = n // 2
    xs_left = crack_left_end0[0] + radii[:n_half] * tf.cos(angles[:n_half])
    ys_left = crack_left_end0[1] + radii[:n_half] * tf.sin(angles[:n_half])
    xs_right = crack_right_end0[0] + radii[n_half:] * tf.cos(angles[n_half:])
    ys_right = crack_right_end0[1] + radii[n_half:] * tf.sin(angles[n_half:])

    xs = tf.concat([xs_left, xs_right], axis=0)
    ys = tf.concat([ys_left, ys_right], axis=0)

    return tf.stack([xs, ys], axis=1)

def sample_boundary_Dirichlet(n_per_side, domain_bottom_left0, domain_top_right0):
    s = tf.random.uniform((n_per_side, 1), 0.0, 1.0)
    # x in [lb_x, ub_x], y in [lb_y, ub_y]
    x0, x1 = domain_bottom_left0[0], domain_top_right0[0]
    y0, y1 = domain_bottom_left0[1], domain_top_right0[1]

    # left   = tf.concat([tf.fill((n_per_side,1), x0), y0 + (y1-y0)*s], axis=1)
    # right  = tf.concat([tf.fill((n_per_side,1), x1), y0 + (y1-y0)*s], axis=1)
    bottom = tf.concat([x0 + (x1-x0)*s, tf.fill((n_per_side,1), y0)], axis=1)
    # return tf.concat([left, right, bottom], axis=0)
    return bottom

# sample top side for traction (Neumann)
def sample_boundary_Neumann_top(n, domain_bottom_left0, domain_top_right0):
    s = tf.random.uniform((n,1), 0.0, 1.0)
    x0, x1 = domain_bottom_left0[0], domain_top_right0[0]
    y1 = domain_top_right0[1]
    top = tf.concat([x0 + (x1-x0)*s, tf.fill((n,1), y1)], axis=1)
    return top
# sample left side for traction (Neumann)
def sample_boundary_Neumann_left(n, domain_bottom_left0, domain_top_right0):
    s = tf.random.uniform((n,1), 0.0, 1.0)
    x0 = domain_bottom_left0[0]
    y0, y1 = domain_bottom_left0[1], domain_top_right0[1]
    left = tf.concat([tf.fill((n,1), x0), y0 + (y1 - y0) * s], axis=1)
    return left
# sample right side for traction (Neumann)
def sample_boundary_Neumann_right(n, domain_bottom_left0, domain_top_right0):
    s = tf.random.uniform((n,1), 0.0, 1.0)
    x1 = domain_top_right0[0]
    y0, y1 = domain_bottom_left0[1], domain_top_right0[1]
    right = tf.concat([tf.fill((n,1), x1), y0 + (y1 - y0) * s], axis=1)
    return right


#===========================================#Visualization of sampled boundary points======================================================
# Xb_Top = sample_boundary_Neumann_top(50, domain_bottom_left, domain_top_right).numpy()
# Xb_Left = sample_boundary_Neumann_left(50, domain_bottom_left, domain_top_right).numpy()
# Xb_Bottom = sample_boundary_Dirichlet(50, domain_bottom_left, domain_top_right).numpy()
# Xb_Right = sample_boundary_Neumann_right(50, domain_bottom_left, domain_top_right).numpy()
# Xb_np=np.vstack([Xb_Top, Xb_Left,  Xb_Bottom, Xb_Right])
# # extract x and y
# x = Xb_np[:, 0]
# y = Xb_np[:, 1]

# # plot
# plt.figure(figsize=(6,6))
# plt.scatter(x, y, s=10, c='red', label='Dirichlet boundary points')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Dirichlet Boundary Sample Points")
# plt.legend()
# plt.axis("equal")   # keep aspect ratio correct
# plt.grid(True)
# plt.show()



#====================================================== training (Forward problem) =======================================================
#=========================================================================================================================================
optimizer = tf.keras.optimizers.Adam(1e-3)

n_interior_point=1000
n_boudary_point=500
isHomogeneousJump = 1.0   #1.0 for non-homogeneous jump condition; 0.0 for homogeneous jump condition

@tf.function
def train_step_pinn(w_phys=1.0, w_bnd_D=1.0, w_bnd_N=1.0, w_interf_disp = 2.0, w_interf_tract = 2.0, Xu=None, Yu=None, w_data=0.0):
    Xf = sample_uniform_points_in_rectangle(n_interior_point, domain_bottom_left, domain_top_right)
    Xcrack_tip = sample_near_crack_tips(n_interior_point//5, crack_left_end, crack_right_end, radius=0.1)
    Xf = tf.concat([Xf, Xcrack_tip], axis=0)

    Xb_Dirichlet = sample_boundary_Dirichlet(n_boudary_point//4, domain_bottom_left, domain_top_right)
   
    Xb_Neumann_top_points = sample_boundary_Neumann_top(n_boudary_point//4, domain_bottom_left, domain_top_right)
    Xb_Neumann_top_normal = tf.tile(tf.constant([[0.0,1.0]], dtype=tf.float32), [tf.shape(Xb_Neumann_top_points)[0],1])

    Xb_Neumann_left_points   = sample_boundary_Neumann_left(n_boudary_point//4,   domain_bottom_left, domain_top_right)
    Xb_Neumann_left_normal = tf.tile(tf.constant([[-1.0,0.0]], dtype=tf.float32), [tf.shape(Xb_Neumann_left_points)[0],1])

    Xb_Neumann_right_points  = sample_boundary_Neumann_right(n_boudary_point//4,  domain_bottom_left, domain_top_right)
    Xb_Neumann_right_normal = tf.tile(tf.constant([[1.0,0.0]], dtype=tf.float32), [tf.shape(Xb_Neumann_right_points)[0],1])


    XInterface = pinn.crack_line.sample_crack(n_points_horizontal = 1000, n_points_vertical = 1000, n_points_corner = 100)
    # XInterface_points = XInterface[:, 0:2]
    # XInterface_normals = 

    with tf.GradientTape() as tape:
        # physics loss (interior)
        r = residual(Xf)
        loss_phys = tf.reduce_mean(tf.reduce_sum(tf.square(r), axis=1))

        # boundary loss (Dirichlet)
        loss_bnd_Dirichlet = tf.reduce_mean(tf.reduce_sum (tf.square(pinn(Xb_Dirichlet) - 0*boundary_condition_Dirichlet(Xb_Dirichlet)), axis = 1))

        # boundary loss top (Neumann) 
        loss_bnd_Neumann_top = tf.reduce_mean(tf.reduce_sum(tf.square(traction_from_pinn(Xb_Neumann_top_points,Xb_Neumann_top_normal) - tf.zeros_like(Xb_Neumann_top_points)), axis=1))
        loss_bnd_Neumann_left = tf.reduce_mean(tf.reduce_sum(tf.square(traction_from_pinn(Xb_Neumann_left_points,Xb_Neumann_left_normal) - tf.zeros_like(Xb_Neumann_left_points)), axis=1))
        loss_bnd_Neumann_right = tf.reduce_mean(tf.reduce_sum(tf.square(traction_from_pinn(Xb_Neumann_right_points,Xb_Neumann_right_normal) - tf.zeros_like(Xb_Neumann_right_points)), axis=1))
        loss_bnd_Neumann = (loss_bnd_Neumann_top + loss_bnd_Neumann_left + loss_bnd_Neumann_right)
        # interface loss (crack) - jump in displacement
        loss_interface_displacement = tf.reduce_mean(tf.reduce_sum(tf.square(pinn(XInterface[:, 0:2], overide_call=True) - pinn(XInterface[:, 0:2], overide_call=False) - isHomogeneousJump*interface_jump_displacement(XInterface[:, 0:2])), axis=1))

        #interface loss (crack) - jump in traction
        loss_interface_traction = tf.reduce_mean(tf.reduce_sum(tf.square(traction_from_pinn(XInterface[:, 0:2], XInterface[:, 2:4], overide_traction=True) - traction_from_pinn(XInterface[:, 0:2], XInterface[:, 2:4], overide_traction=False) - interface_jump_traction(XInterface[:, 0:2], XInterface[:, 2:4])), axis=1))
    

        # optional supervised/data loss
        if (Xu is not None) and (Yu is not None) and (w_data > 0.0):
            u_d = pinn(Xu, training=True)
            loss_data = tf.reduce_mean(tf.reduce_sum(tf.square(u_d - Yu), axis=1))
        else:
            loss_data = tf.constant(0.0, dtype=tf.float32)

        loss = w_phys*loss_phys + w_bnd_D*loss_bnd_Dirichlet + w_data*loss_data + w_bnd_N*loss_bnd_Neumann + w_interf_disp*loss_interface_displacement + w_interf_tract*loss_interface_traction


    grads = tape.gradient(loss, pinn.trainable_variables)
    optimizer.apply_gradients(zip(grads, pinn.trainable_variables))
    return loss, loss_phys, loss_bnd_Dirichlet, loss_bnd_Neumann ,loss_interface_displacement, loss_interface_traction



for epoch in range(10):
    loss, lp, lbDirichlet, lbNeumann, lJumpDir, lJumpTrac = train_step_pinn()
    if (epoch+1) % 50 == 0:
        print(f"epoch {epoch+1:04d} | total {loss:.3e} | phys {lp:.3e} | bndDir {lbDirichlet:.3e} | bndNeu {lbNeumann:.3e} | dirJump {lJumpDir:.3e} | tracJump {lJumpTrac:.3e}")
    if loss < 1e-10:
        print("Early stopping at epoch", epoch+1)
        break


# # ============================ Testing =================================#
XY_test_tensor = tf.convert_to_tensor(np.column_stack((X_test, Y_test)).astype(np.float32))
u_pred_test = pinn(XY_test_tensor)
u_pred_test_np = u_pred_test.numpy()
u1_pred_test = u_pred_test_np[:,0]
u2_pred_test = u_pred_test_np[:,1]

fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test, Y_test, u1_pred_test, c=u1_pred_test, cmap='coolwarm', s=5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('U1')
ax1.set_title('3D map of predicted U1')
ax1.view_init(elev=0, azim=90)

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test, Y_test,  u2_pred_test, c=u2_pred_test, cmap='coolwarm', s=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('U2')
ax2.set_title('3D map of predicted U2')
ax2.view_init(elev=0, azim=90)
plt.tight_layout()
plt.show(block = False)


# # Compute testing error
diff_u1 = U1_test - u1_pred_test
diff_u2 = U2_test - u2_pred_test
error_u1 = np.linalg.norm(U1_test - u1_pred_test, 2) / np.linalg.norm(U1_test, 2)
error_u2 = np.linalg.norm(U2_test - u2_pred_test, 2) / np.linalg.norm(U2_test, 2)
print(f"Testing Error in U1: {error_u1:.3e}")
print(f"Testing Error in U2: {error_u2:.3e}")   

# ---- 3D scatter plots of U1 and U2 ----
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test, Y_test, diff_u1, c=diff_u1, cmap='coolwarm', s=5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('U1')
ax1.set_title('3D map of diff U1')
ax1.view_init(elev=0, azim=90)

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test, Y_test, diff_u2, c=diff_u2, cmap='coolwarm', s=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('U2')
ax2.set_title('3D map of diff U2')
ax2.view_init(elev=0, azim=90)
plt.tight_layout()
plt.show()






# X_test = np.linspace(0.0, 1.0, 100)
# Y_test = np.linspace(0.0, 1.0, 100)
# X, Y = np.meshgrid(X_test, Y_test)
# X_star = np.column_stack((X.ravel(), Y.ravel())).astype(np.float32)
# u_start = true_solution(X_star)
# u_pred = pinn(X_star)
# # u_diff = np.abs(u_start - u_pred)

# testing_error = np.linalg.norm(u_start - u_pred,2) / np.linalg.norm(u_start,2)
# print(f"Testing Error: {testing_error:.3e}")


# # ---- evaluation & plotting ----
# # grid
# nx = ny = 80
# x = np.linspace(0,1,nx)
# y = np.linspace(0,1,ny)
# Xg, Yg = np.meshgrid(x,y)
# XY = np.column_stack((Xg.ravel(), Yg.ravel())).astype(np.float32)
# XY_tf = tf.convert_to_tensor(XY)

# u_true = true_solution(XY_tf).numpy()              # (nx*ny, 2)
# u_pred = pinn(XY_tf).numpy()
# err = np.linalg.norm(u_true - u_pred, axis=1)
# err_grid = err.reshape((ny,nx))

# # displacement components
# u1_true = u_true[:,0].reshape((ny,nx))
# u1_pred = u_pred[:,0].reshape((ny,nx))
# u2_true = u_true[:,1].reshape((ny,nx))
# u2_pred = u_pred[:,1].reshape((ny,nx))

# print("L2 relative error (displacement vector):",
#       np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true))

# # plot error magnitude
# plt.figure(figsize=(6,5))
# plt.contourf(Xg, Yg, err_grid, levels=50)
# plt.colorbar(label='|u_true - u_pred|')
# plt.title('Error magnitude')
# plt.xlabel('x'); plt.ylabel('y')
# plt.show()

# # quick plots of u1_true vs pred along centerline, for debugging
# center_idx = ny//2
# plt.figure()
# plt.plot(x, u1_true[center_idx,:], label='u1_true_line')
# plt.plot(x, u1_pred[center_idx,:], '--', label='u1_pred_line')
# plt.legend()
# plt.show()

