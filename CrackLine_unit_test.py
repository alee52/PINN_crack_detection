

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class CrackLine:
    def __init__(self, left=0.25, right=0.75, up=0.7, down=0.5, radius=0.04):
        #the end points of the crack and its opposite sides

        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.radius = radius

        # center of the circular corners
        self.upLeftCorner = (self.left, self.up - radius)
        self.upRightCorner = (self.right, self.up - radius)
        self.downLeftCorner = (self.left, self.down + radius)
        self.downRightCorner = (self.right, self.down + radius)

    def is_inside_crack(self, x, y):
 
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # central rectangle
        inside_rect = (
            (self.left < x) & (x < self.right) &
            (self.down < y) & (y < self.up)
        )

        # middle band expanded left/right by radius
        inside_mid = (
            (self.left - self.radius < x) & (x < self.right + self.radius) &
            (y > self.down + self.radius) & (y < self.up - self.radius)
        )

        # four rounded corners (quarter-circles), vectorized
        centers = np.array([
            self.downLeftCorner,
            self.downRightCorner,
            self.upLeftCorner,
            self.upRightCorner
        ])

        dx = x[..., None] - centers[:, 0]    
        dy = y[..., None] - centers[:, 1]     
        dist = np.sqrt(dx*dx + dy*dy)         
        inside_corners = (dist < self.radius).any(axis=-1)  
        out = inside_rect | inside_mid | inside_corners
        return out

    def sample_crack(self,n_points_horizontal = 100, n_points_vertical = 25, n_points_corner = 20 ,dtype=tf.float32):
        #sample on horizontal edge (crack)
        edgeUpSample = tf.stack([tf.linspace(self.left, self.right, n_points_horizontal, name=None),
                      tf.fill([n_points_horizontal], tf.cast(self.up, dtype))], axis=1)
        edgeDownSample = tf.stack([tf.linspace(self.left, self.right, n_points_horizontal, name=None),
                      tf.fill([n_points_horizontal], tf.cast(self.down, dtype))], axis=1)
        edgeLeftSample = tf.stack([tf.fill([n_points_vertical], tf.cast(self.left - self.radius, dtype)),
                      tf.linspace(self.down + self.radius,self.up - self.radius,n_points_vertical, name=None)], axis=1)
        edgeRightSample = tf.stack([tf.fill([n_points_vertical], tf.cast(self.right + self.radius, dtype)),
                      tf.linspace(self.down + self.radius,self.up - self.radius,n_points_vertical, name=None)], axis=1)
        #sample on circular corners 
        #upper right corner
        theta = tf.random.uniform([n_points_corner], 0.0, tf.constant(np.pi/2, dtype), dtype=dtype, seed=None)
        cx, cy = self.upRightCorner
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        upRightCornerSample = tf.stack([x, y], axis=1)

        #upper left corner 
        theta = tf.random.uniform([n_points_corner],  tf.constant(np.pi/2, dtype), tf.constant(np.pi, dtype), dtype=dtype, seed=None)
        cx, cy = self.upLeftCorner
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        upLeftCornerSample = tf.stack([x, y], axis=1)

        #lower left corner 
        theta = tf.random.uniform([n_points_corner],  tf.constant(np.pi, dtype), tf.constant(3*np.pi/2, dtype), dtype=dtype, seed=None)
        cx, cy = self.downLeftCorner
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        downLeftCornerSample = tf.stack([x, y], axis=1)

        #lower right corner 
        theta = tf.random.uniform([n_points_corner],  tf.constant(3*np.pi/2, dtype), tf.constant(2*np.pi, dtype), dtype=dtype, seed=None)
        cx, cy = self.downRightCorner
        x = cx + self.radius * tf.cos(theta)
        y = cy + self.radius * tf.sin(theta)
        downRightCornerSample = tf.stack([x, y], axis=1)

    
        return tf.concat([edgeUpSample, edgeDownSample, edgeLeftSample, edgeRightSample, upRightCornerSample,upLeftCornerSample, downLeftCornerSample, downRightCornerSample], axis=0)
    

myCrack = CrackLine()
myCrackPoints = myCrack.sample_crack(50)

pts = myCrackPoints.numpy()          # shape: (2*n_points, 2)
x, y = pts[:, 0], pts[:, 1]

X = np.random.multivariate_normal(mean = [0.5,0.5], cov = [[0.02,0], [0.0, 0.02]], size = 1000)
XisInCrack = myCrack.is_inside_crack(X[:,0], X[:,1])
print("this is my type: ",type(XisInCrack))
mask = (np.asarray(XisInCrack) == 1)


#plot all points together
plt.figure()
plt.scatter(x, y, s=4)

plt.scatter(X[~mask, 0], X[~mask, 1], s=12, c="green", alpha=0.9, label="X (not in crack)")
plt.scatter(X[mask, 0],  X[mask, 1],  s=12, c="red",   alpha=0.9, label="X (in crack)")


plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Crack sample points (both edges)')
plt.grid(True)
plt.show()

