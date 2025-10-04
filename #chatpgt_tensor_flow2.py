class SmallNet(tf.keras.Model):
    def __init__(self, widths, out_dim=1, activation="tanh"):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(w, activation=activation) for w in widths]
        self.out = tf.keras.layers.Dense(out_dim)

    def call(self, x):
        z = x
        for l in self.hidden:
            z = l(z)
        return self.out(z)

class ThreeSubnets(tf.keras.Model):
    def __init__(self, lb, ub, cfg1, cfg2, cfg3):
        super().__init__()
        self.lb = tf.convert_to_tensor(lb, tf.float32)
        self.ub = tf.convert_to_tensor(ub, tf.float32)
        self.net1 = SmallNet(**cfg1)
        self.net2 = SmallNet(**cfg2)
        self.net3 = SmallNet(**cfg3)

    def call(self, X1, X2=None, X3=None, training=False):
        # Allow different inputs per net; default to X1 if others are None
        X2 = X1 if X2 is None else X2
        X3 = X1 if X3 is None else X3

        X1 = tf.clip_by_value(X1, self.lb, self.ub)
        X2 = tf.clip_by_value(X2, self.lb, self.ub)
        X3 = tf.clip_by_value(X3, self.lb, self.ub)

        return {
            "y1": self.net1(X1),
            "y2": self.net2(X2),
            "y3": self.net3(X3),
        }

# Example usage
cfg = dict(widths=[20,20,20], out_dim=1, activation="tanh")
model_sep = ThreeSubnets(lb=[0,0], ub=[1,1], cfg1=cfg, cfg2=cfg, cfg3=cfg)
opt = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step_separate(X, targets):
    with tf.GradientTape() as tape:
        out = model_sep(X, training=True)
        loss = (
            tf.reduce_mean((out["y1"] - targets["y1"])**2) +
            tf.reduce_mean((out["y2"] - targets["y2"])**2) +
            tf.reduce_mean((out["y3"] - targets["y3"])**2)
        )
    grads = tape.gradient(loss, model_sep.trainable_variables)
    opt.apply_gradients(zip(grads, model_sep.trainable_variables))
    return loss
