# Thermo-electrical (coupled multiphysics) - single-branch S-DeepONet
#
# Code release for:
#   "Single vs. Multiple Branches in DeepONet and S-DeepONet:
#    Network Architecture Follows Coupling in Multiphysics Systems"
#   J. Park, K. Kobayashi, Q. Liu, S. Koric, D. Abueidda, S. B. Alam.
#   arXiv:2507.03660  -  https://arxiv.org/abs/2507.03660
#
# Single-branch (1br) vs. multi-branch / MIONet (2br) S-DeepONet.
# Data paths are relative to the repository root; run scripts from there
# and place the datasets under ./data/ (see data/README.md).
# ---------------------------------------------------------------------------

# %%
import time as TT
import keras.backend as K
from deepxde.data.sampler import BatchSampler
from deepxde.data.data import Data
import deepxde as dde
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from deepxde.backend import tf
tf.config.optimizer.set_jit(True)  # This_line_here
dde.config.disable_xla_jit()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# %%
training_data = np.load(
    'data/thermo_electrical/coupled/[phi(t)]_coupled_Qrhoe(t).npz')
grid_input = training_data['grid_input']
target_T_phi = training_data['target_T_phi']
input_Qext_rhoe = training_data['input_Qext_rhoe']
rhoe = input_Qext_rhoe[:, :, 1]
Qext = input_Qext_rhoe[:, :, 0]
rhoe_shift, rhoe_scale = rhoe.mean(), rhoe.std()
rhoes = (rhoe - rhoe_shift) / rhoe_scale
inputs = np.stack([Qext, rhoe], axis=-1)

n_step = inputs.shape[1]
n_cases = inputs.shape[0]
n_nodes = target_T_phi.shape[1]//n_step
N_input_fn = 2
HIDDEN = 100
m = n_step
N_output_frame = 1  # First, predicting the last frame
N_component = 2  # of predicted solution fileds,  temp and stress

xy_train_testing = grid_input.reshape(n_step, n_nodes, 2)[-1, :, :1]
target_T_phi = target_T_phi.reshape(n_cases, n_step, n_nodes, N_component)
# %%
batch_size = 64
seed = 2024
try:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
except:
    pass
dde.config.set_default_float("float64")


class DeepONetCartesianProd(dde.maps.NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
            layer_sizes_branch: A list of integers as the width of a fully connected network,
                    or `(dim, f)` where `dim` is the input dimension and `f` is a network
                    function. The width of the last layer in the branch and trunk net should be
                    equal.
            layer_sizes_trunk (list): A list of integers as the width of a fully connected
                    network.
            activation: If `activation` is a ``string``, then the same activation is used in
                    both trunk and branch nets. If `activation` is a ``dict``, then the trunk
                    net uses the activation `activation["trunk"]`, and the branch net uses
                    `activation["branch"]`.
    """

    def __init__(
            self,
            layer_sizes_branch,
            layer_sizes_trunk,
            activation,
            kernel_initializer,
            regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(
                activation["trunk"])
        else:
            activation_branch = self.activation_trunk = dde.maps.activations.get(
                activation)

        # User-defined network
        self.branch = layer_sizes_branch[1]
        self.trunk = layer_sizes_trunk[0]
        # self.b = tf.Variable(tf.zeros(1),dtype=np.float64)
        self.b = tf.Variable(tf.zeros(1, dtype=dde.config.real(tf)))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        # print( x_func.shape )
        # print( x_loc.shape )
        # exit()

        # Branch net to encode the input function
        x_func = self.branch(x_func)  # [ bs , HD , N_TS ]
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(
            self.trunk(x_loc))  # [ N_pts , HD , N_comp ]

        # Dot product
        x = tf.einsum("bht,nhc->btnc", x_func, x_loc)

        # Add bias
        x += self.b

        # if self._output_transform is not None:
        #       x = self._output_transform(inputs, x)
        # return tf.math.sigmoid(x) # This_line_different_here_here
        return x


class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
            X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
                    `dim1`), and the second element has the shape (`N2`, `dim2`).
            y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y


# %%
fraction_train = 0.8
N_valid_case = len(inputs)
N_train = int(N_valid_case * fraction_train)
train_case = np.random.choice(N_valid_case, N_train, replace=False)
test_case = np.setdiff1d(np.arange(N_valid_case), train_case)
u0_train = inputs[train_case]
u0_testing = inputs[test_case]
s_train = target_T_phi[train_case, -1:, :, :]
s_testing = target_T_phi[test_case, -1:, :, :]
x_train = (u0_train, xy_train_testing)
y_train = s_train
x_test = (u0_testing, xy_train_testing)
y_test = s_testing
data = TripleCartesianProd(x_train, y_train, x_test, y_test)
my_act1 = "tanh"
branch = tf.keras.models.Sequential([
    tf.keras.layers.GRU(units=256, batch_input_shape=(batch_size, m, N_input_fn),
                        activation=my_act1, return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
    # tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.GRU(units=128, activation=my_act1,
                        return_sequences=False, dropout=0.00, recurrent_dropout=0.00),
    # tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.RepeatVector(HIDDEN),
    tf.keras.layers.GRU(units=128, activation=my_act1,
                        return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
    # tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.GRU(units=256, activation=my_act1,
                        return_sequences=True, dropout=0.00, recurrent_dropout=0.00),
    # tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(N_output_frame))])
branch.summary()

my_act2 = "relu"
trunk = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(xy_train_testing.shape[1],)),
    tf.keras.layers.Dense(101, activation=my_act2,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act2,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act2,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act2,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(101, activation=my_act2,
                          kernel_initializer='GlorotNormal'),
    tf.keras.layers.Dense(
        HIDDEN * N_component, activation=my_act2, kernel_initializer='GlorotNormal'),
    tf.keras.layers.Reshape([HIDDEN, N_component]),
])
trunk.summary()


def COP(y_true, y_pred):
    sqr_err = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
    var_true = y_true.shape[0] * tf.math.reduce_variance(K.flatten(y_true))
    data_loss = tf.math.divide(tf.math.reduce_sum(sqr_err), var_true)
    return data_loss


def err(y_train, y_pred):
    ax = 1
    return np.linalg.norm(y_train - y_pred, axis=ax) / np.linalg.norm(y_train, axis=ax)


def metric1(y_train, y_pred):
    #     y_train_original = scalerS.inverse_transform(y_train[:, 0, :, 1])
    #     y_pred_original = scalerS.inverse_transform(y_pred[:, 0, :, 1])
    y_train_original = y_train[:, 0, :, 1]
    y_pred_original = y_pred[:, 0, :, 1]
    return np.mean(err(y_train_original, y_pred_original).flatten())


def metric_mae_stress(y_train, y_pred):
    # y_train_original = scalerS.inverse_transform(y_train[:,0,:,1])
    # y_pred_original = scalerS.inverse_transform(y_pred[:,0,:,1])
    y_train_original = y_train[:, 0, :, 1]
    y_pred_original = y_pred[:, 0, :, 1]
    tmp = tf.math.abs(K.flatten(y_train_original) - K.flatten(y_pred_original))
    data_loss = tf.math.reduce_mean(tmp)
    return data_loss


# %%
net = DeepONetCartesianProd([m, branch], [trunk], my_act2, "Glorot normal")

model = dde.Model(data, net)
model.compile(
    "adam",
    lr=1e-3,
    decay=("inverse time", 1, 1e-4),
    loss=COP,
    metrics=[metric1, metric_mae_stress],
)

print("y_train shape:", y_train.shape)
print("y_test shape:", model.predict(data.test_x).shape)
#
# %%

losshistory, train_state = model.train(
    iterations=310000, batch_size=batch_size, model_save_path="./Models/model_1br/model")
np.save('./Models/model_1br/losshistory.npy', losshistory)


st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ', duration, ' s')
print('Prediction speed = ', duration / float(len(y_pred)), ' s/case')

y_pred_T = y_pred[:, 0, :, 0]
y_pred_S = y_pred[:, 0, :, 1]
y_true_T = data.test_y[:, 0, :, 0]
y_true_S = data.test_y[:, 0, :, 1]
np.savez('./Models/model_1br/predictions.npz', x_grid=data.test_x[1][:, 0],
         y_pred_T=y_pred_T, y_pred_S=y_pred_S, y_true_T=y_true_T, y_true_S=y_true_S)

# %%
