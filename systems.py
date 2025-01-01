from dataclasses import dataclass

import numpy as np
import math

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi

# Parameters robot
C_f = 10.0 # N/rad
C_r = 10.0 # N/rad
C_y = 0.5*(C_f + C_r)
C = C_y
m = 11.8 # kg
b = 2 * C_y/m
k = 1
L = 0.49
Lf = L/2
Lr = L/2
Iz = 0.6484 # kg m^2
tau = 0.1 # s


@dataclass
class TwoDimDoubleIntegratorNominal:
    xdim: int = 4
    xnames = ["x", "y", "x_dot", "y_dot"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-100.0, 100.0],
                                         [-100.0, 100.0]])

    def dynamics(self, state, action, dt, feature_map=None, NN=None):
        x, x_dot, y, y_dot = state
        u1, u2 = action

        if NN is not None:
            Phi_NN = NN[0]
            a_vec = NN[1]
            dim_a = a_vec.shape[0]
            vels = np.array([x_dot, y_dot])
            xf = np.concatenate((vels, feature_map[int(y + 0.5), int(x + 0.5), :]))
            out_nn = Phi_NN.forward(xf)

            out_nn = out_nn.reshape((2, dim_a))
            Phi_times_a = out_nn @ a_vec # 2 x 1
            assert Phi_times_a.shape == (2,)

            x_dot = x_dot + dt * u1 + dt * Phi_times_a[0]
            y_dot = y_dot + dt * u2 + dt * Phi_times_a[1]

        else:
            x_dot = x_dot + dt * u1
            y_dot = y_dot + dt * u2

        x = x + dt * x_dot
        y = y + dt * y_dot

        next_state = np.array([x, x_dot, y, y_dot])

        # TODO: verify the Jacobians
        Jx = np.eye(4) + dt * np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])

        Ju = dt * np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],
        ])

        return next_state, Jx, Ju

    def reward(self, state, desired_state):
        xy = np.array([state[0], state[2]])
        des_xy = np.array([desired_state[0], desired_state[2]])
        r = -np.linalg.norm(xy - des_xy)
        return r


@dataclass
class TwoDimDoubleIntegratorDamping:
    xdim: int = 4
    xnames = ["x", "x_dot", "y", "y_dot"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-100.0, 100.0],
                                         [-100.0, 100.0]])

    def dynamics(self, state, action, dt, damping_map):
        x, x_dot, y, y_dot = state

        u1, u2 = action

        b1 = damping_map[int(y + 0.5), int(x + 0.5)]
        b2 = damping_map[int(y + 0.5), int(x + 0.5)]
        x_dot = x_dot + dt * u1 - dt * x_dot * b1
        y_dot = y_dot + dt * u2 - dt * y_dot * b2

        x = x + dt * x_dot
        y = y + dt * y_dot

        next_state = np.array([x, x_dot, y, y_dot])

        # TODO: verify the Jacobians
        Jx = np.eye(4) + dt * np.array([
            [0, 1, 0, 0],
            [0, -b1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -b2],
        ])

        Ju = dt * np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],
        ])

        return next_state, Jx, Ju

    def reward(self, state, desired_state):
        xy = np.array([state[0], state[2]])
        des_xy = np.array([desired_state[0], desired_state[2]])
        r = -np.linalg.norm(xy - des_xy)
        return r


@dataclass
class TwoDimSingleIntegratorDamping:
    xdim: int = 2
    xnames = ["x", "y"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-100.0, 100.0],
                                         [-100.0, 100.0]])

    def dynamics(self, state, action, dt, damping_map):
        x, y = state

        u1, u2 = action
        b1 = damping_map[int(y + 0.5), int(x + 0.5)]
        b2 = damping_map[int(y + 0.5), int(x + 0.5)]

        x = x + dt * b1 * u1
        y = y + dt * b2 * u2
        next_state = np.array([x, y])

        # TODO: verify the Jacobians
        Jx = np.eye(2) + dt * np.zeros((2, 2))

        Ju = dt * np.array([
            [1.0*b1, 0],
            [0, 1.0*b2],
        ])

        return next_state, Jx, Ju


@dataclass
class TwoDimSingleIntegratorNominal:
    xdim: int = 2
    xnames = ["x", "y"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-100.0, 100.0],
                                         [-100.0, 100.0]])

    def dynamics(self, state, action, dt):
        x, y = state

        u1, u2 = action

        x = x + dt * u1
        y = y + dt * u2
        next_state = np.array([x, y])

        # TODO: verify the Jacobians
        Jx = np.eye(2) + dt * np.zeros((2, 2))

        Ju = dt * np.array([
            [1.0, 0],
            [0, 1.0],
        ])

        return next_state, Jx, Ju

    def reward(self, state, desired_state):
        xy = np.array([state[0], state[1]])
        des_xy = np.array([desired_state[0], desired_state[1]])
        r = -np.linalg.norm(xy - des_xy)
        return r


system_lookup = dict(
    two_dim_double_integrator_damping=TwoDimDoubleIntegratorDamping,
    two_dim_double_integrator_nominal=TwoDimDoubleIntegratorNominal,
    two_dim_single_integrator_damping=TwoDimSingleIntegratorDamping,
    two_dim_single_integrator_nominal=TwoDimSingleIntegratorNominal,
)