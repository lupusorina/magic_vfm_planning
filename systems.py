from dataclasses import dataclass

import numpy as np
import math

@dataclass
class TwoDimDoubleIntegratorNominal:
    xdim: int = 4
    xnames = ["x", "y", "x_dot", "y_dot"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-30.0, 30.0],
                                         [-30.0, 30.0]])

    def __init__(self, padding=5):
        self.time_since_last_flip_x = 0.0
        self.time_since_last_flip_y = 0.0
        self.PAD = padding

    def turn_around_edges(self, x, u_x_dir, u_y_dir, mapshape):
        """
        Enforces bouncing off boundaries of the simulation.
        """

        if not (x[0] >= self.PAD and x[0] <= mapshape[1] - self.PAD):
            if self.time_since_last_flip_x > self.PAD:
                u_x_dir *= -1
                x[1] *= -1
                self.time_since_last_flip_x = 0

        if not (x[1] >= self.PAD and x[1] <= mapshape[0] - self.PAD):
            if self.time_since_last_flip_y > self.PAD:
                u_y_dir *= -1
                x[3] *= -1
                self.time_since_last_flip_y = 0

        self.time_since_last_flip_x += 1
        self.time_since_last_flip_y += 1

        return x, u_x_dir, u_y_dir

    def dynamics(self, state, action, dt, feature_map=None, NN=None):
        u1, u2 = action
        state, u1, u2 = self.turn_around_edges(state, u1, u2, feature_map.shape)
        x, x_dot, y, y_dot = state

        if NN is not None:
            Phi_NN = NN[0]
            a_vec = NN[1]
            dim_a = a_vec.shape[0]
            vels = np.array([x_dot, y_dot])
            xf = np.concatenate((vels, feature_map[int(y + 0.5), int(x + 0.5), :]))
            out_nn = Phi_NN.forward(xf)

            out_nn = out_nn.reshape((2, dim_a))
            Phi_times_a = out_nn @ a_vec # [2 x 1]
            assert Phi_times_a.shape == (2,)
            # clamp Phi
            for i in range(2):
                Phi_times_a[i] = np.clip(Phi_times_a[i], -10.0, 10.0)
            x_dot = x_dot + dt * u1 + dt * Phi_times_a[0]
            y_dot = y_dot + dt * u2 + dt * Phi_times_a[1]

        else:
            x_dot = x_dot + dt * u1
            y_dot = y_dot + dt * u2

        x = x + dt * x_dot
        y = y + dt * y_dot

        next_state = np.array([x, x_dot, y, y_dot])

        return next_state

    def reward(self, state, desired_state, control_input):
        xy = np.array([state[0], state[2]])
        des_xy = np.array([desired_state[0], desired_state[2]])
        r = -np.linalg.norm(xy - des_xy) - 0.1 * np.linalg.norm(control_input)
        return r


@dataclass
class TwoDimDoubleIntegratorTrue:
    xdim: int = 4
    xnames = ["x", "x_dot", "y", "y_dot"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-30.0, 30.0],
                                         [-30.0, 30.0]])

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

        return next_state

    def reward(self, state, desired_state, control_input):
        xy = np.array([state[0], state[2]])
        des_xy = np.array([desired_state[0], desired_state[2]])
        # penalize u
        r = -np.linalg.norm(xy - des_xy) - 0.1 * np.linalg.norm(control_input)
        return r


system_lookup = dict(
    two_dim_double_integrator_damping=TwoDimDoubleIntegratorTrue,
    two_dim_double_integrator_nominal=TwoDimDoubleIntegratorNominal,
)