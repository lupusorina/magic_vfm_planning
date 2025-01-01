## Author: John Lathrop

import numpy as np
import matplotlib.pyplot as plt

def clip_vector(vector: np.ndarray, hypercube: np.ndarray) -> np.ndarray:
    return np.clip(vector, hypercube[:, 0], hypercube[:, 1])

class CEM:
    def __init__(self, horizon, num_particles, num_iterations, num_elite, mean, cov, smoothing=0.5, max_std_stop=0.01,  seed=0):
        self.horizon = horizon
        self.num_particles = num_particles
        assert num_iterations >= 1
        self.num_iterations = num_iterations
        self.num_elite = int(num_elite/100.0 * num_particles)
        self.mean = mean # (horizon x action_dim)
        self.cov = cov # (horizon x action_dim x action_dim)
        self.cov_noise = 1e-2
        self.generator = np.random.default_rng(seed)
        self.smoothing = smoothing
        self.max_std_stop = max_std_stop

    def optimize(self, initial_state, system, dt, goal, damping_map=None, terrain_map=None, NN=None):
        action_dim = system.udim
        best_actions = np.zeros((self.num_iterations, self.horizon, action_dim))
        best_action = np.zeros((self.horizon, action_dim))
        best_value = -np.inf
        mean = self.mean
        cov = self.cov
        for iter_nb in range(self.num_iterations):
            print('Iteration', iter_nb)
            particles = np.array([
                [clip_vector(self.generator.multivariate_normal(self.mean[i, :], self.cov[i, :, :]),
                            system.action_lims(),
                            ) for i in range(self.horizon)] # (horizon x action_dim)
                            for _ in range(self.num_particles)
            ]) # (num_particles x horizon x action_dim)
            values = np.empty(self.num_particles)
            for i in range(self.num_particles):
                state = initial_state
                value = 0
                for j in range(self.horizon):
                    action = particles[i, j, :]
                    if damping_map is not None:
                        next_state, _, _ = system.dynamics(state, action, dt, damping_map)
                    else:
                        next_state, _, _ = system.dynamics(state, action, dt)
                        
                    if NN is not None:
                        next_state, _, _ = system.dynamics(state, action, dt, terrain_map, NN)
                        
                    value += system.reward(state, goal)
                    state = next_state.copy()
                values[i] = value
            elite_indices = np.argsort(values)[-self.num_elite:]
            elite_particles = particles[elite_indices, :, :] # (num_elite x horizon x action_dim)
            
            # TODO add smoothing
            mean = np.mean(elite_particles, axis=0)
            cov = self.computecovariance(elite_particles, mean)
        
            if values[elite_indices[-1]] > best_value:
                best_value = values[elite_indices[-1]]
                best_action = elite_particles[-1]
            self.mean = mean
            self.cov = cov + np.array([self.cov_noise * np.eye(action_dim) for _ in range(self.horizon)])
            best_actions[iter_nb] = best_action
            # check cov for convergence.
            stds = np.sqrt(np.linalg.eigvalsh(cov))
            if np.amax(stds) <= self.max_std_stop:
                print("Converged at iteration", iter_nb)
                break
        return best_action, best_actions

    def computecovariance(self, particles, mean):
        # particles is (num_particles x horizon x action_dim)
        # mean is (horizon x action_dim)
        centered_particles = particles - np.tile(mean, (particles.shape[0], 1, 1)) # (num_particles x horizon x action_dim)
        cov = np.array([np.mean([np.outer(particle, particle) for particle in centered_particles[:, i, :]], axis=0) for i in range(particles.shape[1])])
        return cov

