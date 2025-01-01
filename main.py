# There is grass everywhere and there are sand pits.
# The agent has to reach the goal position.
# The agent has to avoid the sand pits.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import mlpfile
import mlpfile.torch

import torch.nn as nn

import argparse
import yaml
import sys

# Custom imports.
from systems import TwoDimSingleIntegratorDamping, TwoDimSingleIntegratorNominal
from systems import TwoDimDoubleIntegratorDamping, TwoDimDoubleIntegratorNominal
from cem import CEM
from plotting_utils import *
from dino_nn import Dino

# Set plotting style.
font = {'family' : 'serif',
        'serif' : 'Computer Modern Roman',
        'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


# World dim, divisible by the DINO patch size.
w = 240
h = 120
PAD = 5

# Integration DT.
dt = 0.05

# Controls & State Indices
IDX_UX = 0
IDX_UY = 1
system = 'double_integrator'

if system == 'single_integrator':
    # State.
    IDX_PX = 0
    IDX_PY = 1

if system == 'double_integrator':
    IDX_PX = 0
    IDX_VX = 1
    IDX_PY = 2
    IDX_VY = 3

grass_img_path = 'images/grass.jpg'
sand_img_path = 'images/sand.jpg'
ice_img_path = 'images/ice.jpg'

device = "cpu"

def PhiNN(input_size, hidden_size, output_size):
    layers = [
        nn.utils.spectral_norm(nn.Linear(input_size, hidden_size)),
        nn.ReLU(),
        nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        nn.utils.spectral_norm(nn.Linear(hidden_size, output_size)),
    ]
    return nn.Sequential(*layers)

def update_feat_rgb_and_damping_maps(xy, feature_map, rgb_map, damping_map, damping_scaling,
                                        x_coord_list, y_coord_list, img_dino_map, img_rgb_map_lower_res):
    """Update the feature_map, rgb_map & damping map
    Args:
        xy (array): h x w x 2 meshgrid
        feature_map (array): h x w x n_feats
        rgb_map (array): h x w x 3
        damping_map (array): h x w
        damping_scaling (array): scaling factors for the damping map
        x_coord_list (list): list of coordinates in x for the circle center to put the img_dino_map
        y_coord_list (list): list of coordinates in y for the circle center to put the img_dino_map
        img_dino_map (array): input dino map from an image
        img_rgb_map_lower_res (array): RGB map at a lower resolution (same resolution as the Dino map)
    """
    for xcoord in x_coord_list:
        for ycoord in y_coord_list:
            c = np.linalg.norm(xy - [xcoord, ycoord], axis=-1) < 7
            counter_j_dino = 0
            counter_i_dino = 0
            for i in range(h):
                counter_j_dino = 0
                for j in range(w):
                    if c[i, j] == True:
                        feature_map[i, j, :] = img_dino_map[:, counter_i_dino, counter_j_dino] # TODO: improve fcn.
                        rgb_map[i, j, :] = img_rgb_map_lower_res[counter_i_dino, counter_j_dino, :]
                        damping_map[i, j] = damping_scaling
                        counter_j_dino += 1
                counter_i_dino += 1
                if counter_i_dino == img_dino_map.shape[1]:
                    counter_i_dino = 0

def turn_around_edges(x, u_x_dir, u_y_dir, time_since_last_flip_x, time_since_last_flip_y, mapshape):
    """
    Enforces bouncing off boundaries of the simulation.
    """
    if not (x[IDX_PX] >= PAD and x[IDX_PX] <= mapshape[0] - PAD):
        if time_since_last_flip_x > PAD:
            u_x_dir *= -1
            x[IDX_VX] *= -1
            time_since_last_flip_x = 0

    if not (x[IDX_PY] >= PAD and x[IDX_PY] <= mapshape[1] - PAD):
        if time_since_last_flip_y > PAD:
            u_y_dir *= -1
            x[IDX_VY] *= -1
            time_since_last_flip_y = 0

    time_since_last_flip_x += 1
    time_since_last_flip_y += 1

    return x, u_x_dir, u_y_dir, time_since_last_flip_x, time_since_last_flip_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_folder", type=str, default="plots/")
    parser.add_argument("--dataset_folder", type=str, default="dataset/")
    parser.add_argument("--model_folder", type=str, default="models/")
    parser.add_argument("--generate_dataset", type=bool, default=False)
    parser.add_argument("--use_NN", type=bool, default=False)
    parser.add_argument("--run_cem", type=bool, default=False)
    parser.add_argument("--horizon_planning", type=int, default=10)

    parser.add_argument("--cem_num_particles", type=int, default=30)
    parser.add_argument("--cem_num_iterations", type=int, default=100)
    parser.add_argument("--cem_num_elite", type=int, default=10)
    parser.add_argument("--cem_plot", type=bool, default=True)
    
    args = parser.parse_args()

    print("Running with args:")
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    create_plot_folder(args.plots_folder)
    clear_figures_in_folder(args.plots_folder)

    np.random.seed(0)

    # Setup Dino model.
    dino = Dino(name_model="dino_vits16")

    # Load NN Model.
    if args.use_NN == True:
        # Load the trained 'Phi' and 'a' model.
        with open("nn_config.yml", 'r') as stream:  # TODO put the file in args parser
            nn_options = yaml.safe_load(stream)
        dim_a = nn_options['dim_a']
        Phi_NN = PhiNN(nn_options['input_size'], nn_options['hidden_size'], 2 * dim_a)
        Phi_NN.load_state_dict(torch.load(args.model_folder + 'model.pth'))
        A_trained = np.load(args.model_folder + 'mean_a_list.npy')
        mlpfile.torch.write(Phi_NN, args.model_folder + "net.mlp")
        Phi_NN_mlp = mlpfile.Model.load(args.model_folder + "net.mlp")
        for lay in Phi_NN_mlp.layers:
            print(lay)

    # 2D Feature Maps.
    grass_img_features_dino, grass_img_lower_res = dino.process_img_and_get_dino_feat(grass_img_path)
    sand_img_features_dino, sand_img_lower_res = dino.process_img_and_get_dino_feat(sand_img_path)
    ice_img_features_dino, ice_img_lower_res = dino.process_img_and_get_dino_feat(ice_img_path)

    feature_map = np.zeros((h, w, dino.n_feats))
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    tiles_y = h // grass_img_lower_res.shape[0] # number of tiles in y (height)
    tiles_x = w // grass_img_lower_res.shape[1] # number of tiles in x (width)
    rgb_map = np.tile(grass_img_lower_res, (tiles_y, tiles_x, 1))
    assert rgb_map.shape == (h, w, 3)
    tiled_img_dino = np.tile(grass_img_features_dino, (1, tiles_y, tiles_x))
    np.save(args.dataset_folder + 'rgb_map.npy', rgb_map)

    # Background Map.
    feature_map = tiled_img_dino.transpose(1, 2, 0)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    xy = np.concatenate([x[:, :, None], y[:, :, None]], axis=-1)

    # 2D Damping Map.
    damping_map = np.ones((h, w))
    if system == 'single_integrator':
        damping_scaling = 0.01
    if system == 'double_integrator':
        damping_scaling = 5.0

    # Sand pits.
    update_feat_rgb_and_damping_maps(xy, feature_map, rgb_map, damping_map,
                                            damping_scaling=damping_scaling,
                                            x_coord_list=[40, 90, 150, 210],
                                            y_coord_list=[h//4, h//4, h//1.5, h//1.5],
                                            img_dino_map=sand_img_features_dino,
                                            img_rgb_map_lower_res=sand_img_lower_res)

    # Systems.
    if system == 'single_integrator':
        true_system = TwoDimSingleIntegratorDamping()
        nominal_system = TwoDimSingleIntegratorNominal()
    if system == 'double_integrator':
        true_system = TwoDimDoubleIntegratorDamping()
        nominal_system = TwoDimDoubleIntegratorNominal()

    # Generate Dataset for Training and Testing.
    if args.generate_dataset == True:
        print('Generate Dataset')
        create_plot_folder(args.dataset_folder)

        def generate_dataset_nb_steps(system,
                                      init_cond,
                                      nb_steps_total,
                                      step_size,
                                      u_x_lims, u_y_lims,
                                      gen_plots=True,
                                      data_type='training'):

            num_steps = nb_steps_total // step_size + 1
            nb_steps = np.linspace(0, nb_steps_total, num_steps, dtype=int)

            # Logs.
            xs_total = np.zeros((nb_steps_total, init_cond.size))
            us_total = np.zeros((nb_steps_total, system.udim))
            dino_features_total = np.zeros((nb_steps_total, dino.n_feats))
            time_total_seconds = np.zeros(nb_steps_total)
            time_since_last_flip_x = 1e+4
            time_since_last_flip_y = 1e+4

            x = init_cond.copy()
            if gen_plots == True:
                _, ax = plt.subplots()
                ax.imshow(rgb_map)

            for i in range(1, len(nb_steps)):
                duration = nb_steps[i] - nb_steps[i - 1]
                xs = np.zeros((duration, system.xdim))
                us = np.zeros((duration, system.udim))
                dino_features = np.zeros((duration, dino.n_feats))
                u_x = np.random.uniform(u_x_lims[0], u_x_lims[1])
                u_y = np.random.uniform(u_y_lims[0], u_y_lims[1])

                for t in range(duration):
                    xs[t] = x
                    us[t] = np.array([u_x, u_y])
                    dino_features[t] = feature_map[int(x[IDX_PY] + 0.5), int(x[IDX_PX] + 0.5), :]
                    x, _, _ = system.dynamics(x, us[t], dt, damping_map)

                    # Turn around at edges of map.
                    mapshape = np.array(feature_map.shape)[[1, 0]]
                    x, u_x, u_y, time_since_last_flip_x, time_since_last_flip_y = \
                        turn_around_edges(x, u_x, u_y, time_since_last_flip_x, time_since_last_flip_y, mapshape)

                xs_total[nb_steps[i - 1] : nb_steps[i], :] = xs
                us_total[nb_steps[i - 1] : nb_steps[i], :] = us
                time_total_seconds[nb_steps[i - 1] : nb_steps[i]] = np.arange(nb_steps[i - 1], nb_steps[i], 1) * dt
                dino_features_total[nb_steps[i - 1] : nb_steps[i], :] = dino_features

                x = xs[-1].copy()

                if gen_plots == True:
                #     if i == len(nb_steps) - 1:
                    ax.plot(xs[:, IDX_PX], xs[ :, IDX_PY], alpha=1.0)
                    ax.set(xlabel="x [m]", ylabel="y [m]", aspect='equal') #, title='Trajectory for ' + data_type)

            if gen_plots == True:
                plt.savefig(args.plots_folder + str(i) + data_type + "_trajectory.pdf", dpi=200, transparent=True)
                plt.show()

            return xs_total, us_total, dino_features_total, time_total_seconds

        def save_data_to_h5(xs_total, us_total, time_stamp, dino_features_total, name_file):
                df = pd.DataFrame()
                df['position.x'] = xs_total[:, IDX_PX]
                df['position.y'] = xs_total[:, IDX_PY]
                df['velocity.x'] = xs_total[:, IDX_VX]
                df['velocity.y'] = xs_total[:, IDX_VY]
                df['cmd_vel_u.x'] = us_total[:, IDX_UX]
                df['cmd_vel_u.y'] = us_total[:, IDX_UY]
                df['timestamp'] = time_stamp

                # create a dictionary to hold the new columns
                dino_dict = {}
                # loop to create all the new columns
                for idx_dino_features in range(dino_features_total.shape[1]):
                    column_name = 'dino_features_field.data.data' + str(idx_dino_features)
                    dino_dict[column_name] = dino_features_total[:, idx_dino_features]
                dino_df = pd.DataFrame(dino_dict)
                df = pd.concat([df, dino_df], axis=1)
                df.to_hdf(args.dataset_folder + name_file, key='stage', mode='w')


        data_types = ['train', 'test']
        total_steps = 1000000
        data_lens = [int(80/100*total_steps), int(20/100*total_steps)]
        for data_type, data_len in zip(data_types, data_lens):
            x0 = np.array([50.0, 0.0, 50.0, 0.0])
            gen_plots = True; data_type = data_type
            nb_steps_total_train = data_len
            step_size_train = 500
            u_x_min, u_x_max = -10.0, 10.0
            u_y_min, u_y_max = -10.0, 10.0

            # Generate Training Data.
            xs_total_train, us_total_train, dino_features_total_train, t_total_sec_train = generate_dataset_nb_steps(system=true_system,
                init_cond=x0,
                nb_steps_total=nb_steps_total_train,
                step_size=step_size_train,
                u_x_lims=[u_x_min, u_x_max], u_y_lims=[u_y_min, u_y_max],
                gen_plots=gen_plots, data_type=data_type)

            # Save Data.
            save_data_to_h5(xs_total_train, us_total_train, t_total_sec_train, dino_features_total_train, name_file=data_type + '_data.h5')

            # Plot control inputs.
            fig, ax = plt.subplots(us_total_train.shape[1], 1)
            ax[0].plot(us_total_train[:, IDX_UX], label="u_x", color='red')
            ax[1].plot(us_total_train[:, IDX_UY], label="u_y", color='blue')
            ax[0].set(title='Ux Input')
            ax[1].set(title='Uy Input')
            for _ax in ax:
                _ax.legend()
                _ax.set(xlabel="Time [s]")

        import sys
        sys.exit(0)

    # ---- Initial and Final Conditions ---
    # Set x0 and xT.
    if system == 'single_integrator':
        x0 = np.array([10.0, 25.0])
        xT = np.array([x0[IDX_PX] + 60.0, x0[IDX_PY]])
    if system == 'double_integrator':
        x0 = np.array([10.0, 0.0, 25.0, 0.0])
        xT = np.array([x0[IDX_PX] + 200.0, 0.0, x0[IDX_PY] + 40, 0.0])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(rgb_map)
    ax[1].imshow(damping_map, cmap='jet')
    plt.savefig(args.plots_folder + 'map.pdf', dpi=200, transparent=True)

    fig, ax = plt.subplots()
    ax.imshow(rgb_map, alpha=0.5)
    ax.plot(x0[IDX_PX], x0[IDX_PY], "o", label="start")
    ax.plot(xT[IDX_PX], xT[IDX_PY], "x", label="end")
    ax.legend()
    ax.set(xlabel="x [m]", ylabel="y [m]")
    plt.show()

    if args.run_cem == True:
        system_cem = nominal_system

        mean = np.zeros((args.horizon_planning, system_cem.udim)) # (horizon x action_dim)
        cov = np.array([20.0 * np.eye(2) for _ in range(args.horizon_planning)]) # (horizon x action_dim x action_dim)

        cem = CEM(horizon=args.horizon_planning,
                num_particles=args.cem_num_particles,
                num_iterations=args.cem_num_iterations,
                num_elite=args.cem_num_elite,
                mean=mean,
                cov=cov)

        if args.use_NN == True:
            best_action, best_actions = cem.optimize(x0, system_cem, dt, xT, damping_map=damping_map,
                                                    terrain_map=feature_map, NN=[Phi_NN_mlp, A_trained])
        else:
            best_action, best_actions = cem.optimize(x0, system_cem, dt, xT, damping_map=damping_map)

        t = np.arange(args.horizon_planning) * dt

        state = x0.copy()
        state_list = [state]
        state_list_all_actions = []
        for i in range(args.horizon_planning):
            # if args.use_NN == True:
            #     next_state, _, _ = true_system.dynamics(state, best_action[i], dt, feature_map, NN=[Phi_NN_mlp, A_trained])
            # else:
            next_state, _, _ = true_system.dynamics(state, best_action[i], dt, damping_map)

            state = next_state.copy()
            state_list.append(state)

        for i in range(args.cem_num_iterations):
            state = x0.copy()
            state_list_all_actions.append([state])
            for j in range(args.horizon_planning):
                # if args.use_NN == True:
                #     next_state, _, _ = true_system.dynamics(state, best_actions[i][j], dt, feature_map, NN=[Phi_NN_mlp, A_trained])
                # else:
                next_state, _, _ = true_system.dynamics(state, best_actions[i][j], dt, damping_map)
                state = next_state.copy()
                state_list_all_actions[i].append(state)

        state_list = np.array(state_list)

        if args.cem_plot:
            # plot the trajectory
            fig, ax = plt.subplots()
            ax.imshow(rgb_map, alpha=0.5)
            ax.set(xlabel="x [m]", ylabel="y [m]", aspect='equal')
            for i in range(args.cem_num_iterations):
                ax.plot(np.array(state_list_all_actions[i])[:, IDX_PX], np.array(state_list_all_actions[i])[:, IDX_PY], alpha=0.5)
            ax.plot(state_list[:, IDX_PX], state_list[:, IDX_PY], label="true", linewidth=2, color='black')
            ax.plot(x0[IDX_PX], x0[IDX_PY], "o", label="start")
            ax.plot(xT[IDX_PX], xT[IDX_PY], "o", label="end")
            ax.legend()
            ax.set(xlabel="x", ylabel="y")

            # plot us
            fig, ax = plt.subplots()
            ax.plot(best_action[:, IDX_UX], label="u_x", linewidth=2, color='black')
            ax.plot(best_action[:, IDX_UY], label="u_y", linewidth=2, color='black')
            plt.show()

        # save the best action
        np.save(args.dataset_folder + 'best_action_cem.npy', best_action)
        np.save(args.dataset_folder + 'best_state_cem.npy', state_list)