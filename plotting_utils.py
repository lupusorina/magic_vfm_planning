
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
import glob
import os

import numpy as np
import matplotlib

import pandas as pd


# Plotting constants.
HINDSIGHT_COLOR = "orange"
IDENTITY_COLOR = "#ABD"
OURS_COLOR = "black"
font = {'size'   : 16}

matplotlib.rc('font', **font)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Create a new colormap from a sub-interval of a supplied one.

    https://stackoverflow.com/a/18926541
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def draw_map(featuremap, cbar=True, folder_name=None):
    for i in range(featuremap.shape[2]):
        fig, ax = plt.subplots(figsize=(8, 4))
        handle = ax.imshow(featuremap[:, :, i], cmap='jet', interpolation="nearest", vmin=-4, vmax=4)
        ax.set(xlabel="x", ylabel="y", aspect='equal')
        if cbar:
            fig.colorbar(handle, ax=ax, label='dino feature', ticks=[], location="left")
        if folder_name != None:
            plt.savefig(folder_name + '/featuremap_for_feature_nb_' + str(i) + '.png', dpi=200)


def draw_map_image(img_dino, cbar=True, folder_name=None):
    for i in range(img_dino.shape[2]):
        fig, ax = plt.subplots(figsize=(8, 4))
        handle = ax.imshow(img_dino[i, :, :], cmap='jet', interpolation="nearest", vmin=-4, vmax=4)
        ax.set(xlabel="x", ylabel="y", aspect='equal')
        if cbar:
            fig.colorbar(handle, ax=ax, label='dino feature', ticks=[], location="left")
        if folder_name != None:
            plt.savefig(folder_name + '/image_map_for_feature_nb_' + str(i) + '.png', dpi=200)


def draw_map_single_feature(fig, ax, featuremap, idx_feature, cbar=True, folder_name=None):
    """Draws the scalar feature map for one feature in low contrast, such that traj can be plotted over."""
    cmap  = truncate_colormap("gist_gray", minval=0.4, maxval=0.9)
    handle = ax.imshow(featuremap[:, :, idx_feature], cmap=cmap, interpolation="nearest", vmin=-4, vmax=4)
    ax.set(xlabel="x", ylabel="y", aspect='equal')
    if cbar:
        fig.colorbar(handle, ax=ax, label='dino feature', ticks=[], location="left")
    if folder_name != None:
        plt.savefig(folder_name + '/single_feature_nb_' + str(idx_feature) + '.png', dpi=200)


def plot_color_segments(fig, ax, xysegs, err, worst_err, label=None):
    lc = mpl.collections.LineCollection(
    xysegs, cmap="rainbow", linewidth=4, norm=mpl.colors.PowerNorm(0.5))
    lc.set_array(err)
    lc.set_clim((0, worst_err))
    handle = ax.add_collection(lc)
    if label is not None:
        fig.colorbar(handle, ax=ax, label=label)


def generate_gif_from_images(img_dir, name_file, output_path, output_name, delete_images=False):
    """ Generates a gif from a list of images."""
    files = glob.glob(img_dir + '*' + name_file)
    images = [Image.open(image) for image in files]
    images.sort(key=lambda x: int(x.filename.split('/')[-1].split('_')[0]))
    images[0].save(output_path + output_name + '.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
    if delete_images:
        for image in files:
            os.remove(image)

def plot_loss_fcns_and_adapt_vectors(loss_train_pd, loss_test_pd, as_pd):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(loss_train_pd['Step'], loss_train_pd['Value'], label='train')
    ax[0].plot(loss_test_pd['Step'], loss_test_pd['Value'], label='test')
    ax[0].legend()
    ax[0].set_ylim([-0.0001, 0.01])
    for a_pd in as_pd:
        ax[1].plot(a_pd['Step'], a_pd['Value'])
    plt.tight_layout()
    plt.show()


def create_plot_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def clear_figures_in_folder(folder_name):
    folders_to_delete_files_from = [folder_name]
    for folders in folders_to_delete_files_from:
        filelist = [ f for f in os.listdir(folders) if f.endswith(".png") or f.endswith(".pdf") or f.endswith(".gif")]
        for f in filelist:
            os.remove(os.path.join(folders, f))

