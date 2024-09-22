import numpy as np
import matplotlib.pyplot as plt
import ffmpeg
from matplotlib.image import imread


def plot_l2_norm():
    l2_norm = np.load('./TimeReversal/l2_norm.npy')

    plt.figure()
    plt.imshow(l2_norm, aspect='auto')
    plt.colorbar()
    plt.grid()
    plt.title('L2-Norm - Time Reversal')
    plt.show()


def save_image(image, path):
    plt.imsave(path, image)


def create_video(path, output_path):
    ffmpeg.input(fr'{path}/frame_%d.png', framerate=25).output(output_path).run()


def save_rtm_image(upper_left, upper_right, bottom_left, bottom_right, path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Upper left subplot
    axs[0, 0].imshow(upper_left, cmap='viridis', interpolation='none')
    axs[0, 0].set_title('Up-Going')

    # Upper right subplot
    axs[0, 1].imshow(upper_right, cmap='viridis', interpolation='none')
    axs[0, 1].set_title('Product')

    # Bottom left subplot
    axs[1, 0].imshow(bottom_left, cmap='viridis', interpolation='none')
    axs[1, 0].set_title('Down-Going')

    # Bottom right subplot
    axs[1, 1].imshow(bottom_right, cmap='viridis', interpolation='none')
    axs[1, 1].set_title('Accumulated Product')

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def convert_image_to_matrix(image_path):
    rgb_raw_image = np.int32(imread(image_path))

    velocity_map = {
        'white': 'receptors',
        'black': '0',
        'blue': '1500',
        'green': '3200',
        'red': '6400',
    }
    binary_color = {
        7: 'white',
        0: 'black',
        1: 'red',
        2: 'green',
        4: 'blue',
    }

    rgb_2d_grid = np.zeros_like(rgb_raw_image[:, :, 0])

    b = 1
    for i in range(3):
        b += i
        rgb_2d_grid += rgb_raw_image[:, :, i] * b

    rgb_string = np.array(rgb_2d_grid, dtype='str')
    for k in binary_color.keys():
        rgb_string[rgb_string == str(k)] = velocity_map[binary_color[k]]

    receptor_pos = np.where(rgb_string == 'receptors')
    rgb_string[receptor_pos] = '1500'
    receptor_z, receptor_x = np.int32(receptor_pos)

    rgb_float = np.array(rgb_string, dtype=np.float32)

    return rgb_float, receptor_z, receptor_x
