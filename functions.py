import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def plot_l2_norm():
    l2_norm = np.load('./TimeReversal/l2_norm.npy')

    plt.figure()
    plt.imshow(l2_norm, aspect='auto')
    plt.colorbar()
    plt.grid()
    plt.title('L2-Norm - Time Reversal')
    plt.show()


def save_image(image, path):
    plt.imshow(image, cmap='viridis', interpolation='none')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_video(total_time, animation_step, image_size, frame_path, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    out = cv2.VideoWriter(output_path, fourcc, fps, image_size)

    for i in range(0, total_time, animation_step):
        current_frame_path = f'{frame_path}/frame_{i}.png'
        img = cv2.imread(current_frame_path)
        out.write(cv2.resize(img, image_size))

    out.release()


def save_rtm_image(upper_left, upper_right, bottom_left, bottom_right, path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Upper left subplot
    axs[0, 0].imshow(upper_left, cmap='viridis', interpolation='none')
    axs[0, 0].axis('off')

    # Upper right subplot
    axs[0, 1].imshow(upper_right, cmap='viridis', interpolation='none')
    axs[0, 1].axis('off')

    # Bottom left subplot
    axs[1, 0].imshow(bottom_left, cmap='viridis', interpolation='none')
    axs[1, 0].axis('off')

    # Bottom right subplot
    axs[1, 1].imshow(bottom_right, cmap='viridis', interpolation='none')
    axs[1, 1].axis('off')

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
