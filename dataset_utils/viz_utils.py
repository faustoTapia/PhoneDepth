import numpy as np
from matplotlib import pyplot as plt
from typing import List
from pathlib import Path

def visualize_aug(original, augmented, depth_colormap=None):
    fig = plt.figure(figsize=[6.4*2, 4.8*2])
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)

def visualize_aug(original, augmented, original_target, augmented_target, depth_colormap="Greys"):
    fig = plt.figure(figsize=[6.4*4, 4.8*4])
    plt.subplot(1,4,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,4,2)
    plt.title('Augmented image')
    plt.imshow(augmented)

    plt.subplot(1,4,3)
    plt.title('Original target')
    plt.imshow(original_target, cmap=depth_colormap)

    plt.subplot(1,4,4)
    plt.title('Augmented target')
    plt.imshow(augmented_target, cmap=depth_colormap)


def visualize_aug_batch(original_batch, augmented_batch, num_elems=3):
    if num_elems > min(original_batch.shape[0], num_elems):
        raise ValueError("num_elems must be <5")

    for i in range(min(original_batch.shape[0], num_elems)):
        visualize_aug(original_batch[i], augmented_batch[i])

def visualize_aug_batch(orig_batch, input_aug_batch, orig_target, target_aug_batch, num_elems = 3, depth_colormap="Greys"):
    if num_elems > min(orig_batch.shape[0], 6):
        raise ValueError("num_elems must be <5")
    
    for i in range(min(orig_batch.shape[0], num_elems)):
        visualize_aug(orig_batch[i], input_aug_batch[i], orig_target[i], target_aug_batch[i], depth_colormap=depth_colormap)

def visualize_multiple_imag_rows(batch_list, labels, n_samples=3, histogram=False, color_map=None, vmin=None, vmax=None, figsize=(20,10)):
    for i in range(batch_list[0].shape[0]):
        if i >= n_samples:
            break
        img_row = [np.squeeze(batch[i]) for batch in batch_list]
        visualize_images_row(img_row, labels, color_map, vmin=vmin, vmax=vmax, figsize=figsize,  histogram=histogram)

def visualize_images_row(images, labels, color_map=None, vmin=None, vmax=None, figsize=(20,10), histogram=False):
    plt.figure(figsize=figsize)
    size = len(images)
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(1, size, i+1)
        plt.imshow(img, cmap=color_map, vmin=vmin, vmax=vmax)
        plt.title(label)
        print("-"*200)
        print(f"Img min, max, mean, median: [{np.min(img)}, {np.max(img)}, {np.mean(img)}, {np.median(img)}]")
        if histogram:
            hist_img, edges_img = np.histogram(img, 10)
            print(f"Histogram {i+1}: {hist_img} \nEdges: {edges_img}")
    plt.show()

def visualize_histograms_row(histograms, labels, figsize=(20,10)):
    fig = plt.figure(figsize=figsize)
    size = len(histograms)
    for i, (hist, label) in enumerate(zip(histograms, labels)):
        plt.subplot(1, size, i+1)
        plt.plot(hist[1][:-1], hist[0])
        plt.title(label)
    plt.show()

def save_corresponding_batches_viz (batch_list: List[np.ndarray], labels: List[str], output_dir="./", n_samples=4):
    for batch, label in zip(batch_list, labels):
        save_batch_viz(batch, label, output_dir, n_samples)

def save_batch_viz(batch: np.ndarray, label: str, output_dir: str, n_images = None):
    n_images = batch.shape[0] if n_images is None else n_images
    imgs = []
    file_names = []
    for i in range(n_images):
        imgs.append(batch[i])
        file_name = label + "{:03d}".format(i) + ".png"
        file_names.append(file_name)
    save_imgs_viz(imgs, file_names, output_dir)

def save_imgs_viz(imgs: List[np.ndarray], file_names: List[str], output_dir: str="./", cmap='gray'):
    for img, file_name in zip(imgs, file_names):
        file_path = Path(output_dir) / Path(file_name)
        save_img_for_viz(img, file_path.as_posix(), cmap=cmap)

def save_img_for_viz(img: np.ndarray, file_name: str, cmap='gray'):
    img = np.squeeze(img)
    if len(img.shape) > 2:
        img_type = "rgb"
    else:
        img_type = "depth"

    if img_type == "rgb":
        min_val = np.min(img)
        if min_val < 0:
            min_val = -1.0
            max_val = np.max(img)
        else:
            min_val = 0.0
            max_val = 255.0

        img = (img - min_val) * 255 / (max_val - min_val)
        img = np.array(img, dtype=np.uint8)
        plt.imsave(fname=file_name, arr=img)

    else:
        max_val = np.max(img)
        min_val = np.min(img)
        # For ordinal images
        img = (img - min_val) / (max_val - min_val)
        plt.imsave(fname=file_name, arr=img, vmin=0.0, vmax=1.0, cmap=cmap)