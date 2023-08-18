import numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt


def generate_gradcam(model, target_layers, images, use_cuda=True, transparency=0.6):
    results = []

    targets = None
    cam = EigenCAM(model, target_layers, use_cuda=use_cuda)

    for image in images:
        input_tensor = image.unsqueeze(0)
        grayscale_cam = cam(input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        img = input_tensor.squeeze(0).to("cpu")
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        cam_image = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency
        )
        results.append(cam_image)
    return results


def visualize_gradcam(images, figsize=(10, 10), rows=2, cols=5):
    fig = plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
