import torch.nn.functional as F
import cv2  # type: ignore
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register the hook to retrieve gradients and activations
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def forward(self, input_image):
        return self.model(input_image)

    def backward(self, output):
        output.backward(gradient=output)

    def generate_heatmap(self, input_image, target_class):
        self.model.zero_grad()
        output = self.forward(input_image)
        predicted_class = output.argmax(dim=1)

        if target_class is None:
            target_class = predicted_class

        self.backward(output[:, target_class])
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap /= heatmap.max()

        return heatmap


def generate_gradcam_visualizations(model, data, gradcam):
    rgb_images = []
    heatmaps = []
    overlayed_images = []

    for image_index in range(len(data)):
        rgb_image = data[image_index].permute(1, 2, 0).numpy()
        input_image = data[image_index].unsqueeze(0)
        input_image.requires_grad_()
        heatmap = gradcam.generate_heatmap(input_image, target_class=None)
        heatmap = cv2.resize(heatmap[0].numpy(), (32, 32))
        heatmap = heatmap[0]
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Convert RGB image to float32
        rgb_image = rgb_image.astype(np.float32)
        # Normalize the RGB image
        rgb_image = (rgb_image - np.min(rgb_image)) / (
            np.max(rgb_image) - np.min(rgb_image)
        )

        # Scale the heatmap
        heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]))
        heatmap = heatmap.astype(np.float32) / 255

        # Apply the overlay by blending the heatmap with the RGB image
        overlayed_image = cv2.addWeighted(rgb_image, 0.6, heatmap, 0.4, 0)

        # Add heatmap to the RGB image
        overlayed_image = np.clip(overlayed_image, 0, 1)

        rgb_images.append(rgb_image)
        heatmaps.append(heatmap)
        overlayed_images.append(overlayed_image)

    return rgb_images, heatmaps, overlayed_images
