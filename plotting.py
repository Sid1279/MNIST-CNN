import matplotlib.pyplot as plt  # type: ignore


def plot_train_test_accuracies(train_accuracies, test_accuracy):
    epochs = len(train_accuracies)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
    plt.axhline(y=test_accuracy, color="r", linestyle="--", label="Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracies")
    plt.legend()
    plt.show()


def plot_heatmap_samples(rgb_images, heatmaps, overlayed_images, sample_indices):
    plt.figure(figsize=(12, 4 * len(sample_indices)))

    for i, index in enumerate(sample_indices):
        plt.subplot(len(sample_indices), 3, i * 3 + 1)
        plt.imshow(rgb_images[index])
        plt.title(f"Original Image (Index {index})")
        plt.axis("off")

        plt.subplot(len(sample_indices), 3, i * 3 + 2)
        plt.imshow(heatmaps[index])
        plt.title(f"Heatmap (Index {index})")
        plt.axis("off")

        plt.subplot(len(sample_indices), 3, i * 3 + 3)
        plt.imshow(overlayed_images[index])
        plt.title(f"Overlayed Image (Index {index})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
