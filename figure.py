import matplotlib.pyplot as plt
import numpy as np


reconstruction_title = "Reconstructed Images"
error_chart_title = "Mean Reconstruction Error"

labels = [
    "Ground Truth",
    "Standard",
    "Variational",
    "KL Loss"
]

reconstruction_scale = 3
error_chart_scale = 6


def _reconstruct_images(images, autoencoders):
    reconstructions = []
    for encoder, decoder in autoencoders:
        embeddings = encoder.predict(images)
        if isinstance(embeddings, list):
            embeddings = embeddings[0]

        reconstructions.append(decoder.predict(embeddings))
    return reconstructions


def _generate_reconstructions_from_rows(rows):
    num_samples = len(rows[0])

    fig = plt.figure(figsize=(num_samples * reconstruction_scale, len(rows) * reconstruction_scale))
    fig.suptitle(reconstruction_title, fontsize=10 * reconstruction_scale)
    plt.gray()

    for y, images in enumerate(rows):
        for x, image in enumerate(images):
            axes = plt.subplot(len(rows), num_samples, y * num_samples + x + 1)
            plt.imshow(image)

            axes.get_xaxis().set_visible(False)
            plt.yticks([])
            if x == 0:
                plt.ylabel(labels[y], fontsize=7 * reconstruction_scale)

    plt.savefig("reconstructions.png")


def generate_reconstructions(images, autoencoders):
    rows = [images] + _reconstruct_images(images, autoencoders)
    _generate_reconstructions_from_rows(rows)


def generate_error_chart(images, autoencoders):
    errors = []
    for reconstructions in _reconstruct_images(images, autoencoders):
        error = np.mean((images - np.squeeze(reconstructions))**2)
        errors.append(error)

    fig = plt.figure(figsize=(error_chart_scale, error_chart_scale))
    plt.bar(labels[1:], errors)
    plt.suptitle(error_chart_title, fontsize=3 * error_chart_scale)
    fig.axes[0].set_xticklabels(labels[1:], fontsize=2 * error_chart_scale)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

    plt.savefig("errors.png", dpi=300)
