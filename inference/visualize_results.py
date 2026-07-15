import matplotlib.pyplot as plt
from PIL import Image


def show_results(query_image, results):
    """
    Display query image and retrieved similar images.
    """

    n = len(results)

    plt.figure(figsize=(18, 4))

    # Query image
    plt.subplot(1, n + 1, 1)
    plt.imshow(Image.open(query_image))
    plt.title("Query")
    plt.axis("off")

    # Retrieved images
    for i, result in enumerate(results):

        plt.subplot(1, n + 1, i + 2)

        img = Image.open(result["image_path"])

        plt.imshow(img)

        plt.title(
            f"Rank {i+1}\n{result['distance']:.4f}",
            fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()