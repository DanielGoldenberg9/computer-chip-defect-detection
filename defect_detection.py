import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans


def create_images(paths: str) -> list[list]:
    all_images = []
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            all_images.append(img)
        else:
            print(f"Unable to load image located at {path}. skipping...")
            continue
    return all_images


def display_images(images: list[list]) -> None:
    imgs_len = len(images)
    num_cols = 3
    num_rows = (
        imgs_len // num_cols if imgs_len % num_cols == 0 else (imgs_len // num_cols) + 1
    )
    _, ax = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(11, 5))
    for i, img in enumerate(images):
        ax[i // num_cols, i % num_cols].imshow(img, cmap="gray")
        ax[i // num_cols, i % num_cols].set_title(f"Image  {i}")
        ax[i // num_cols, i % num_cols].axis("off")
    plt.tight_layout()
    plt.show()


def map_matrix_numpy(kmeans: KMeans, shape: tuple):
    """map each color to its corresponding label in the kmeans and give focus to a specific color in the labeled image.

    Args:
        kmeans (KMeans): trained kmean algorithm
        shape (tuple): shape of the original image
    """

    def map_label_centroid(i_color: int):
        mapping = {}
        label_centroid_map = label_centroid_mapping.copy()
        for i in range(len(label_keys)):
            label = min(label_centroid_map, key=label_centroid_map.get)
            value = 255 if i == i_color else 0
            mapping[label] = value
            label_centroid_map.pop(label)

        map_func = np.vectorize(lambda x: mapping.get(x, x))
        mapped_matrix = map_func(kmeans.labels_.reshape(shape))

        return mapped_matrix

    label_keys = set(kmeans.labels_)
    label_centroid_mapping = dict(zip(label_keys, kmeans.cluster_centers_))
    # black is the lowest value (0),gray is the second value (1) and white is the highest value(2)
    white_mapped_matrix = map_label_centroid(2)
    black_mapped_matrix = map_label_centroid(0)
    gray_mapped_matrix = map_label_centroid(1)

    return white_mapped_matrix, black_mapped_matrix, gray_mapped_matrix


def apply_k_means(img: np.ndarray):
    kmeans = KMeans(n_clusters=3, init="k-means++", n_init="auto")
    kmeans.fit(img.reshape(-1, 1))
    return map_matrix_numpy(kmeans, img.shape)


def find_defects(images: list[list]):
    """main function.
    This function takes in a list of pair images and saves the defected parts of the images.

    Args:
        images (list[list]): list of all the images

    """
    # Ensure the number of images is even
    assert len(images) % 2 == 0, "The number of images must be divisible by 2."
    for i_img in range(0, len(images), 2):
        insp, ref = images[i_img], images[i_img + 1]
        white_denoise, black_denoise, gray_denoise = clean_data(insp, ref)
        translation = align_images(
            white_denoise[0].astype(np.uint8), white_denoise[1].astype(np.uint8)
        )
        sub_array = []
        for denoise_insp, denoise_ref in [white_denoise, black_denoise, gray_denoise]:
            rows, cols = denoise_ref.shape

            aligned_ref, aligned_insp = cropp_images(
                denoise_ref, denoise_insp, translation, rows, cols
            )

            sub_array.append(high_recall(aligned_ref, aligned_insp))
            # sub_array.append(high_precision(aligned_ref, aligned_insp))

        # concat_image = np.sum(sub_array, axis=0).clip(0, 255).astype(np.uint8)
        concat_image = concatenate_images(sub_array)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(concat_image, cmap="gray")
        ax.set_title(f"image {i_img}")
        ax.axis("off")
        image_directory = "./images"

        if not os.path.exists(image_directory):
            os.mkdir(image_directory)
        fig.savefig(os.path.join(image_directory, f"image{i_img}"))


def clean_data(insp, ref):
    """apply the kmeans algorithm by the intensity of the image

    Returns:
        tuple: (white,black,gray) pickle-focus
    """
    white_denoise_insp, black_denoise_insp, gray_denoise_insp = apply_k_means(insp)
    white_denoise_ref, black_denoise_ref, gray_denoise_ref = apply_k_means(ref)
    return (
        (white_denoise_insp, white_denoise_ref),
        (
            black_denoise_insp,
            black_denoise_ref,
        ),
        (gray_denoise_insp, gray_denoise_ref),
    )


def concatenate_images(list_images: list):
    return np.sum(list_images, axis=0).clip(0, 255).astype(np.uint8)


def high_recall(ref, insp):
    kernel1 = np.ones((3, 3))
    kernel2 = np.ones((7, 7))
    img1 = ref - cv2.dilate(insp, kernel1)
    img2 = insp - cv2.dilate(ref, kernel2)
    outputs = []
    for img in [img1, img2]:
        white_counts = cv2.filter2D(img // 255, -1, kernel2)
        blob_mask = (white_counts >= 6).astype(np.uint8) * 255
        outputs.append(cv2.bitwise_and(img, img, mask=blob_mask))
    return cv2.bitwise_or(*outputs)


def high_precision(ref, insp):

    images = []
    # sub = cv2.absdiff(ref, insp)
    for sub in [ref - insp, insp - ref]:
        sub = cv2.erode(sub, np.ones((3, 3)))
        sub = cv2.dilate(sub, np.ones((3, 3)))
        images.append(sub)

    return concatenate_images(images)


def align_images(ref_image, insp_image):
    """
    calulates the phase correlation between reference image and inspected image
    """
    f_ref = np.fft.fft2(ref_image)
    f_insp = np.fft.fft2(insp_image)

    cross_power_spectrum = f_ref * np.conj(f_insp)
    # nomalizing in order to reduce the sensitivity of the intensity image.
    cross_corr = np.fft.ifft2(cross_power_spectrum / np.abs(cross_power_spectrum))
    shift = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)

    rows, cols = ref_image.shape
    translation = np.array(
        [
            -shift[0] if shift[0] < rows / 2 else rows - shift[0],
            -shift[1] if shift[1] < cols / 2 else cols - shift[1],
        ]
    )
    return translation


def cropp_images(ref_img, insp_img, translation, rows, cols):
    """
    apply the transformation on both reference and inspection images
    """
    aligned_ref = ref_img[
        max(0, translation[0]) : min(rows, rows + translation[0]),
        max(0, translation[1]) : min(cols, cols + translation[1]),
    ]
    aligned_insp = insp_img[
        max(0, -translation[0]) : min(rows, rows - translation[0]),
        max(0, -translation[1]) : min(cols, cols - translation[1]),
    ]
    return aligned_ref.astype(np.uint8), aligned_insp.astype(np.uint8)


def main():
    PATH_DEFECT = r"./defective_examples"
    PATH_NON_DEFECT = r"./non_defective_examples"

    defect_images_paths = [
        os.path.join(PATH_DEFECT, filename)
        for filename in os.listdir(PATH_DEFECT)
        if filename.endswith("tif")
    ]
    non_defect_images_paths = [
        os.path.join(PATH_NON_DEFECT, filename)
        for filename in os.listdir(PATH_NON_DEFECT)
        if filename.endswith("tif")
    ]
    all_images_paths = defect_images_paths + non_defect_images_paths
    all_images = create_images(all_images_paths)
    find_defects(all_images)
    # display_images(all_images)


if __name__ == "__main__":
    main()
