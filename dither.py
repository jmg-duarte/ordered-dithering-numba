from PIL import Image
import numba
import numpy as np

from bayer import bayer_dithering_2x2, bayer_dithering_4x4, bayer_dithering_8x8


def main():
    factor = 1

    image = Image.open("image.JPG")
    image = image.resize((int(image.size[0] * factor), int(image.size[1] * factor)))
    image_arr = np.array(image)

    output_arr: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.ndarray(
        image_arr.shape[:2]
    )
    process(image_arr.astype(np.uint8), output_arr)

    output = Image.fromarray(output_arr.astype(np.uint8), "L")
    output.save("output.jpeg")


@numba.njit(cache=True)
def process(
    input: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    output: np.ndarray[tuple[int, int], np.dtype[np.float32]],
):
    h, w, _ = input.shape
    for y in range(0, h):
        for x in range(0, w):
            luminance = luminance_itu_bt709(
                input[y, x, 0], input[y, x, 1], input[y, x, 2]
            )
            color = bayer_dithering_8x8(w, h, x, y, luminance, 0.45)
            output[y, x] = color


@numba.njit(cache=True)
def luminance_itu_bt709(
    r: int,
    g: int,
    b: int,
) -> float:
    return 0.2126 * (r / 255) + 0.7152 * (g / 255) + 0.0722 * (b / 255)


if __name__ == "__main__":
    main()
