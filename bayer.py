import numba
import numpy as np

BAYER_2X2 = (
    np.array(
        [
            [0, 2],
            [3, 1],
        ]
    )
    / 4
)


@numba.njit(cache=True)
def bayer_dithering_2x2(
    rx: int, ry: int, x: int, y: int, lum: float, bias: float = 0.4
) -> int:
    """
    rxy (tuple[int, int]): image resolution
    xy (tuple[int, int]): pixel coordinates
    lum (float): pixel's luminance value, between 0 and 1 (inclusive)
    bias (float): bias, between 0 and 1 (inclusive)
    """
    x = int((x * rx) % 2)
    y = int((y * ry) % 2)
    threshold = BAYER_2X2[x, y]
    return 0 if lum < (threshold + bias) else 255


BAYER_4X4 = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ]
    )
    / 16
)


@numba.njit(cache=True)
def bayer_dithering_4x4(
    rx: int, ry: int, x: int, y: int, lum: float, bias: float = 0.4
) -> int:
    """
    rxy (tuple[int, int]): image resolution
    xy (tuple[int, int]): pixel coordinates
    lum (float): pixel's luminance value, between 0 and 1 (inclusive)
    bias (float): bias, between 0 and 1 (inclusive)
    """
    x = int((x * rx) % 4)
    y = int((y * ry) % 4)
    threshold = BAYER_4X4[x, y]
    return 0 if lum < (threshold + bias) else 255


BAYER_8X8 = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],  # 8x8 Bayer ordered dithering */
            [48, 16, 56, 24, 50, 18, 58, 26],  # pattern. Each input pixel */
            [12, 44, 4, 36, 14, 46, 6, 38],  # is scaled to the 0..63 range */
            [60, 28, 52, 20, 62, 30, 54, 22],  # before looking in this table */
            [3, 35, 11, 43, 1, 33, 9, 41],  # to determine the action. */
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ]
    )
    / 64
)


@numba.njit(cache=True)
def bayer_dithering_8x8(
    rx: int, ry: int, x: int, y: int, lum: float, bias: float = 0.5
) -> int:
    """
    rxy (tuple[int, int]): image resolution
    xy (tuple[int, int]): pixel coordinates
    lum (float): pixel's luminance value, between 0 and 1 (inclusive)
    bias (float): bias, between 0 and 1 (inclusive)
    """
    x = int((x * rx) % 8)
    y = int((y * ry) % 8)
    threshold = BAYER_8X8[y, x]
    return 0 if lum < (threshold + bias) else 255
