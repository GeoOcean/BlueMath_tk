import os
import numpy as np


def write_array_in_file(array: np.ndarray, filename: str) -> None:
    """
    Write an array in a file.

    Parameters
    ----------
    array : np.ndarray
        The array to be written. Can be 1D or 2D.
    filename : str
        The name of the file.
    """

    with open(filename, "w") as f:
        if array.ndim == 1:
            for item in array:
                f.write(f"{item}\n")
        elif array.ndim == 2:
            for row in array:
                f.write(" ".join(map(str, row)) + "\n")
        else:
            raise ValueError("Only 1D and 2D arrays are supported")


def copy_files(src: str, dst: str) -> None:
    """
    Copy file(s) from source to destination.

    Parameters
    ----------
    src : str
        The source file.
    dst : str
        The destination file.
    """

    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for file in os.listdir(src):
            with open(file, "r") as f:
                content = f.read()
            with open(os.path.join(dst, file), "w") as f:
                f.write(content)
    else:
        with open(src, "r") as f:
            content = f.read()
        with open(dst, "w") as f:
            f.write(content)
