import numpy as np


def block_maxima(
    x: np.ndarray,
    block_size: int | float = 365.25,
    min_sep: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to obtain the Block Maxima of given size taking into account
    minimum independence hypothesis

    Parameters
    ----------
    x : np.ndarray
        Data used to compute the Block Maxima
    block_size : int, default=5
        Size of BM in index units (if daily data, nº of days), by default 5
    min_sep : int, optional
        Minimum separation between maximas in index units, by default 2

    Returns
    -------
    idx : np.ndarray
        Indices of BM
    bmaxs : np.ndarray
        BM values

    Raises
    ------
    ValueError
        Minimum separation must be smaller than (block_size+1)/2

    Example
    -------
    >>> # 1-year of daily values
    >>> x = np.random.lognormal(1, 1.2, size=365)

    >>> # 5-day Block Maxima with 72h of independency
    >>> idx, bmaxs = block_maxima(
    >>>     x,
    >>>     block_size=5,
    >>>     min_sep=3
    >>> )
    """
    block_size = int(np.ceil(block_size))

    if min_sep > (block_size + 1) / 2:
        raise ValueError("min_sep must be smaller than (block_size+1)/2")

    x = np.asarray(x)
    n = x.size

    # Partition into non-overlapping blocks
    n_blocks = int(np.ceil(n / block_size))
    segments_idx = []
    segments = []
    blocks = []
    # For each block, keep a *ranked* list of (idx, value) candidates (desc by value)
    for b in range(n_blocks):
        start = b * block_size
        stop = min((b + 1) * block_size, n)
        segment = x[start:stop]

        candidates_idx = np.argsort(segment)[::-1]

        segments_idx.append(np.arange(start, stop))
        segments.append(segment)
        blocks.append(candidates_idx)

    def violates(i_left, i_right):
        if i_left is None or i_right is None:
            return False
        return i_right - i_left < min_sep

    changed = True
    while changed:
        changed = False

        for b in range(n_blocks - 1):
            idx_left = segments_idx[b][blocks[b][0]]
            idx_right = segments_idx[b + 1][blocks[b + 1][0]]
            if not violates(idx_left, idx_right):
                continue
            else:
                idx_block_adjust = (
                    0
                    if segments[b][blocks[b][0]] < segments[b + 1][blocks[b + 1][0]]
                    else 1
                )
                blocks[b + idx_block_adjust] = np.delete(
                    blocks[b + idx_block_adjust], 0
                )
                changed = True
                break

    bmaxs = np.asarray([segments[b][blocks[b][0]] for b in range(n_blocks)])
    idx = np.asarray([segments_idx[b][blocks[b][0]] for b in range(n_blocks)])

    return idx, bmaxs
