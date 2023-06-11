"""Utilities for array processing."""

import logging
from itertools import chain, islice
from typing import Iterable

logger = logging.getLogger(__name__)


def chunk_iterable(iterable: Iterable, size: int = 1000) -> Iterable:
    """Function to break an iterable into groups of length `size`.

    The intermediate array chunks (`array`) can be handled independently as
    numpy arrays while preserving memory. Each returned chunk `x`` can be converted
    back to a list via `list(x)`. It is recommended to use np.fromiter to convert
    the output to a numpy array.

    Source: https://stackoverflow.com/a/24527424.

    Parameters
    ----------
    iterable: Iterable
        An iterable object such as an array, tuple, np.array, etc.
    size: int
        The number of elements per array chunk.

    Returns
    -------
    itertools.chain
        An iterable object.

    Examples
    --------
    The example below enables chunked array processing on a large numpy array.

    >>> very_large_array = np.random.randn(10000000000000)
    >>> for chunk in chunks(very_large_array, size=10000):
    ...    array = np.fromiter(chunk, dtype=very_large_array.dtype)
    ...    # do processing on array chunk
    """
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))
