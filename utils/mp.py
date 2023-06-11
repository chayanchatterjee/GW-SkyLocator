"""Utilities for Python multiprocessing."""

import concurrent.futures
import logging
import multiprocessing as mp
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .array import chunk_iterable

logger = logging.getLogger(__name__)


def validate_cpu_count(nproc: int) -> int:
    """Validates whether the provided nproc argument is within a valid range of
    multiprocessing CPU cores based on the user's machine specifications.

    Parameters
    ----------
    nproc: int
        The number of processes to be used for CPU multiprocessing.

    Returns
    -------
    int
        A valid CPU count (nproc) integer for use in multiprocessing programs.
    """
    if nproc == -1:
        nproc = mp.cpu_count()
    if not isinstance(nproc, int):
        raise TypeError(f"nproc must be of type int, not type {type(nproc)}.")
    if not 1 <= nproc <= mp.cpu_count():
        raise ValueError(f"nproc ({nproc}) must be 1 <= nproc <= {mp.cpu_count()}.")
    return nproc


def parallel_apply(
    func: Callable,
    data: Union[Sequence, np.ndarray, pd.DataFrame],
    nproc: int = 4,
    chunk_size: Optional[int] = None,
    concat: bool = True,
    desc: Optional[str] = None,
    verbose: bool = False,
    *args,
    **kwargs,
) -> Any:
    """Simple multiprocessing for arbitrary functions or class methods on large arrays.

    This function will break up input data into chunks of size `chunk_size`, and passes
    each chunk to the pool of `nproc` processes that each runs the `func` callable
    using concurrent.futures.ProcessPoolExecutor. The input data is iterated over its
    length (i.e. **first dimension**), and will return an output of equal length
    according to the output signature of the specified function if `concat` is True. If
    `concat` is False, the values of chunked array processing will be returned as
    separated elements within the returned list.

    Parameters
    ----------
    func: Callable
        An instantiated class method or function that processes arrays when called.
        Any args or kwargs passed to this function will be passed through to `func`.
    data: Sequence | np.ndarray | pd.DataFrame
        An iterable n-dimensional sequence, np.ndarray, pd.DataFrame that is a valid
        input argument to `func` - i.e. if the function requires 2-dimensional data,
        then we would require data to have shape (n, 2), where n is its iterable length.
    nproc: int = 4
        The number of CPU processes to use for multiprocessing.
    chunk_size: int | None
        The intermediate size to break the input array before being passed to processes.
        If None, chunks will be (approximately) evenly split between each process with a
        ceiling if the data length is not evenly divisible by the number of processes.
    concat: bool
        Whether to concatenate the output result chunks together according to its dtype.
    desc: str | None
        The text description for the progress bar.
    verbose: bool
        Whether or not to display the progress bar.

    Returns
    -------
    Any:
        The output computed by `func` given the input data.

    Notes
    -----
    If any parameters need to be passed to the `func` function beforehand, they can be
    passed as args and kwargs inputs respectively, which uses functools.partial to
    partially instantiate a version of their function with their appropriate arguments
    and keyword arguments before being passed to each multiprocessing worker.

    For more complex cases (e.g. having different function parameters for each element
    of the input data, or where multiple multiprocessing steps are required per
    element), we recommend that the user writes their own multiprocessing
    implementation using the code in this function as a skeleton.

    Examples
    --------
    For example, consider a Kernel Density Estimator model. The computational cost for
    scoring samples from a fitted KDE greatly increases with respect to the number of
    samples. Additionally, KDEs fitted with a large number of samples also take much
    longer to estimate scores than KDEs fitted with a smaller number of samples.
    For these reasons, we may wish to improve the speed of computing the score_samples
    method of KDE models via multiprocessing with this function.

    For a thorough summary of the computational efficiency and trade-offs for KDE see:
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/.

    >>> import numpy as np
    >>> from sklearn.neighbors import KernelDensity
    ...
    >>> # assume we have some 2-dimensional np.ndarray data we want to fit a KDE to
    >>> seed = 100
    >>> rng = np.random.default_rng(seed)
    >>> data = np.stack([
    ...    rng.normal(loc=0, scale=1, size=100000),
    ...    rng.normal(loc=5, scale=3, size=100000)
    ... ], axis=1)
    ...
    >>> # randomly create train and test split
    >>> train_idx = rng.choice(range(data.shape[0]), 50000, replace=False)
    >>> train = data[train_idx]
    >>> test = data[~train_idx]
    ...
    >>> # fit kde to training data
    >>> kde = KernelDensity(bandwidth=0.3, rtol=1e-4)
    >>> kde.fit(train)  # cannot parallelise .fit method
    ...
    >>> # estimate likelihood scores for test samples with 4 workers in parallel
    >>> log_density = parallel_apply(kde.score_samples, test, nproc=4)
    """
    # validate chunk size
    nproc = validate_cpu_count(nproc)
    total = len(data)
    chunk_size = chunk_size or int(np.ceil(total / nproc))
    if not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size {chunk_size} must be an integer.")
    if not (1 <= chunk_size <= total):
        raise ValueError(f"chunk_size {chunk_size} must be 1 <= chunk_size <= {total}")
    logger.debug(f"Calling {str(func)} with nproc {nproc} and chunk_size {chunk_size}.")

    _func = partial(func, *args, **kwargs)
    with tqdm(total=total, desc=desc, disable=not verbose) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
            # NOTE: futures may complete in a different order than they are submitted
            #   so we enumerate chunks to can keep track of futures after submission
            future_to_chunk_idx = {}
            chunk_idx_size = {}
            for idx, chunk in enumerate(chunk_iterable(range(total), size=chunk_size)):
                indices = list(chunk)  # converts from iterable to list for indexing
                if not isinstance(data, pd.DataFrame):
                    data_chunk = data[indices]  # type: ignore
                else:
                    data_chunk = data.iloc[indices]
                future = executor.submit(_func, data_chunk)
                future_to_chunk_idx[future] = idx  # futures stored as keys in this dict
                chunk_idx_size[idx] = len(indices)  # len may change for last chunk(s)

            # as_completed will block later code until all futures have finished
            for future in concurrent.futures.as_completed(future_to_chunk_idx):
                idx = future_to_chunk_idx[future]
                pbar.update(chunk_idx_size[idx])
            pbar.refresh()

            # join all futures in order of submission (sorted by chunk_idx value)
            ordered_futures = sorted(future_to_chunk_idx.items(), key=lambda it: it[1])
            results = [future[0].result() for future in ordered_futures]

            if concat:
                # concatenate results together based on first result type
                if isinstance(results[0], np.ndarray):
                    logger.debug(f"Concatenating {idx+1} chunks as a np.ndarray...")
                    return np.concatenate(results)
                elif isinstance(results[0], pd.DataFrame):
                    logger.debug(f"Concatenating {idx+1} chunks as a pd.DataFrame...")
                    return pd.concat(results)
                else:
                    logger.debug(f"Flattening {idx+1} chunks together as a list...")
                    return [x for chunk in results for x in chunk]
            else:
                logger.debug("Returning result chunks without concatenation...")
                return results
