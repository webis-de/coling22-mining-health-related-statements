import multiprocessing as mp
from functools import partial
from itertools import repeat, takewhile
from typing import Any, Callable

import numpy as np
import pandas as pd


def count_lines(file, verbose=False):
    bufgen = takewhile(lambda x: x, (file.read(1024 * 1024) for _ in repeat(None)))
    count = 0
    for buf in bufgen:
        count += buf.count(b"\n")
        if verbose:
            print(count, end="\r")
    if verbose:
        print()
    return count


def parallelize_dataframe(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    num_processes: int,
    **kwargs: Any
) -> pd.DataFrame:
    df_split = np.array_split(df, num_processes)
    pool = mp.Pool(num_processes)
    func = partial(func, **kwargs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
