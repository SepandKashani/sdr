import random

import sdr.util.argcheck as pua


@pua.check(N=pua.is_integer)
def alphanumeric(N: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    stem = "".join(random.choices(alphabet, k=N))
    return stem
