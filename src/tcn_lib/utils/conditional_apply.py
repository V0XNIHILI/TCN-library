from typing import Callable


def conditional_apply(func: Callable, whether: bool):
    if whether:
        return func

    return lambda x: x
