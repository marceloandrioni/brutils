#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__all__ = [
    "random_str",
]


import random
import string

from ._type_validation import validate_types_in_func_call


@validate_types_in_func_call
def random_str(n: int = 6,
               lowercase: bool = True,
               uppercase: bool = True,
               digits: bool = True,
               ) -> str:
    """Random string.

    Parameters
    ----------
    n: int, optional
        Number of characters. A string with 6 characters and sample space of 62
        [a-zA-Z0-9] has ~57 billions (62**6) possibilities.
    lowercase: bool, optional
        Use lowercase letters [a-z].
    uppercase: bool, optional
        Use uppercase letters [A-Z].
    digits: bool, optional
        Use digits [0-9].

    Returns
    -------
    out : str
        Random string with `n` charactes.

    """

    if not any((lowercase, uppercase, digits)):
        raise ValueError("At least one option must be True.")

    sample_space: list = []
    if lowercase:
        sample_space += string.ascii_lowercase
    if uppercase:
        sample_space += string.ascii_uppercase
    if digits:
        sample_space += string.digits

    return ''.join(random.choices(sample_space, k=n))
