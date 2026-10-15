"""Where a statement of a test's kernel is, as a refusal names it.

A refusal names each instance of its pair by the statement as the source
writes it and the line it is on, ``y[r] += v[r, j] (test_scatter.py:85) at
r=0, j=0``. A test that pins the line pins it with :func:`where`, read off
the kernel's source, so that a test added above the kernel does not move it.
"""

from __future__ import annotations

import inspect
import os
from typing import Any


def where(fn: Any, text: str) -> str:
    """``file:line`` of the one line of ``fn``'s source that holds ``text``.

    ``fn`` is a function, or a kernel or program, whose body is ``fn.fn``.
    """
    body = inspect.unwrap(getattr(fn, "fn", fn))
    lines, start = inspect.getsourcelines(body)
    (found,) = [k for k, line in enumerate(lines) if text in line]
    return f"{os.path.basename(inspect.getsourcefile(body))}:{start + found}"
