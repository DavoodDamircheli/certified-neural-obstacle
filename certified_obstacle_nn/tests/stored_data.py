"""Skip marker for tests that compare against stored run outputs.

Those outputs (trained checkpoints, frozen partitions, reference data) are distributed in the data archive,
not in the code repository; unpack the archive into the repository root to run these tests.
"""

import pytest

from certified_obstacle_nn.experiments.v14.common import REPO_ROOT


def needs(*paths: str):
    missing = [p for p in paths if not (REPO_ROOT / p).exists()]
    return pytest.mark.skipif(bool(missing), reason="stored data not present: " + ", ".join(missing))
