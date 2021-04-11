import pytest
import numpy as np
from src import utils

@pytest.fixture
def fix_route():
    return np.array([0, 1, 2])

@pytest.fixture
def fix_distance():
    return np.array([[0, 3, 2],
                     [3, 0, 3],
                     [2, 3, 0]])

@pytest.fixture
def fix_Q():
    return np.array([[0, 3, 2],
                     [3, 0, 3],
                     [2, 3, 0]])

@pytest.fixture
def fix_mask():
    return np.array([True, True, True])
                    

def test_distance(fix_route, fix_distance):
    """ Test for route distance computation.

    Args:
        fix_route : fixture, example route
        fix_distance : fixture, example distance matrix
    """
    assert utils.route_distance(fix_route, fix_distance) == 8

def test_custom_argmax(fix_mask, fix_Q):
    """ Test for custom argmax function.

    Args:
        fix_mask : fixture, mask used in the argmax operator.
        fix_Q ([type]): fixture, example Q table.
    """
    assert utils.custom_argmax(fix_Q, 0, fix_mask) == 1
    assert utils.custom_argmax(fix_Q, 1, fix_mask) in [0, 2]
    assert utils.custom_argmax(fix_Q, 2, fix_mask) == 1


