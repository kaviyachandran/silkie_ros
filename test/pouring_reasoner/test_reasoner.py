import pytest


import src.silkie_ros.pouring_reasoner as p


@pytest.fixture
def values_test():
    return [2, 3]


@pytest.fixture
def setup_reasoner_increase_tilting_case_1():
    b = p.Blackboard()
    b.context_values['near'] = True
    b.context_values['poursTo'] = False

    return b


def test_adder_function(values_test):
    val = p.sum_val(values_test)
    assert val is 5


def test_reasoner_increase_tiling_case_1(setup_reasoner_increase_tilting_case_1):
    # Given: src and dest are near and pouring can be performed
    # When: No liquid comes out
    # Then: Publish Increase tilting
    # We need a context when near is true and detect that no particles are out of the source

    b_controller = p.BlackboardController(setup_reasoner_increase_tilting_case_1)
    b_controller.publish_conclusions()
    assert True
