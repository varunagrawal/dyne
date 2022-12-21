from dyne import utils


def test_get_limits_1():
    limits = (10, 20)
    expected = (9, 22)
    actual = utils.get_limits(limits[0], limits[1])
    assert expected == actual

def test_get_limits_2():
    limits = (-10, 10)
    expected = (-11, 11)
    actual = utils.get_limits(limits[0], limits[1])
    assert expected == actual

def test_get_limits_3():
    limits = (-20, -10)
    expected = (-22, -9)
    actual = utils.get_limits(limits[0], limits[1])
    assert expected == actual
