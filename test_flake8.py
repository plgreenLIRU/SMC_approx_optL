import subprocess

"""
Tests that all scripts obey PEP8 specs.
"""


def test_flake8():

    # Run flake8
    output = subprocess.getoutput("flake8 --ignore=E265,W504,E722,W291")

    # Print output (usually empty, but useful for debugging when
    # the test fails)
    print(output)

    # Test passes is length of output is zero
    assert len(output) == 0
