# pathlib is great
from pathlib import Path
from _pytest.main import Session
import pytest
#from sympy import sin, Symbol, log, exp, zoo, Id, sqrt

# Let's define our failures.txt as a constant as we will need it later
FAILURES_FILE = Path() / "failures.txt"

@pytest.hookimpl()
def pytest_sessionstart(session: Session):
    if FAILURES_FILE.exists():
        # We want to delete the file if it already exists
        # so we don't carry over failures form last run
        FAILURES_FILE.unlink()
    FAILURES_FILE.touch()


@pytest.fixture(scope="module")
def intialize_values_multivariate():
    #symbols = [Symbol("x"),Symbol("y"),Symbol("z"),Symbol("n"),Symbol("p")]
    #basis_functions = [Id,exp,log,sin,sqrt,]
    basis_functions = ["Id","exp","log","sin","sqrt","inv"] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    symbols = ["x","y","z","n","p"]
    return basis_functions, symbols

@pytest.fixture(scope="module")
def intialize_values_multivariate():
    #symbols = [Symbol("x"),Symbol("y"),Symbol("z"),Symbol("n"),Symbol("p")]
    #basis_functions = [Id,exp,log,sin,sqrt,]
    basis_functions = ["Id","exp","log","sin","sqrt","inv"] #Pay attention as the order is indeed important, for testing we put it in alphabetical order (apart from x)
    symbols = ["x","y","z","n","p"]
    return basis_functions, symbols
