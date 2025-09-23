import sys
import os


class DataError(RuntimeError):
    """ Raise error related to loading data. """
    pass

class NotConvergedError(RuntimeError):
    """ Raise error related to convergence. """
    pass
