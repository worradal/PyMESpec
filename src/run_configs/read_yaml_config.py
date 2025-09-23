#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the file that handles has the yaml processing functions
'''

# Written by
# Alfred Worrad <worrada@udel.edu>,

__author__ = "Afred Worrad"
__version__ = "1.2.1"
__maintainer__ = "Alfred Worrad"
__email__ = "worrada@udel.edu"
__status__ = "Development"
__project__ = "MES wavey"
__date__ = "September 10, 2025"

# built-in modules
from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Tuple,
                    Union)
from itertools import count
import sys
import os
import yaml
# Add the project root to sys.path to allow local imports when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# third-party modules
import matplotlib.pyplot as plt
import numpy as np


class ReadYAMLConfig():
    @staticmethod
    def read_yaml_config(file_path: str) -> Dict:
        """
        Reads a YAML configuration file and returns its contents as a dictionary that works for the OverallRunConfig class.

        Parameters:
        file_path (str): The path to the YAML file.

        Returns:
        dict: The contents of the YAML file as a dictionary.
        """
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
