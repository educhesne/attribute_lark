from __future__ import absolute_import, print_function

import unittest
import logging
import sys
from lark import logger

from .test_tree_templates import *  # We define __all__ to list which TestSuites to run

# from .test_selectors import TestSelectors
# from .test_grammars import TestPythonG, TestConfigG


from .test_parser import *  # We define __all__ to list which TestSuites to run

if sys.version_info >= (3, 10):
    pass

logger.setLevel(logging.INFO)

if __name__ == "__main__":
    unittest.main()
