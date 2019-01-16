my_script = __import__("main")
print(my_script.tst())

import nose2
import unittest

class TestFeatures(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()