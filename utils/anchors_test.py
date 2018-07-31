""" unit test for function meshgrid
"""
from unittest import TestCase, main

from anchors import meshgrid
import numpy as np

import keras.backend as K

class UnitTestCase(TestCase):
    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    def setUp(self):
        self.x = np.array([33, 22.22, 0], dtype=np.float32)
        self.y = np.array([1, 2, 3, 4], dtype=np.float32)
    
    def test_meshgrid(self):
        print("3333")
        x, y = np.meshgrid(self.x, self.y)
        print(x)
        print(y)
        self.assertEqual(np.meshgrid(self.x, self.y), meshgrid(K.variable(self.x), K.variable(self.y)))



if __name__ == '__main__':
    main()