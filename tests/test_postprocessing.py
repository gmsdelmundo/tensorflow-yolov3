import unittest

import numpy as np

from config import config
from tests import BaseTest
from utils import postprocessing


class TestPostprocessing(BaseTest, unittest.TestCase):
    """Test class for postprocessing functions."""

    def test_postprocessing_iou_complete_overlap(self):
        """Test whether IOU calculation is correct when two boxes completely overlap."""
        x = np.array([0, 0, 20, 20])
        y = np.array([0, 0, 20, 20])
        iou = postprocessing.get_iou(x, y)
        self.assertAlmostEqual(iou, 1)

    def test_postprocessing_iou_partial_overlap(self):
        """Test whether IOU calculation is correct when two boxes partially overlap."""
        x = np.array([0, 0, 20, 20])
        y = np.array([10, 10, 20, 20])
        iou = postprocessing.get_iou(x, y)
        self.assertAlmostEqual(iou, 0.25)

    def test_postprocessing_iou_no_overlap(self):
        """Test whether IOU calculation is correct when two boxes do not overlap."""
        x = np.array([0, 0, 20, 20])
        y = np.array([100, 100, 200, 200])
        iou = postprocessing.get_iou(x, y)
        self.assertGreater(iou, 1)
