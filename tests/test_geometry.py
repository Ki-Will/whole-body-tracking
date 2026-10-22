import pytest
from whole_body_tracking.geometry import box_iou, center
from whole_body_tracking.centering import centering_offset


def test_iou_identical_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert box_iou([0, 0, 2, 2], [3, 3, 5, 5]) == 0.0


def test_center_and_centering():
    assert center([0, 0, 10, 20]) == (5.0, 10.0)
    assert centering_offset([40, 40, 60, 60], 100, 100) == (0.0, 0.0)
