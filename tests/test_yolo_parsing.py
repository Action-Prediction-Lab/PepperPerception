"""Unit test: YOLODetector.detect turns ultralytics results into the documented dict shape.

Deterministic, unlike the integration guard, which only sees detections when a probe frame trips one.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from perception_service.detectors.yolo_detector import YOLODetector  # noqa: E402


class _Tensor(list):
    def tolist(self):
        return list(self)


class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [_Tensor(xyxy)]
        self.conf = [conf]
        self.cls = [cls]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Stands in for ultralytics YOLO: callable, and exposes a class-id to name mapping."""

    names = {0: "person", 32: "sports ball"}

    def __init__(self, results):
        self._results = results

    def __call__(self, image, **kwargs):
        return self._results


def _detector(results):
    # Bypass __init__ (no weights are loaded).
    detector = object.__new__(YOLODetector)
    detector.model = _FakeModel(results)
    return detector


def test_detect_returns_documented_fields():
    out = _detector([_Result([_Box([1.0, 2.0, 3.0, 4.0], 0.9, 0)])]).detect(None)
    assert out == [{"class": "person", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}]


def test_detect_returns_an_empty_list_when_nothing_is_found():
    assert _detector([_Result([])]).detect(None) == []


def test_detect_flattens_multiple_results_and_boxes():
    out = _detector([
        _Result([_Box([0.0, 0.0, 1.0, 1.0], 0.5, 0)]),
        _Result([_Box([2.0, 2.0, 3.0, 3.0], 0.25, 32)]),
    ]).detect(None)
    assert [d["class"] for d in out] == ["person", "sports ball"]
    assert all(len(d["bbox"]) == 4 for d in out)


def test_detect_coerces_confidence_and_class_id_types():
    out = _detector([_Result([_Box([0, 0, 1, 1], 0.75, 0)])]).detect(None)
    assert isinstance(out[0]["confidence"], float)
    assert out[0]["class"] == "person"
