"""Integration guard: the ZMQ REP contract the robot clients depend on.

Needs a running service; `data` varies by backend, so the guard reads the backend from the reply.
"""
from guard import (Guard, blank_image, encode, main, noise_image, request,
                   service_reachable)

try:
    import pytest
    pytestmark = pytest.mark.skipif(not service_reachable(), reason="requires a running service")
except ImportError:
    pass

LANDMARK_KEYS = {"face_landmarks", "left_hand_landmarks",
                 "pose_landmarks", "right_hand_landmarks"}
DETECTION_KEYS = {"class", "confidence", "bbox"}


def check_blank_shape(g, backend, data):
    """A blank frame must yield an empty detection set in whichever shape the backend uses."""
    if backend == "yolo":
        g.check("yolo data is a list", isinstance(data, list), f"{type(data).__name__}")
        g.check("blank frame yields no detections", data == [], f"{data}")
        return
    g.check("data is a dict", isinstance(data, dict), f"{type(data).__name__}")
    if not isinstance(data, dict):
        return
    expected = LANDMARK_KEYS | ({"detections"} if backend == "combined" else set())
    # Clients index these keys directly (a rename is a breaking change).
    g.check(f"{backend} data carries the documented keys", set(data) == expected,
            f"unexpected={sorted(set(data) ^ expected)}")
    if backend == "combined":
        g.check("detections is a list", isinstance(data.get("detections"), list))
        g.check("blank frame yields no detections", data.get("detections") == [],
                f"{data.get('detections')}")
    for key in sorted(LANDMARK_KEYS):
        g.check(f"{key} is None on a blank frame", data.get(key) is None)


def test_service_contract():
    g = Guard("perception contract")

    reply = request([b'{"meta":"x"}', encode(blank_image())])
    g.check("service answers a valid request", reply is not None)
    if reply is None:
        g.finish()
        return

    backend = reply.get("backend")
    g.check("status is success", reply.get("status") == "success", f"{reply.get('status')}")
    g.check("backend is one of the supported three", backend in ("yolo", "mediapipe", "combined"),
            f"{backend}")
    g.check("inference_ms is a positive number",
            isinstance(reply.get("inference_ms"), (int, float)) and reply["inference_ms"] > 0,
            f"{reply.get('inference_ms')}")
    check_blank_shape(g, backend, reply.get("data"))

    # The last frame is the image (a request without metadata work).
    single = request([encode(noise_image())])
    g.check("single-frame request is accepted",
            single is not None and single.get("status") == "success",
            f"{single.get('status') if single else 'timeout'}")

    # Any detection returned must carry the fields clients read. A synthetic frame rarely trips the detector, so this is opportunistic; tests/test_yolo_parsing.py pins the shape.
    if single and backend in ("yolo", "combined"):
        found = single["data"] if backend == "yolo" else single["data"]["detections"]
        if found:
            g.check("detections have the documented fields",
                    all(DETECTION_KEYS <= set(d) for d in found), f"n={len(found)}")
            g.check("bbox is four numbers",
                    all(len(d["bbox"]) == 4 for d in found), f"n={len(found)}")
        else:
            print("[SKIP] detection field shape: probe frame produced none "
                  "(pinned by tests/test_yolo_parsing.py)")

    # An undecodable payload must return a structured error.
    bad = request([b"not-an-image"])
    g.check("undecodable payload returns an error",
            bad is not None and bad.get("status") == "error", f"{bad}")
    if bad:
        g.check("error carries a message", bool(bad.get("message")), f"{bad.get('message')}")

    # The service must still serve after an error, or one bad frame kills the session.
    after = request([encode(blank_image())])
    g.check("service still serves after a bad request",
            after is not None and after.get("status") == "success")

    g.finish()


if __name__ == "__main__":
    main(test_service_contract)
