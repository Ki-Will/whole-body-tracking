import math
from whole_body_tracking.identity_state import IdentityEvidence, IdentityState


def test_face_observation_confirms_track():
    evidence = IdentityEvidence()
    evidence.observe_face(1.0, 0.9)
    assert evidence.state is IdentityState.CONFIRMED


def test_confidence_decays_exponentially():
    evidence = IdentityEvidence(decay_lambda=0.5)
    evidence.observe_face(0.0, 1.0)
    assert evidence.confidence(2.0) == math.exp(-1.0)


def test_missing_face_becomes_temporary_then_unknown():
    evidence = IdentityEvidence(decay_lambda=0.2, retention_seconds=2.0)
    evidence.observe_face(0.0, 0.9)
    assert evidence.update(1.0, body_track_valid=True) is IdentityState.TEMPORARY
    assert evidence.update(3.0, body_track_valid=True) is IdentityState.UNKNOWN


def test_never_seen_face_is_unknown():
    evidence = IdentityEvidence()
    assert evidence.update(1.0, True) is IdentityState.UNKNOWN
