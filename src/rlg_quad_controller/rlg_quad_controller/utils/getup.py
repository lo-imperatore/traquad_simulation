"""
Rest / stand / activation sequences of the omni_controller (CentroEPiaggio/omni_controller, update_transition),
replicated here so that the same code drives the legs in Gazebo and on the real robot (inference_omniquad_robot).

Controller behaviour reproduced:
  * every segment is a cosine interpolation q = q0 + 0.5 (1 - cos(pi t)) (q1 - q0), t = elapsed / duration in [0, 1];
  * activation: only the HFE joints move, by `activation_hip_offset_deg` from the starting pose, in the direction
    opposite to the stand direction (sign = -1 if q_stand > 0 else +1); the other joints hold the starting pose;
  * stand: all the legs go from where they are to q_stand; rest: to q_rest;
  * wheels locked (velocity 0) during the transitions (done by the caller).
"""
import numpy as np


def cosine_interp(q0, q1, t):
    t = float(np.clip(t, 0.0, 1.0))
    return q0 + 0.5 * (1.0 - np.cos(np.pi * t)) * (q1 - q0)


class Sequence:
    """Chain of cosine segments [(name, q_from, q_to, duration)]; at(t) -> (q, segment name, done)."""

    def __init__(self, segments):
        self.segments = [(n, np.asarray(a, float), np.asarray(b, float), float(d)) for n, a, b, d in segments]
        self.total = sum(d for *_, d in self.segments)
        assert self.segments, 'empty sequence'

    def at(self, t):
        done = t >= self.total
        for i, (name, q0, q1, dur) in enumerate(self.segments):
            if t < dur or i == len(self.segments) - 1:
                return cosine_interp(q0, q1, t / dur if dur > 0 else 1.0), name, done
            t -= dur


def stand_sequence(q_start, q_stand, hfe_mask, activation_offset_deg=0.0, activation_time=0.0, stand_time=5.0):
    """activation (HFE raised from the starting pose) then stand (all the legs to q_stand).
    hfe_mask: boolean array, True for the HFE joints. Activation is skipped if offset or time is 0."""
    q_start, q_stand = np.asarray(q_start, float), np.asarray(q_stand, float)
    segs, q_from = [], q_start
    if activation_time > 0.0 and abs(activation_offset_deg) > 1e-9:
        sign = np.where(q_stand > 0.0, -1.0, 1.0)
        q_act = np.where(hfe_mask, q_start + sign * np.deg2rad(activation_offset_deg), q_start)
        segs.append(('activation', q_start, q_act, activation_time))
        q_from = q_act
    segs.append(('stand', q_from, q_stand, stand_time))
    return Sequence(segs)


def rest_sequence(q_start, q_rest, rest_time=5.0):
    return Sequence([('rest', q_start, q_rest, rest_time)])
