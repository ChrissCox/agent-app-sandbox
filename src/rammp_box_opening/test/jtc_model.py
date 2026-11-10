"""What the ros2_controllers JTC (humble, 'splines') commands from a
message carrying positions, velocities and accelerations: a quintic
Hermite per interval. Used to assert the EXECUTED acceleration of a
re-timed trajectory, which the executor's gates never look at."""
import numpy as np


def quintic_coeffs(p0, v0, a0, p1, v1, a1, h):
    c3 = (20 * (p1 - p0) - (8 * v1 + 12 * v0) * h - (3 * a0 - a1) * h**2) / (2 * h**3)
    c4 = (-30 * (p1 - p0) + (14 * v1 + 16 * v0) * h + (3 * a0 - 2 * a1) * h**2) / (2 * h**4)
    c5 = (12 * (p1 - p0) - 6 * (v1 + v0) * h + (a1 - a0) * h**2) / (2 * h**5)
    return p0, v0, a0 / 2.0, c3, c4, c5


def peak_accel(msg, nsub=24):
    """Peak |joint acceleration| the spline commands, sampled finely inside
    every interval whatever its length, plus the worst interval index."""
    q = np.asarray([p.positions for p in msg.points])
    v = np.asarray([p.velocities for p in msg.points])
    a = np.asarray([p.accelerations for p in msg.points])
    t = np.asarray([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
    worst, where = 0.0, -1
    for k in range(len(q) - 1):
        h = t[k + 1] - t[k]
        c = quintic_coeffs(q[k], v[k], a[k], q[k + 1], v[k + 1], a[k + 1], h)
        s = np.linspace(0, h, nsub)[:, None]
        aa = 2 * c[2] + 6 * c[3] * s + 12 * c[4] * s**2 + 20 * c[5] * s**3
        m = float(np.abs(aa).max())
        if m > worst:
            worst, where = m, k
    return worst, where


def _interval_peak_speed(c, h):
    """Peak |qd| per joint over [0, h] of one interval's quintic, exactly:
    qd is a quartic, so its extremes are the ends and the real roots of
    the acceleration cubic 2c2 + 6c3 s + 12c4 s^2 + 20c5 s^3 inside."""
    dof = len(np.atleast_1d(c[1]))
    cs = [np.broadcast_to(np.asarray(x, dtype=float), (dof,)) for x in c]
    out = np.empty(dof)
    for j in range(dof):
        c1, c2, c3, c4, c5 = (x[j] for x in cs[1:])
        s = [0.0, h]
        cubic = np.trim_zeros([20 * c5, 12 * c4, 6 * c3, 2 * c2], "f")
        if len(cubic) > 1:
            r = np.roots(cubic)
            s += [float(x.real) for x in r if abs(x.imag) <= 1e-9 * max(1.0, abs(x.real)) and 0.0 < x.real < h]
        s = np.asarray(s)
        out[j] = np.abs(c1 + 2 * c2 * s + 3 * c3 * s**2 + 4 * c4 * s**3 + 5 * c5 * s**4).max()
    return out


def peak_velocity(msg):
    """Peak |joint velocity| the spline commands, found exactly in every
    interval (both ends and every interior extreme). Returns (peak per
    joint (dof,), worst interval index, worst joint index), the worst
    being the largest peak in rad/s; pass caps to peak_velocity_ratio to
    rank against per-joint caps instead."""
    peaks = _per_interval_peaks(msg)
    per_joint = peaks.max(axis=0)
    k, j = np.unravel_index(int(np.argmax(peaks)), peaks.shape)
    return per_joint, int(k), int(j)


def peak_velocity_ratio(msg, caps):
    """Worst executed |qd| / cap over every interval and joint: (ratio,
    interval index, joint index). A joint with an infinite cap never binds."""
    peaks = _per_interval_peaks(msg) / np.asarray(caps, dtype=float)[None, :]
    k, j = np.unravel_index(int(np.argmax(peaks)), peaks.shape)
    return float(peaks[k, j]), int(k), int(j)


def _per_interval_peaks(msg):
    q = np.asarray([p.positions for p in msg.points])
    v = np.asarray([p.velocities for p in msg.points])
    a = np.asarray([p.accelerations for p in msg.points])
    t = np.asarray([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points])
    out = np.zeros((max(len(q) - 1, 0), q.shape[1]))
    for k in range(len(q) - 1):
        h = t[k + 1] - t[k]
        c = quintic_coeffs(q[k], v[k], a[k], q[k + 1], v[k + 1], a[k + 1], h)
        out[k] = _interval_peak_speed(c, h)
    return out
