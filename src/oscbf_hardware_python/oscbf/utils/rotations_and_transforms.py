"""Tools for working with rotations and transforms"""

from typing import Union

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np


def xyzw_to_rotation(quat: ArrayLike) -> Array:
    """Converts XYZW quaternions to a rotation matrix

    Args:
        quat (npt.ArrayLike): XYZW quaternions

    Returns:
        np.ndarray: (3,3) rotation matrix
    """
    quat = jnp.asarray(quat)
    quat /= jnp.linalg.norm(quat)
    x, y, z, w = quat
    x2 = x * x
    y2 = y * y
    z2 = z * z
    return jnp.array(
        [
            [1 - 2 * y2 - 2 * z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x2 - 2 * z2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x2 - 2 * y2],
        ]
    )


def xyzw_to_rotation_numpy(quat: ArrayLike) -> np.ndarray:
    """Converts XYZW quaternions to a rotation matrix

    Args:
        quat (npt.ArrayLike): XYZW quaternions

    Returns:
        np.ndarray: (3,3) rotation matrix
    """
    quat = np.asarray(quat)
    quat /= np.linalg.norm(quat)
    x, y, z, w = quat
    x2 = x * x
    y2 = y * y
    z2 = z * z
    return np.array(
        [
            [1 - 2 * y2 - 2 * z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x2 - 2 * z2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x2 - 2 * y2],
        ]
    )


def invert_transform(transform: ArrayLike) -> Array:
    """Inverts a transformation matrix

    Args:
        transform (ArrayLike): (4,4) transformation matrix

    Returns:
        Array: (4,4) inverted transformation matrix
    """
    transform = jnp.asarray(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return jnp.vstack(
        (
            jnp.hstack((rotation.T, (-rotation.T @ translation)[:, None])),
            [0.0, 0.0, 0.0, 1.0],
        )
    )


def invert_transform_numpy(transform: ArrayLike) -> np.ndarray:
    """Inverts a transformation matrix

    Args:
        transform (ArrayLike): (4,4) transformation matrix

    Returns:
        np.ndarray: (4,4) inverted transformation matrix
    """
    transform = np.asarray(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.vstack(
        (
            np.hstack((rotation.T, (-rotation.T @ translation)[:, None])),
            [0.0, 0.0, 0.0, 1.0],
        )
    )


def Rx(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the X axis

    Args:
        theta (float): Angle in radians

    Returns:
        np.ndarray: Rotation matrix, shape (3, 3)
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the Y axis

    Args:
        theta (float): Angle in radians

    Returns:
        np.ndarray: Rotation matrix, shape (3, 3)
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta: float) -> np.ndarray:
    """Rotation matrix for a rotation by theta radians about the Z axis

    Args:
        theta (float): Angle in radians

    Returns:
        np.ndarray: Rotation matrix, shape (3, 3)
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def construct_transform_numpy(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Construct a transformation matrix from a rotation matrix and a translation vector

    Args:
        rotation (np.ndarray): Rotation matrix, shape (3, 3)
        translation (np.ndarray): Translation vector, shape (3,)

    Returns:
        np.ndarray: Transformation matrix, shape (4, 4)
    """
    return np.vstack(
        (
            np.hstack((rotation, translation.reshape(3, 1))),
            [0.0, 0.0, 0.0, 1.0],
        )
    )


def construct_transform(rotation: Array, translation: Array) -> Array:
    """Construct a transformation matrix from a rotation matrix and a translation vector

    Args:
        rotation (Array): Rotation matrix, shape (3, 3)
        translation (Array): Translation vector, shape (3,)

    Returns:
        Array: Transformation matrix, shape (4, 4)
    """
    return jnp.vstack(
        (
            jnp.hstack((rotation, translation.reshape(3, 1))),
            [0.0, 0.0, 0.0, 1.0],
        )
    )


def normalize(vec: ArrayLike) -> Array:
    """Normalizes a vector to have magnitude 1

    If normalizing an array of vectors, each vector will have magnitude 1

    Args:
        vec (ArrayLike): Input vector or array. Shape (dim,) or (n_vectors, dim)

    Returns:
        Array: Unit vector(s), shape (dim,) or (n_vectors, dim) (same shape as the input)
    """
    vec = np.atleast_1d(vec)
    norms = np.linalg.norm(vec, axis=-1)
    return vec / norms[..., np.newaxis]


def skew(v: ArrayLike) -> Array:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (npt.ArrayLike): Vector to convert, shape (3,)

    Returns:
        Array: (3, 3) skew-symmetric matrix
    """
    v = np.ravel(v)
    assert v.shape == (3,)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion derivative for a given world-frame angular velocity

    Args:
        q (np.ndarray): XYZW quaternion, shape (4,)
        omega (np.ndarray): Angular velocity (wx, wy, wz) in world frame, shape (3,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    x, y, z, w = q
    GT = np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])
    return (1 / 2) * GT @ omega


def quaternion_integration(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Propagate a quaternion forward one timestep based on the current angular velocity

    Args:
        q (np.ndarray): Initial XYZW quaternion, shape (4,)
        w (np.ndarray): Angular velocity (wx, wy, wz), shape (3,)
        dt (float): Timestep duration (seconds)

    Returns:
        np.ndarray: Next XYZW quaternion, q(t + dt), shape (4,)
    """
    return normalize(q + dt * quaternion_derivative(q, w))


def rmat_to_quat(rmat: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix into XYZW quaternions

    (computer graphics solution by Shoemake 1994, same as NASA's code)

    Args:
        rmat (np.ndarray): (3,3) rotation matrix

    Returns:
        np.ndarray: XYZW quaternions
    """

    tr = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]
    if tr >= 0:
        s4 = 2.0 * np.sqrt(tr + 1.0)
        x = (rmat[2, 1] - rmat[1, 2]) / s4
        y = (rmat[0, 2] - rmat[2, 0]) / s4
        z = (rmat[1, 0] - rmat[0, 1]) / s4
        w = s4 / 4.0
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
        x = s4 / 4.0
        y = (rmat[0, 1] + rmat[1, 0]) / s4
        z = (rmat[2, 0] + rmat[0, 2]) / s4
        w = (rmat[2, 1] - rmat[1, 2]) / s4
    elif rmat[1, 1] > rmat[2, 2]:
        s4 = 2.0 * np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
        x = (rmat[0, 1] + rmat[1, 0]) / s4
        y = s4 / 4.0
        z = (rmat[1, 2] + rmat[2, 1]) / s4
        w = (rmat[0, 2] - rmat[2, 0]) / s4
    else:
        s4 = 2.0 * np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
        x = (rmat[2, 0] + rmat[0, 2]) / s4
        y = (rmat[1, 2] + rmat[2, 1]) / s4
        z = s4 / 4.0
        w = (rmat[1, 0] - rmat[0, 1]) / s4

    return np.array([x, y, z, w])


def twist_from_transforms(
    tf_prev: np.ndarray, tf_cur: np.ndarray, dt: float
) -> np.ndarray:
    """Determines the twist from two transformation matrices

    Args:
        tf_prev (np.ndarray): Previous transformation matrix, shape (4, 4)
        tf_cur (np.ndarray): Current transformation matrix, shape (4, 4)
        dt (float): Time difference between the two transformations

    Returns:
        np.ndarray: Twist, shape (6,)
    """
    assert dt > 0
    assert tf_prev.shape == (4, 4)
    assert tf_cur.shape == (4, 4)
    R_prev = tf_prev[:3, :3]
    R_cur = tf_cur[:3, :3]
    p_prev = tf_prev[:3, 3]
    p_cur = tf_cur[:3, 3]
    angular_diff = -0.5 * (
        np.cross(R_cur[:, 0], R_prev[:, 0])
        + np.cross(R_cur[:, 1], R_prev[:, 1])
        + np.cross(R_cur[:, 2], R_prev[:, 2])
    )
    omega = angular_diff / dt
    vel = (p_cur - p_prev) / dt
    return np.concatenate([vel, omega])


def integrate_twist(
    transform: np.ndarray, twist: np.ndarray, duration: float
) -> np.ndarray:
    """Determine the new transform after integrating a twist for a specified duration

    Args:
        transform (np.ndarray): Initial transformation matrix, shape (4, 4)
        twist (np.ndarray): Current twist (linear and angular velocity), shape (6,)
        duration (float): Integration time

    Returns:
        np.ndarray: New transformation matrix, shape (4, )
    """
    assert transform.shape == (4, 4)
    assert twist.shape == (6,)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    vel = twist[:3]
    omega = twist[3:]
    xyzw = rmat_to_quat(rotation)
    xyzw_new = quaternion_integration(xyzw, omega, duration)
    new_rotation = xyzw_to_rotation(xyzw_new)
    new_translation = translation + vel * duration
    return construct_transform_numpy(new_rotation, new_translation)


def quaternion_angular_error_numpy(q: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """Gives the instantaneous angular error between two quaternions (q w.r.t q_des)

    - This is similar (but not the same) as a difference between fixed-XYZ conventions
      (for small angles, these are very close).
    - This error is defined in WORLD frame, not the robot's body-fixed frame

    Args:
        q (np.ndarray): Current XYZW quaternion, shape (4,)
        q_des (np.ndarray): Desired XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: Instantaneous angular error, shape (3,)
    """
    x, y, z, w = q
    return 2 * np.array([[-w, z, -y, x], [-z, -w, x, y], [y, -x, -w, z]]) @ q_des


def quaternion_angular_error(q: Array, q_des: Array) -> Array:
    """Gives the instantaneous angular error between two quaternions (q w.r.t q_des)

    - This is similar (but not the same) as a difference between fixed-XYZ conventions
      (for small angles, these are very close).
    - This error is defined in WORLD frame, not the robot's body-fixed frame

    Args:
        q (Array): Current XYZW quaternion, shape (4,)
        q_des (Array): Desired XYZW quaternion, shape (4,)

    Returns:
        Array: Instantaneous angular error, shape (3,)
    """
    x, y, z, w = q
    return 2 * jnp.array([[-w, z, -y, x], [-z, -w, x, y], [y, -x, -w, z]]) @ q_des


# Jax version of the implementation from pytransform3d, with minor modifications
# Optional TODO: make a version where t is an array?
def slerp(start: Array, end: Array, t: float) -> Array:
    """Spherical linear interpolation between two quaternions

    Args:
        start (Array): Starting quaternion, shape (4,)
        end (Array): Ending quaternion, shape (4,)
        t (float): Interpolation parameter, 0 <= t <= 1.

    Returns:
        Array: Interpolated quaternion(s), shape (4,) or (n, 4)
    """
    # This parameter is used to resolve sign ambiguity in the quaternions
    # TODO: Make this an input that is marked as static
    shortest_path = True

    def _pick_closest_quaternion(q: Array, q_target: Array) -> Array:
        distance_flipped = jnp.linalg.norm(-q - q_target)
        distance_normal = jnp.linalg.norm(q - q_target)
        return jnp.where(distance_flipped < distance_normal, -q, q)

    def _angle_between_normalized_vectors(a: Array, b: Array) -> float:
        return jnp.arccos(jnp.clip(jnp.dot(a, b), -1.0, 1.0))

    def _slerp_weights(angle: float, t: float) -> tuple:
        sin_angle = jnp.sin(angle)
        # Linear interpolation for very small angles
        w1_linear = 1.0 - t
        w2_linear = t
        # SLERP weights for larger angles
        w1_slerp = jnp.sin((1.0 - t) * angle) / sin_angle
        w2_slerp = jnp.sin(t * angle) / sin_angle
        # Decide which weights to use based on angle
        small_angle = angle < 1e-6
        w1 = jnp.where(small_angle, w1_linear, w1_slerp)
        w2 = jnp.where(small_angle, w2_linear, w2_slerp)
        return w1, w2

    assert start.shape == (4,)
    assert end.shape == (4,)
    start = start / jnp.linalg.norm(start)
    end = end / jnp.linalg.norm(end)
    if shortest_path:
        end = _pick_closest_quaternion(end, start)
    angle = _angle_between_normalized_vectors(start, end)
    w1, w2 = _slerp_weights(angle, t)
    return w1 * start + w2 * end


def slerp_numpy(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions

    Args:
        start (np.ndarray): Starting quaternion, shape (4,)
        end (np.ndarray): Ending quaternion, shape (4,)
        t (float): Interpolation parameter, 0 <= t <= 1.

    Returns:
        np.ndarray: Interpolated quaternion(s), shape (4,) or (n, 4)
    """
    # This parameter is used to resolve sign ambiguity in the quaternions
    # TODO: Make this an input
    shortest_path = True

    def _pick_closest_quaternion(q: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        if np.linalg.norm(-q - q_target) < np.linalg.norm(q - q_target):
            return -q
        return q

    def _angle_between_normalized_vectors(a: np.ndarray, b: np.ndarray) -> float:
        return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

    def _slerp_weights(angle: float, t: float) -> tuple:
        if angle < 1e-6:
            return 1.0 - t, t
        sin_angle = np.sin(angle)
        w1 = np.sin((1.0 - t) * angle) / sin_angle
        w2 = np.sin(t * angle) / sin_angle
        return w1, w2

    assert start.shape == (4,)
    assert end.shape == (4,)
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    if shortest_path:
        end = _pick_closest_quaternion(end, start)
    angle = _angle_between_normalized_vectors(start, end)
    w1, w2 = _slerp_weights(angle, t)
    return w1 * start + w2 * end
