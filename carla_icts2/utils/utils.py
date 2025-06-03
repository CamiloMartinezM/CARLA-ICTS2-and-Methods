"""Author: Dikshant Gupta
Time: 23.03.21 14:29
"""

import math
import re
from math import cos, radians, sin

import carla
import numpy as np
import torch
import transforms3d


def find_weather_presets():
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters(), strict=False):
        target_param.data.copy_(param.data)


def carla_vec2numpy(vec: carla.Vector3D) -> np.ndarray:
    """Convert a `carla.Vector3D` to a numpy array.

    From: https://github.com/carla-simulator/carla/issues/2915#issue-630393812
    """
    return np.array([vec.x, vec.y, vec.z])


def transform2mat(transform: carla.Transform) -> np.ndarray:
    """Generate the transform 4x4 matrix from a given `carla.Transform`.

    From: https://github.com/carla-simulator/carla/issues/2915#issuecomment-744020598
    """
    location, rotation = transform.location, transform.rotation
    x, y, z = location.x, location.y, location.z
    pitch, yaw, roll = radians(rotation.pitch), radians(rotation.yaw), radians(rotation.roll)

    Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[cos(pitch), 0, -sin(pitch)], [0, 1, 0], [sin(pitch), 0, cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, cos(roll), sin(roll)], [0, -sin(roll), cos(roll)]])
    R = Rz.dot(Ry).dot(Rx)
    t = np.array([[x], [y], [z]])

    M = np.hstack((R, t))
    return np.vstack((M, np.array([[0, 0, 0, 1]])))


def mat2transform(M: np.ndarray) -> carla.Transform:
    """Generate `carla.Transform` from given `M`, a 4x4 matrix.

    From: https://github.com/carla-simulator/carla/issues/2915#issue-630393812
    """
    pitch, roll, yaw = transforms3d.taitbryan.mat2euler(M[0:3, 0:3])
    roll = np.rad2deg(roll)
    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)

    return carla.Transform(
        carla.Location(x=M[0, 3], y=M[1, 3], z=M[2, 3]),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
    )


def relative_transform(source: carla.Transform, target: carla.Transform) -> carla.Transform:
    """Calculate the relative transform from `source` to `target`.

    From: https://github.com/carla-simulator/carla/issues/2915#issuecomment-744020598
    """
    source_t = transform2mat(source)
    target_t = transform2mat(target)
    target_inv = np.linalg.inv(target_t)

    relative_transform_mat = np.dot(target_inv, source_t)

    return mat2transform(relative_transform_mat)
