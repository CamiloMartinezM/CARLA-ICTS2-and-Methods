"""Utility functions for the CARLA ICTS2 benchmark environment."""

import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import carla
import torch

from carla_icts2.config import logger


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


def normalize_angle(angle: float) -> float:
    """Normalize angle to be within [-180, 180]."""
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def round_dict_values(
    data: Mapping[Any, Iterable[float]],
    round_func: Callable[[float], float] = int,
) -> dict:
    """Round the numeric elements of iterable values in a dictionary.

    Parameters
    ----------
    data : Mapping[Any, Iterable[float]]
        A dictionary where each key maps to an iterable (e.g. tuple, list) of numeric (float) values.

    round_func : Callable[[float], float], optional
        A function that takes a float and returns a rounded value.
        Default is `int`, which truncates decimals.

    Returns
    -------
    dict
        A dictionary with the same keys as the input, where each iterable of floats
        has been transformed to a tuple of rounded values using `round_func`.

    Raises
    ------
    TypeError:
        If any of the values in the dictionary are not iterable or contain non-numeric elements.

    Examples
    --------
    >>> poses = {'hip': (12.3, 45.6, 78.9), 'knee': [90.4, 12.6, 5.5]}
    >>> round_dict_values(poses)
    {'hip': (12, 45, 78), 'knee': (90, 12, 5)}

    >>> import math
    >>> round_dict_values(poses, round_func=math.floor)
    {'hip': (12, 45, 78), 'knee': (90, 12, 5)}
    """
    result = {}
    for key, values in data.items():
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise TypeError(f"Value for key '{key}' is not a valid iterable of floats.")

        try:
            rounded_values = tuple(round_func(v) for v in values)
        except Exception as e:
            raise TypeError(f"Error rounding values for key '{key}': {e}") from e

        result[key] = rounded_values

    return result


def trigger_warn_or_error(
    variable: Any,  # noqa: ANN401
    variable_name: str = "",
    src: str = "",
    *,
    warn: bool = True,
    raise_error: bool = False,
) -> bool:
    """Trigger a warning or error based on the state of a variable.

    Args:
        variable: The variable to check (e.g., `walker`, `player`)
        variable_name (str): Name of the variable for logging purposes.
        src (str): Optional source string to prepend to the log message.
        warn (bool): If True, log a warning instead of raising an error.
        raise_error (bool): If True, raise a ValueError if the variable is None or not valid.

    Returns:
        bool: True if the variable is valid (not None and alive), False if it was None or not
            alive and `warn` was True.

    Raises:
        ValueError: If `variable is None` or `not variable.is_alive` and `raise_error` is True.
    """
    warn = not raise_error  # Avoid unnecessary warn if raise_error = True

    if not variable_name:
        if isinstance(variable, str):
            variable_name = variable
        elif hasattr(variable, "__class__"):
            variable_name = f"{variable.__class__.__name__}"
        else:
            variable_name = f"{variable}"

    prefix = f"{src} called but " if src else ""
    e1 = e2 = ""
    if not variable:
        e1 = f"{prefix}{variable_name} is None."

    if variable is not None and not variable.is_alive:
        e2 = f"{prefix}{variable_name} is not alive."

    no_warn = True
    for e in [e1, e2]:
        if e:
            if warn:
                logger.warning(e)
                no_warn = False
            if raise_error:
                raise ValueError(e)

    return no_warn
