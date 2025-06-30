# ruff: noqa: D102
"""Pedestrian controller module for CARLA ICTS2 benchmark."""

import abc
import math
import time
from enum import Enum
from typing import Literal

import carla
import numpy as np

from carla_icts2.benchmark.environment.utils import round_dict_values
from carla_icts2.config import logger


class BasePose(abc.ABC):
    """Abstract Base Class for all poses."""

    def __init__(self, walker: carla.Walker) -> None:
        """Initialize the BasePose.

        Args:
            walker (carla.Walker): The walker actor this pose is associated with.
        """
        self.walker = walker
        self.blend_duration = 0.25

    @abc.abstractmethod
    def step(self) -> Literal["Running", "Done", "Idle"]:
        """Execute one simulation step's logic specific for this pose."""
        raise NotImplementedError

    def _apply_arm_pose(
        self,
        arm_bone_targets: dict[str, tuple[float, float, float]],
        *,
        relative: bool = True,
    ) -> dict[str, tuple[float, float, float]]:
        """Apply a target pose to the arm bones of the walker.

        Args:
            arm_bone_targets (dict[str, tuple[float, float, float]]): A dict like
                ```
                {
                    "crl_arm__R": (pitch, roll, yaw),
                    "crl_shoulder__R": (pitch, roll, yaw),
                    ...
                }
                ```
                If a bone is not in the dict, its current relative transform is maintained.
            relative (bool, optional): If True, the specified rotations are applied relative to the
                current pose. If False, they are set as absolute values. Default is True.

        Returns:
            (dict[str, tuple[float, float, float]]): A dictionary mapping each arm bone name to its
            updated absolute (pitch, roll, yaw) rotation.
        """
        bones = self.walker.get_bones()
        new_pose = []
        updated_pose = {}
        for bone in bones.bone_transforms:
            if bone.name in arm_bone_targets:
                pitch, roll, yaw = arm_bone_targets[bone.name]
                if pitch is not None:
                    if relative:
                        bone.relative.rotation.pitch += pitch
                    else:
                        bone.relative.rotation.pitch = pitch
                if roll is not None:
                    if relative:
                        bone.relative.rotation.roll += roll
                    else:
                        bone.relative.rotation.roll = roll
                if yaw is not None:
                    if relative:
                        bone.relative.rotation.yaw += yaw
                    else:
                        bone.relative.rotation.yaw = yaw

                updated_pose[bone.name] = (
                    bone.relative.rotation.pitch,
                    bone.relative.rotation.roll,
                    bone.relative.rotation.yaw,
                )

            new_pose.append((bone.name, bone.relative))

        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(self.blend_duration)
        return updated_pose


class PathController:
    # Adapted from scenario runner
    def __init__(self, world, walker, path, target_speed, speed_schedule=None):
        self.world = world
        self.walker = walker
        self.path = path
        self.target_speed = target_speed
        self.cur_speed = target_speed
        self.speed_schedule = speed_schedule
        self.done = False

    def step(self):
        if self.done:
            return "Done"
        actor_location = self.walker.get_location()
        if self.path:
            location = self.path[0]
            direction = location - actor_location
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            control = self.walker.get_control()
            control.speed = self._get_speed()
            control.direction = direction / direction_norm
            self.walker.apply_control(control)
            if direction_norm < 0.2:
                self.path = self.path[1:]
                if len(self.path) == 0:
                    self.done = True
                    return "Done"
        else:
            control = self.walker.get_control()
            control.speed = self.cur_speed
            control.direction = self.walker.get_transform().rotation.get_forward_vector()
            self.walker.apply_control(control)
        return "Running"

    def set_walker_speed_relative(self, rel):
        self.cur_speed *= rel

    def _get_speed(self):
        if self.speed_schedule is None:
            return self.cur_speed
        if len(self.speed_schedule) == 0:
            return self.cur_speed
        location, value = self.speed_schedule[0]
        actor_location = self.walker.get_location()
        distance = l2_distance(location, actor_location)
        # print(distance)
        if distance < 0.2:
            self.cur_speed *= value
            self.speed_schedule = self.speed_schedule[1:]
            # print(self.cur_speed)
        return self.cur_speed

    def set_done(self):
        self.done = True


class WaveHand(BasePose):
    """A controller to make the walker wave their hand."""

    def __init__(
        self,
        walker: carla.Walker,
        start_pos: carla.Location | None = None,
        raise_duration: float = 0.75,
        wave_cycles: int = 2,
        wave_segment_duration: float = 0.3,
        lower_duration: float = 0.5,
    ) -> None:
        super().__init__(walker)

        self.walker = walker
        self.start_pos = start_pos
        self.raise_duration = raise_duration
        self.wave_cycles = wave_cycles
        self.wave_segment_duration = wave_segment_duration
        self.lower_duration = lower_duration

        self.state = "Idle"
        self.start_time = None
        self.current_wave_count = 0

        # Stores original pose before any action
        self._initial_arm_pose_bones: dict[str, carla.Transform] = {}

        # Define target bone rotations for "raised" state consistently
        self.raised_arm_pitch = -140  # More upright raise
        self.raised_arm_roll = 10
        self.raised_shoulder_pitch = -1
        self.raised_shoulder_roll = 20
        self.raised_forearm_pitch = -10  # Bent elbow for waving
        self.raised_forearm_yaw = 0  # Forearm straight initially when raised

    def _store_initial_arm_pose(self) -> None:
        """Store the initial pose of the arm bones for later reset."""
        self._initial_arm_pose_bones.clear()
        bones = self.walker.get_bones()
        for bone_transform in bones.bone_transforms:
            # Store relevant bones for a complete reset
            if (
                "arm__R" in bone_transform.name
                or "shoulder__R" in bone_transform.name
                or "foreArm__R" in bone_transform.name
                or "hand__R" in bone_transform.name
            ):  # Optional: include hand
                relative_transform = bone_transform.relative
                self._initial_arm_pose_bones[bone_transform.name] = carla.Transform(
                    location=carla.Location(
                        x=relative_transform.location.x,
                        y=relative_transform.location.y,
                        z=relative_transform.location.z,
                    ),
                    rotation=carla.Rotation(
                        pitch=relative_transform.rotation.pitch,
                        yaw=relative_transform.rotation.yaw,
                        roll=relative_transform.rotation.roll,
                    ),
                )

    def _apply_arm_pose(self, target_pitches_rolls_yaws: dict[str, tuple[float, float, float]], blend_duration: float) -> None:
        """Apply a target pose to the arm bones.

        Args:
            target_pitches_rolls_yaws: A dict like {
                "crl_arm__R": (pitch, roll, yaw),
                "crl_shoulder__R": (pitch, roll, yaw),
                ...
            }
            If a bone is not in the dict, its current relative transform is maintained.
            blend_duration: Duration to blend into this pose.
        """
        bones = self.walker.get_bones()
        new_pose = []
        for bone_transform in bones.bone_transforms:
            name = bone_transform.name
            # Start with the current relative transform as a base
            current_relative_transform = carla.Transform(
                location=bone_transform.relative.location,
                rotation=bone_transform.relative.rotation,
            )

            if name in target_pitches_rolls_yaws:
                pitch, roll, yaw = target_pitches_rolls_yaws[name]
                if pitch is not None:
                    current_relative_transform.rotation.pitch = pitch
                if roll is not None:
                    current_relative_transform.rotation.roll = roll
                if yaw is not None:
                    current_relative_transform.rotation.yaw = yaw

            new_pose.append((name, current_relative_transform))

        control = carla.WalkerBoneControlIn(new_pose)
        self.walker.set_bones(control)
        self.walker.blend_pose(blend_duration)

    def step(self) -> Literal["Running", "Done", "Idle"]:
        if self.state == "Done":
            return "Done"

        current_loc = self.walker.get_location()
        current_time = time.time()

        if self.state == "Idle":
            if self.start_pos is None or l2_distance(current_loc, self.start_pos) <= 0.5:
                self.state = "Raising"
                self.start_time = current_time
                self._store_initial_arm_pose()
                logger.debug("WaveHand triggered: Raising arm.")

                raise_pose_targets = {
                    "crl_arm__R": (
                        self.raised_arm_pitch,
                        self.raised_arm_roll,
                        None,
                    ),  # Yaw not set yet
                    "crl_shoulder__R": (
                        self.raised_shoulder_pitch,
                        self.raised_shoulder_roll,
                        None,
                    ),
                    "crl_foreArm__R": (
                        self.raised_forearm_pitch,
                        None,
                        self.raised_forearm_yaw,
                    ),  # roll not set
                }
                self._apply_arm_pose(raise_pose_targets, self.raise_duration)
            else:
                return "Running"

        elif self.state == "Raising":
            if self.start_time is None:
                raise ValueError(
                    "WaveHand: start_time is None in Raising state. This should not happen.",
                )

            if (
                current_time - self.start_time >= self.raise_duration and self.wave_cycles > 0
            ):  # Wait for raise to complete
                self.state = "Waving_Out"
                self.start_time = current_time
                self.current_wave_count = 0
                logger.debug("WaveHand: Arm raised, starting wave (outward).")
                self._apply_wave_segment_pose(outward=True)

        elif self.state == "Waving_Out":
            if self.start_time is None:
                raise ValueError(
                    "WaveHand: start_time is None in Waving_Out state. This should not happen.",
                )

            # Check if the blend for Waving_Out is complete before moving to Waving_In
            if (
                current_time - self.start_time >= self.wave_segment_duration
                and self.wave_cycles > 0
            ):
                self.state = "Waving_In"
                self.start_time = current_time
                logger.debug("WaveHand: Waving inward.")
                self._apply_wave_segment_pose(outward=False)

        elif self.state == "Waving_In":
            if self.start_time is None:
                raise ValueError(
                    "WaveHand: start_time is None in Waving_In state. This should not happen.",
                )

            # Check if the blend for Waving_In is complete
            if current_time - self.start_time >= self.wave_segment_duration:
                self.current_wave_count += 1
                if self.current_wave_count >= self.wave_cycles:
                    self.state = "Lowering"
                    self.start_time = current_time
                    logger.debug("WaveHand: Waving finished, lowering arm.")

                    if self._initial_arm_pose_bones:
                        lower_pose_targets = {}
                        # Get relevant bones from stored initial pose
                        for bone_name in [
                            "crl_arm__R",
                            "crl_shoulder__R",
                            "crl_foreArm__R",
                            "crl_hand__R",
                        ]:
                            if bone_name in self._initial_arm_pose_bones:
                                t = self._initial_arm_pose_bones[bone_name]
                                lower_pose_targets[bone_name] = (
                                    t.rotation.pitch,
                                    t.rotation.roll,
                                    t.rotation.yaw,
                                )
                        self._apply_arm_pose(lower_pose_targets, self.lower_duration)
                    else:
                        self.walker.blend_pose(
                            self.lower_duration
                        )  # Fallback to default animation
                else:
                    self.state = "Waving_Out"
                    self.start_time = current_time
                    logger.debug(
                        f"WaveHand: Starting wave cycle {self.current_wave_count + 1} (outward)."
                    )
                    self._apply_wave_segment_pose(outward=True)

        elif self.state == "Lowering":
            if self.start_time is None:
                raise ValueError(
                    "WaveHand: start_time is None in Lowering state. This should not happen.",
                )

            if current_time - self.start_time >= self.lower_duration:
                self.state = "Done"
                logger.debug("WaveHand: Arm lowered.")

        return "Running"

    def _apply_wave_segment_pose(self, *, outward: bool) -> None:
        wave_forearm_yaw_amplitude = 45

        wave_pose_targets = {
            "crl_arm__R": (self.raised_arm_pitch, self.raised_arm_roll, None),
            "crl_shoulder__R": (self.raised_shoulder_pitch, self.raised_shoulder_roll, None),
            "crl_foreArm__R": (
                self.raised_forearm_pitch,
                None,  # Keep current roll or define one
                wave_forearm_yaw_amplitude if outward else -wave_forearm_yaw_amplitude,
            ),
        }
        # Blend into this wave segment pose. The duration here is how long it takes
        # to reach this extreme of the wave.
        self._apply_arm_pose(wave_pose_targets, self.wave_segment_duration * 0.8)  # Blend quickly


class SimplifiedWave(BasePose):
    """A very simple waving controller that alternates between two predefined raised-arm poses.

    - It ONLY sets the arm bones.
    - The waving effect is achieved by repeatedly calling its step() method, which alternates the
      target pose for the arm.
    """

    def __init__(
        self,
        walker: carla.Walker,
        blend_duration: float = 2.0,
        *,
        start_waving_inmediately: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the SimplifiedWave controller."""
        super().__init__(walker)
        self.walker = walker
        self.blend_duration = blend_duration
        self.start_waving_inmediately = start_waving_inmediately
        self.currently_pose1 = True  # Start by aiming for pose 1 when first activated
        self.debug = debug  # Enable/disable debug logging

        # Store target rotations for each pose. Only arm bones are defined here.
        self.raise_arm_pose1 = {
            "crl_arm__R": (-120.0, 0.0, 0.0),
            "crl_shoulder__R": (-1.0, 20.0, 0.0),
            "crl_foreArm__R": (-10.0, 40.0, 0.0),
        }
        self.raise_arm_pose2 = {
            "crl_arm__R": (-110.0, 0.0, 0.0),
            "crl_shoulder__R": (-1.0, 20.0, -10.0),
            "crl_foreArm__R": (-10.0, 40.0, 0.0),
        }

        # Stored ABSOLUTE (pitch, roll, yaw) of the bones updated in raise_arm_pose1
        # This is filled inside self.start_waving()
        self.stored_after_raise_arm_pose1: dict[str, tuple[float, float, float]] = {}

        # Stored ABSOLUTE (pitch, roll, yaw) of the bones updated in raise_arm_pose2
        # This is filled inside self.step()
        self.stored_after_raise_arm_pose2: dict[str, tuple[float, float, float]] = {}

        self.is_active = False  # Controller is activated by an external call to start_waving
        self.last_blend_initiated_time = 0.0

    def start_waving(self) -> None:
        """Call this once to begin the waving sequence."""
        self.is_active = True
        self.currently_pose1 = True  # Ensure it starts by targeting Pose 1
        self.stored_after_raise_arm_pose1 = self._apply_arm_pose(
            self.raise_arm_pose1,
            relative=True,
        )
        if self.debug:
            logger.debug(
                f"SimplifiedWave: Waving started, applying Pose 1 and got "
                f"{round_dict_values(self.stored_after_raise_arm_pose1)}",
            )

    def stop_waving(self) -> None:
        """Call this to stop the controller from applying new poses.

        The arm will remain in its last blended-to state. An external ResetPose controller is
        needed to lower it.
        """
        self.is_active = False
        if self.debug:
            logger.debug("SimplifiedWave: Waving stopped. Arm left in last pose.")

    def step(self) -> Literal["Done", "Idle"]:
        """Execute one simulation step's logic for waving.

        If active, and enough time has passed since the last blend was initiated, this method will
        apply the *other* wave pose.

        It should be called repeatedly by the world tick during the waving duration.
        """
        if self.start_waving_inmediately:
            self.start_waving()

        if not self.is_active:
            return "Idle"  # Or "Done" if preferred for an inactive state

        if self.currently_pose1:
            # Was targeting Pose 1, now target Pose 2
            if self.stored_after_raise_arm_pose2:
                self._apply_arm_pose(self.stored_after_raise_arm_pose2, relative=False)
            else:
                self.stored_after_raise_arm_pose2 = self._apply_arm_pose(
                    self.raise_arm_pose2,
                    relative=True,
                )

            if self.debug:
                logger.debug(
                    "SimplifiedWave: Blending from (towards) Pose 1 to Pose 2 and got "
                    f"current absolute pose {round_dict_values(self.stored_after_raise_arm_pose2)}",
                )
            self.currently_pose1 = False
        else:
            # Was targeting Pose 2, now target Pose 1
            if self.stored_after_raise_arm_pose1:
                self._apply_arm_pose(self.stored_after_raise_arm_pose1, relative=False)
            else:
                self._apply_arm_pose(self.raise_arm_pose1, relative=True)

            if self.debug:
                logger.debug(
                    "SimplifiedWave: Blending from (towards) Pose 2 to Pose 1 and got "
                    f"current absolute pose {round_dict_values(self.stored_after_raise_arm_pose1)}",
                )

            self.currently_pose1 = True

        return "Done"


class LookAcrossStreetLeft(object):
    """
    Controller to make the walker look left (across the street, towards +X),
    assuming the walker's body is facing along the curb (-Y, Yaw=180).
    """

    def __init__(self, walker, start_pos=None, duration=0.5):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False
        self.duration = duration
        self.start_time = None
        self.state = "Idle"

    def step(self):
        if self.done:
            return "Done"

        # Trigger condition based on proximity to start_pos
        # Use l2_distance for accurate proximity check
        current_loc = self.walker.get_location()
        # Check if start_pos is valid before calculating distance
        if (
            self.state == "Idle"
            and self.start_pos is not None
            and l2_distance(current_loc, self.start_pos) <= 0.3
        ):  # Increased threshold slightly
            # Triggered - Start Looking
            self.state = "Looking"
            self.start_time = time.time()
            logger.debug("LookAcrossStreetLeft triggered.")  # Add log

            bones = self.walker.get_bones()
            new_pose = []
            for bone in bones.bone_transforms:
                # Walker facing Yaw=180 (-Y). We want to look towards +X (90 deg right turn).
                # Need negative yaw relative to forward direction.
                if bone.name == "crl_neck__C":
                    # (+) Makes him look to his left, (-) to his right
                    bone.relative.rotation.pitch += 40

                    # (-) Twists the neck to the right w.r.t to the body's vertical line
                    # and the pedestrian looking to the front
                    # bone.relative.rotation.yaw -= 80
                    new_pose.append((bone.name, bone.relative))
                elif bone.name == "crl_Head__C":
                    # (+) Tilts the head to his left, (-) to his right
                    bone.relative.rotation.pitch += 25

                    # (+) Looks up, (-) looks down
                    # bone.relative.rotation.roll -= 5

                    # (-) Twists the head to the right w.r.t to the body's vertical line
                    # and the pedestrian looking to the front
                    # bone.relative.rotation.yaw -= 10
                    new_pose.append((bone.name, bone.relative))
                # elif bone.name == "crl_spine01__C":
                #     # (+) Twists spine forward, (-) backward
                #     bone.relative.rotation.roll += 25
                #     new_pose.append((bone.name, bone.relative))
                else:
                    new_pose.append((bone.name, bone.relative))

            control = carla.WalkerBoneControlIn()
            control.bone_transforms = new_pose
            self.walker.set_bones(control)
            self.walker.blend_pose(self.duration)

        elif self.state == "Looking":
            # Check if duration has passed
            if time.time() - self.start_time >= self.duration:
                self.done = True  # Mark as done, actual reset handled externally by ResetPose
                self.state = "Done"
                logger.debug("LookAcrossStreetLeft finished blending.")  # Add log
                return "Done"
        # else if state is "Idle" and trigger condition not met, or state is "Done"
        # just return "Running" or "Done" respectively
        elif self.state == "Done":
            return "Done"

        return "Running"  # Controller is active (Idle or Looking)


class RaiseHandBriefly(object):
    """
    Controller to make the walker briefly raise their right hand.
    """

    def __init__(self, walker, start_pos=None, raise_duration=0.3, hold_duration=0.5):
        self.walker = walker
        self.start_pos = start_pos
        self.raise_duration = raise_duration
        self.hold_duration = hold_duration
        self.state = "Idle"  # Idle, Raising, Holding, Lowering, Done
        self.start_time = None
        self.initial_bones = None  # To store initial pose for lowering

    def step(self):
        if self.state == "Done":
            return "Done"

        # Trigger condition
        if self.state == "Idle" and self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.2:
                return "Running"  # Not yet triggered
            else:
                # Triggered - Start Raising
                self.state = "Raising"
                self.start_time = time.time()
                # Store initial pose of relevant bones if needed for smooth lowering
                # self.initial_bones = self._get_initial_arm_bones() # Implement this if needed

                bones = self.walker.get_bones()
                new_pose = []
                # for bone in bones.bone_transforms:
                #     # Modify Right Arm bones
                #     if bone.name == "crl_arm__R":
                #         bone.relative.rotation.pitch -= 70  # Raise arm forward/up
                #         bone.relative.rotation.roll += 20  # Slight outward roll
                #         new_pose.append((bone.name, bone.relative))
                #     elif bone.name == "crl_shoulder__R":
                #         bone.relative.rotation.pitch -= 10  # Adjust shoulder
                #         new_pose.append((bone.name, bone.relative))
                #     # elif bone.name == "crl_foreArm__R":
                #     #     bone.relative.rotation.pitch += 0 # Keep forearm straight initially
                #     #     new_pose.append((bone.name, bone.relative))
                #     else:
                #         new_pose.append((bone.name, bone.relative))
                for bone in bones.bone_transforms:
                    if bone.name == "crl_arm__R":
                        bone.relative.rotation.pitch -= 45
                        #     bone.relative.rotation.roll = 90
                        #     # bone.relative.rotation.yaw = 0
                        new_pose.append((bone.name, bone.relative))
                    if bone.name == "crl_shoulder__R":
                        bone.relative.rotation.pitch -= -1
                        bone.relative.rotation.roll += 20
                        # bone.relative.rotation.yaw = 90
                        new_pose.append((bone.name, bone.relative))
                    if bone.name == "crl_foreArm__R":
                        bone.relative.rotation.pitch -= 10
                        bone.relative.rotation.roll += 40
                        # bone.relative.rotation.yaw = -45
                        new_pose.append((bone.name, bone.relative))
                    else:
                        new_pose.append((bone.name, bone.relative))

                control = carla.WalkerBoneControlIn()
                control.bone_transforms = new_pose
                self.walker.set_bones(control)
                self.walker.blend_pose(self.raise_duration)

        elif self.state == "Raising":
            if time.time() - self.start_time >= self.raise_duration:
                self.state = "Holding"
                self.start_time = time.time()  # Reset timer for hold duration

        elif self.state == "Holding":
            if time.time() - self.start_time >= self.hold_duration:
                self.state = "Lowering"
                self.start_time = time.time()
                # Reset to default pose smoothly
                # Using blend_pose(0) effectively tells CARLA to return to the base animation
                self.walker.blend_pose(
                    0
                )  # Start blending back to default over ~0.5s (default blend time)

        elif self.state == "Lowering":
            # We rely on the default blend_pose(0) to finish.
            # We can estimate completion or just mark as done after a short delay.
            # Let's assume the default blend takes about 0.5s
            if time.time() - self.start_time >= 0.5:
                self.state = "Done"

        return "Running"  # Controller is active


class LookBehindRight:
    def __init__(self, walker, start_pos, char, scenario="standard"):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False
        if char == "forcing":
            self.spine_roll = 60
        else:
            self.spine_roll = 20
        self.icr_value = ICR.PLANNING_TO if scenario == "standard" else ICR.PLANNING_TO

    def step(self):
        if self.done:
            return "Done"
        direction = self.walker.get_location() - self.start_pos
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm > 0.1:
            return "Running"
        self.walker.icr = self.icr_value
        self.walker.son = self.walker.initial_son
        self.walker.var = 10
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_hips__C":  # Added new
                bone.relative.rotation.pitch += 40  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += self.spine_roll  # Added new
                bone.relative.rotation.pitch += 40
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine01__C":
                bone.relative.rotation.pitch += 90  # Changed from 50 to 13
                new_pose.append((bone.name, bone.relative))
            if bone.name == "crl_neck__C":
                bone.relative.rotation.pitch -= 90  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch += 90  # Changed from 40 to 13
                bone.relative.rotation.roll -= 20  # added new
                bone.relative.rotation.yaw -= 0  # added new
                new_pose.append((bone.name, bone.relative))
            else:
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)
        self.done = True
        return "Done"


class LookBehindLeftSpine:
    # Quick fix should be merged with LookBehindLeft
    def __init__(self, walker, start_pos, char):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False
        if char == "forcing":
            self.spine_roll = 60
        else:
            self.spine_roll = 20

    def step(self):
        if self.done:
            return "Done"
        direction = self.walker.get_location() - self.start_pos
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm > 0.1:
            return "Running"
        self.walker.var = 10
        self.walker.icr = ICR.PLANNING_TO
        self.walker.son = SON.FORCING
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_hips__C":  # Added new
                bone.relative.rotation.pitch -= 40  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += self.spine_roll  # Added new
                bone.relative.rotation.pitch -= 40
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine01__C":
                bone.relative.rotation.pitch -= 90  # Changed from 50 to 13
                new_pose.append((bone.name, bone.relative))
            if bone.name == "crl_neck__C":
                bone.relative.rotation.pitch -= 90  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch -= 90  # Changed from 40 to 13
                bone.relative.rotation.roll -= 20  # added new
                bone.relative.rotation.yaw -= 0  # added new
                new_pose.append((bone.name, bone.relative))
            else:
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)
        self.sone = True
        return "Done"


class RaiseArm:
    """Raises Arm.

    Designed for scenarios where the pedestrian walks past the `start_pos` to trigger the arm raise
    and continues moving until it walks past end_pos to stop it. It's not designed for a stationary
    gesture.
    """

    def __init__(self, walker, start_pos, char, end_pos):
        self.walker = walker
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.done = False
        if char == "forcing":
            self.spine_roll = 60
        else:
            self.spine_roll = 20
        self.head_roll = 40

    def step(self):
        if self.done:
            return "Done"
        direction = self.walker.get_location() - self.start_pos
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm < 0.2:
            return "Running"

        direction_end = self.walker.get_location() - self.end_pos
        direction_norm_end = math.sqrt(direction_end.x**2 + direction_end.y**2)
        if direction_norm_end < 0.2:
            self.walker.blend_pose(0)
            self.done = True
            return "Done"

        bones = self.walker.get_bones()
        new_pose = []

        for bone in bones.bone_transforms:
            if bone.name == "crl_arm__R":
                bone.relative.rotation.pitch -= 45
                #     bone.relative.rotation.roll = 90
                #     # bone.relative.rotation.yaw = 0
                new_pose.append((bone.name, bone.relative))
            if bone.name == "crl_shoulder__R":
                bone.relative.rotation.pitch -= -1
                bone.relative.rotation.roll += 20
                # bone.relative.rotation.yaw = 90
                new_pose.append((bone.name, bone.relative))
            if bone.name == "crl_foreArm__R":
                bone.relative.rotation.pitch -= 10
                bone.relative.rotation.roll += 40
                # bone.relative.rotation.yaw = -45
                new_pose.append((bone.name, bone.relative))
            else:
                new_pose.append((bone.name, bone.relative))

        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)

        self.walker.blend_pose(0.7)

        # if bone.name == "crl_neck__C":
        #     roll = bone.relative.rotation.roll
        #     if roll >170 and self.head_roll >0:
        #         self.head_roll = -1
        #     elif roll < 120 and self.head_roll < 0:
        #         self.head_roll = 1

        # bone.relative.rotation.roll += self.head_roll  # Changed from 30 to 10
        # new_pose.append((bone.name, bone.relative))

        # Added new
        # else:
        #     # get current rotation
        #     new_pose.append((bone.name, bone.relative))

        # control = carla.WalkerBoneControlIn()
        # control.bone_transforms = new_pose
        # self.walker.set_bones(control)
        # self.walker.blend_pose(1.0)

        return "Done"



class RaiseArmStationary:
    def __init__(
        self, walker, start_pos, char, end_pos
    ):  # end_pos might be less relevant for pure static raise
        self.walker = walker
        self.start_pos = start_pos
        self.end_pos = end_pos  # Still used for the original "done" condition if needed
        self.done = False
        self.pose_applied_this_activation = False  # New flag
        self.char = char
        if char == "forcing":
            self.spine_roll = 60
        else:
            self.spine_roll = 20

    def reset_for_new_activation(self):
        """Call this from World.tick() before starting a new raise sequence."""
        self.done = False
        self.pose_applied_this_activation = False

    def step(self):
        if self.done:
            return "Done"

        current_walker_loc = self.walker.get_location()

        # Deactivation by reaching end_pos (original logic, might not be hit if stationary)
        if self.pose_applied_this_activation:  # Only check end_pos if arm is already up
            direction_end = current_walker_loc - self.end_pos
            direction_norm_end = math.sqrt(direction_end.x**2 + direction_end.y**2)
            if direction_norm_end < 0.3:
                logger.info("RaiseArmStationary: Reached end_pos, deactivating.")
                self.walker.blend_pose(0)
                self.done = True
                return "Done"

        # Activation condition
        direction_start = current_walker_loc - self.start_pos
        direction_norm_start = math.sqrt(direction_start.x**2 + direction_start.y**2)

        if not self.pose_applied_this_activation and direction_norm_start < 0.5:
            logger.info("RaiseArmStationary: At start_pos, applying arm-raised pose ONCE.")
            self.pose_applied_this_activation = True  # Set the flag

            bones = self.walker.get_bones()
            new_pose = []
            for bone in bones.bone_transforms:
                if bone.name == "crl_arm__R":
                    bone.relative.rotation.pitch -= 120
                    #     bone.relative.rotation.roll = 90
                    #     # bone.relative.rotation.yaw = 0
                    new_pose.append((bone.name, bone.relative))
                if bone.name == "crl_shoulder__R":
                    bone.relative.rotation.pitch -= -1
                    bone.relative.rotation.roll += 20
                    # bone.relative.rotation.yaw = 90
                    new_pose.append((bone.name, bone.relative))
                if bone.name == "crl_foreArm__R":
                    bone.relative.rotation.pitch -= 10
                    bone.relative.rotation.roll += 40
                    # bone.relative.rotation.yaw = -45
                    new_pose.append((bone.name, bone.relative))
                else:
                    new_pose.append((bone.name, bone.relative))

            control = carla.WalkerBoneControlIn()
            control.bone_transforms = new_pose
            self.walker.set_bones(control)
            self.walker.blend_pose(2)  # Blend to this pose quickly

            return "Running"  # Pose is now being blended to

        # If pose has been applied, or not yet at start_pos, or already done
        # it means the controller is either waiting, holding the pose (passively), or finished.
        # The timed deactivation will happen in World.tick()
        return "Running" if not self.done else "Done"


class LookBehindLeft:
    def __init__(self, walker, start_pos=None, mult=1):
        self.walker = walker
        self.start_pos = start_pos
        self.mult = mult
        self.done = False

    def step(self):
        if self.done:
            return "Done"
        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.1:
                return "Running"
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_hips__C":  # Added new
                bone.relative.rotation.pitch -= 3  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += 0  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine01__C":
                bone.relative.rotation.pitch -= 40 * self.mult  # Changed from 50 to 13
                new_pose.append((bone.name, bone.relative))
            if bone.name == "crl_neck__C":
                bone.relative.rotation.pitch += 40 * self.mult  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch -= 40 * self.mult  # Changed from 40 to 13
                bone.relative.rotation.roll += 20 * self.mult  # added new
                bone.relative.rotation.yaw -= 0  # added new
                new_pose.append((bone.name, bone.relative))
            else:
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)
        self.done = True
        return "Done"


class TurnHeadRightBehind:
    def __init__(self, walker, start_pos=None):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False

    def step(self):
        if self.done:
            return "Done"
        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.2:
                return "Running"
        self.walker.icr = ICR.INTERESTED
        # print("TurnHeadRightBehind")
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_hips__C":  # Added new
                bone.relative.rotation.pitch += 70  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += 70  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine01__C":
                bone.relative.rotation.pitch += 90  # Changed from 50 to 13
                new_pose.append((bone.name, bone.relative))
            if bone.name == "crl_neck__C":
                bone.relative.rotation.pitch -= 90  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch += 90  # Changed from 40 to 13
                bone.relative.rotation.roll -= 20  # added new
                bone.relative.rotation.yaw -= 0  # added new
                new_pose.append((bone.name, bone.relative))
            else:
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)
        self.done = True

        return "Done"


class TurnHeadRightBehindNoICR:
    def __init__(self, walker, start_pos=None):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False

    def step(self):
        if self.done:
            return "Done"
        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.2:
                return "Running"
        # print("TurnHeadRightBehind")
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_hips__C":  # Added new
                bone.relative.rotation.pitch += 70  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += 70  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_spine01__C":
                bone.relative.rotation.pitch += 90  # Changed from 50 to 13
                new_pose.append((bone.name, bone.relative))
            if bone.name == "crl_neck__C":
                bone.relative.rotation.pitch -= 90  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch += 90  # Changed from 40 to 13
                bone.relative.rotation.roll -= 20  # added new
                bone.relative.rotation.yaw -= 0  # added new
                new_pose.append((bone.name, bone.relative))
            else:
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)
        self.done = True
        return "Done"


class TurnHeadRightWalk:
    def __init__(self, walker, start_pos=None, char="yielding"):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False
        if char == "forcing":
            self.spine_roll = 90
        else:
            self.spine_roll = 40
        # print(char)

    def step(self):
        if self.done:
            return "Done"
        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.1:
                return "Running"
        self.walker.icr = ICR.PLANNING_TO
        self.walker.son = self.walker.initial_son
        # print("TurnHeadRightWalk")
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += self.spine_roll  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_neck__C":
                bone.relative.rotation.pitch -= 120  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch += 90  # Changed from 40 to 13
                bone.relative.rotation.roll -= 20  # added new
                bone.relative.rotation.yaw -= 0  # added new
                new_pose.append((bone.name, bone.relative))
            else:
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)
        self.done = True
        self.walker.on_street = True

        return "Done"

    def relax_spine(self):
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll -= self.spine_roll / 2  # self.spine_roll
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)


class TurnHeadLeftWalk:
    def __init__(self, walker, start_pos=None, char="yielding"):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False
        if char == "forcing":
            self.spine_roll = 70
        else:
            self.spine_roll = 40
        # print(char)

    def step(self):
        if self.done:
            return "Done"
        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.1:
                return "Running"
        self.walker.icr = ICR.INTERESTED
        self.walker.son = self.walker.initial_son
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += self.spine_roll  # Added new
                new_pose.append((bone.name, bone.relative))  # Added new
            if bone.name == "crl_neck__C":
                bone.relative.rotation.pitch += 120  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch -= 90  # Changed from 40 to 13
                bone.relative.rotation.roll += 20  # added new
                bone.relative.rotation.yaw -= 0  # added new
                new_pose.append((bone.name, bone.relative))
            else:
                pass
                # new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)
        self.done = True
        self.walker.on_street = True

        return "Done"

    def relax_spine(self):
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll -= self.spine_roll / 2  # self.spine_roll
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)

    def lean_forward(self, mult=1.5):
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += mult * self.spine_roll  # self.spine_roll
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)


class LeanForward:
    def __init__(self, walker, start_pos):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False

    def step(self):
        if self.done:
            return "Done"

        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.1:
                return "Running"

        logger.info("LeanForward: Applying lean forward pose.")

        self.walker.icr = ICR.GOING_TO
        self.walker.son = SON.FORCING
        bones = self.walker.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_spine__C":  # Added new
                bone.relative.rotation.roll += 70  # self.spine_roll
                new_pose.append((bone.name, bone.relative))
            else:
                pass
                # new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn()
        control.bone_transforms = new_pose
        self.walker.set_bones(control)
        self.walker.blend_pose(0.25)

        return "Done"


class LeanForwardAndLook(BasePose):
    """Lean forward and look (either left or right) pose controller."""

    def __init__(
        self,
        walker: carla.Walker,
        start_pos: carla.libcarla.Location = None,
        char: str = "forcing",
        look_to: Literal["right", "left", "custom"] = "right",
        lean_amount: float = 100,
        head_turn_neck_pitch: float = 5,
        head_turn_head_pitch: float = -20,
        head_turn_head_roll: float = 0,
        hips_yaw_offset: float = 5,
        target_spine_yaw: float = 50,
        blend_duration: float = 0.3,
        hold_duration: float = 1.0,
    ) -> None:
        """Initialize a `LeanForward` with a head turn.

        **Details:**
        - `crl_neck__C.rotation.pitch` (`head_turn_neck_pitch`): (-) Makes him look to his left,
            (+) to his right.
        - `crl_Head__C.rotation.pitch` (`head_turn_head_pitch`): (+) Tilts the head to his left,
            (-) to his right.
        - `crl_Head__C.rotation.roll` (`head_turn_head_roll`): Controls the side-to-side tilt of
            the head (ear towards shoulder). If `+20` was part of the "look left", then `-20` for
            "look right" if it's a symmetrical tilt, or 0 if no tilt is desired.
        - `crl_Head__C.rotation.yaw`: If yaw is to be involved in the head turn
            (e.g., `yaw += angle` for left), then for a right turn it would be `yaw -= angle`.
        - `crl_hips__C` (`hips_yaw_offset`): If the actor is already oriented perpendicular to
            street, `hips_yaw_offset` might be 0. If actor faces along street, `hips_yaw_offset`
            might be `+/-90`. This needs to be relative to the actor's current animation base pose.
        - `crl_spine__C` (`target_spine_yaw`): Torso Twist (Spine Yaw relative to Hips, to look
            right/left along street). Degrees to twist spine for right look.
        """
        self.walker = walker
        self.start_pos = start_pos
        self.char = char  # Used to potentially vary spine_roll for leaning
        self.look_to = look_to

        # If look_to is custom, then we take the sign of what the user instantiated the class with
        # If it's either "left" or "right", then we override the sign with the following rules
        if self.look_to != "custom":
            head_turn_neck_pitch = abs(head_turn_neck_pitch)
            head_turn_head_pitch = abs(head_turn_head_pitch)
            head_turn_head_roll = abs(head_turn_head_roll)
            hips_yaw_offset = abs(hips_yaw_offset)
            target_spine_yaw = abs(target_spine_yaw)

        if self.look_to == "left":
            head_turn_head_pitch *= -1
            hips_yaw_offset *= -1
            target_spine_yaw *= -1
        elif self.look_to == "right":
            head_turn_neck_pitch *= -1
            head_turn_head_roll *= -1

        # Pose parameters
        self.lean_amount = lean_amount
        if self.char == "forcing":  # Example: more aggressive lean for forcing
            self.spine_roll_lean = self.lean_amount
        else:  # yielding
            self.spine_roll_lean = self.lean_amount * 0.7  # Less lean for yielding

        self.head_turn_neck_pitch = head_turn_neck_pitch
        self.head_turn_head_pitch = head_turn_head_pitch
        self.head_turn_head_roll = head_turn_head_roll
        self.hips_yaw_offset = hips_yaw_offset
        self.target_spine_yaw = target_spine_yaw

        self.blend_duration = blend_duration  # Time to blend into the combined pose
        self.hold_duration = hold_duration  # Time to hold the pose

        self.state = "Idle"  # Idle, ApplyingPose, Holding, Done
        self.start_time_apply: float | None = None
        self.start_time_hold: float | None = None
        self.done = False

    def step(self) -> Literal["Running", "Done"]:
        """Perform a step in the controller."""
        if self.done:
            return "Done"

        current_loc = self.walker.get_location()
        current_time = time.time()

        if self.state == "Idle":
            if self.start_pos is None or l2_distance(current_loc, self.start_pos) <= 0.5:
                self.state = "ApplyingPose"
                self.start_time_apply = current_time
                logger.info(f"LeanForwardAndLook: Applying lean and look {self.look_to} pose.")

                # Set DBN states
                self.walker.icr = ICR.GOING_TO  # Or another appropriate ICR for this action
                self.walker.son = SON.FORCING if self.char == "forcing" else SON.YIELDING

                bones = self.walker.get_bones()
                new_pose_combined = []
                for bone_transform in bones.bone_transforms:
                    name = bone_transform.name
                    # Create a mutable copy
                    relative_transform = carla.Transform(
                        location=bone_transform.relative.location,
                        rotation=bone_transform.relative.rotation,
                    )

                    # 1. Orient Hips/Legs (if needed, relative to animation's base)
                    #    This is tricky without knowing the base animation's hip orientation.
                    #    If the actor is already spawned facing perpendicular, this might be 0.
                    if name == "crl_hips__C":
                        relative_transform.rotation.yaw += self.hips_yaw_offset

                    # 2. Lean Forward (Spine Roll) - relative to the (potentially re-oriented) hips
                    #    AND Twist Torso (Spine Yaw) - also relative to hips
                    elif name == "crl_spine__C":
                        relative_transform.rotation.roll += self.spine_roll_lean
                        relative_transform.rotation.yaw += self.target_spine_yaw

                    # 3. Turn head left (neck and head pitch/roll)
                    elif name == "crl_neck__C":
                        relative_transform.rotation.pitch += self.head_turn_neck_pitch
                    elif name == "crl_Head__C":
                        relative_transform.rotation.pitch += self.head_turn_head_pitch
                        relative_transform.rotation.roll += self.head_turn_head_roll
                        # Yaw is not modified for a simple left look from TurnHeadLeftWalk example

                    new_pose_combined.append((name, relative_transform))

                control = carla.WalkerBoneControlIn(new_pose_combined)
                self.walker.set_bones(control)
                self.walker.blend_pose(self.blend_duration)
            else:
                return "Running"  # Waiting for trigger

        elif self.state == "ApplyingPose":
            if self.start_time_apply is None:
                raise ValueError("start_time_apply is None entering ApplyingPose state.")

            if current_time - self.start_time_apply >= self.blend_duration:
                self.state = "Holding"
                self.start_time_hold = current_time
                logger.info(
                    f"LeanForwardAndLook: Pose applied, now holding for {self.hold_duration}s.",
                )

        elif self.state == "Holding":
            if self.start_time_hold is None:
                raise ValueError("start_time_hold is None entering Holding state.")

            if current_time - self.start_time_hold >= self.hold_duration:
                self.state = "Done"
                self.done = True  # Mark controller as fully done
                logger.info("LeanForwardAndLook: Hold duration finished. Controller is Done.")
                # The pose remains set. External ResetPose controller should be called next.

        return "Running"  # If in Idle (waiting), ApplyingPose, or Holding state


class ResetPose:
    def __init__(self, walker, start_pos=None, name="ResetPoseAt"):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False

    def step(self):
        if self.done:
            return "Done"
        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.2:
                return "Running"

        self.done = True
        self.walker.blend_pose(0)
        return "Done"


class InternalStateSetter:
    def __init__(self, walker, start_pos, icr, son) -> None:
        self.walker = walker
        self.start_pos = start_pos
        self.icr = icr
        self.son = son
        self.done = False

    def step(self):
        if self.done:
            return "Done"
        if self.start_pos is not None:
            direction = self.walker.get_location() - self.start_pos
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            if direction_norm > 0.2:
                return "Running"
        self.walker.icr = self.icr
        self.walker.son = self.son
        self.done = True


class Relaxer:
    def __init__(self, walker, car, start_pos):
        self.walker = walker
        self.start_pos = start_pos
        self.car = car
        self.done = False

    def step(self):
        walker_loc = self.walker.get_location()
        car_loc = self.car.get_location()
        if self.done:
            return True
        # print(y_distance(walker_loc, self.start_pos), y_distance(walker_loc, car_loc) )
        if y_distance(walker_loc, self.start_pos) >= 0 and y_distance(walker_loc, car_loc) < 0:
            self.walker.blend_pose(0)
            self.done = True
        return self.done


class TurnHeadLeft:
    def _look_left(self, world):
        bones = world.player.get_bones()
        new_pose = []
        for bone in bones.bone_transforms:
            if bone.name == "crl_spine01__C":
                bone.relative.rotation.pitch -= 10  # Changed from 30 to 10
                new_pose.append((bone.name, bone.relative))
            elif bone.name == "crl_Head__C":
                bone.relative.rotation.pitch -= 13  # Changed from 50 to 13
                new_pose.append((bone.name, bone.relative))
        control = carla.WalkerBoneControlIn(new_pose)
        world.player.set_bones(control)
        world.player.blend_pose(0.75)  # Changed from 0.5 to 0.75


class UncertainSteps:
    def __init__(self, walker, uncertain_steps_points, char="yielding"):
        self.walker = walker
        self.uncertain_steps_points = uncertain_steps_points
        self.done = False
        self.current_point = 0
        self.start_direction = 1 if len(uncertain_steps_points) % 2 > 0 else -1
        self.lean = 0 if char == "yielding" else 70

    def step(self):
        if self.done:
            return "Done"

        point = self.uncertain_steps_points[self.current_point]
        direction = self.walker.get_location() - point
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm < 0.2:
            # LOOK LEFT
            if self.start_direction == -1:
                bones = self.walker.get_bones()
                new_pose = []
                for bone in bones.bone_transforms:
                    if bone.name == "crl_spine__C":  # Added new
                        bone.relative.rotation.roll += self.lean  # Added new
                        new_pose.append((bone.name, bone.relative))  # Added new
                    if bone.name == "crl_neck__C":
                        bone.relative.rotation.pitch += 120  # Changed from 30 to 10
                        new_pose.append((bone.name, bone.relative))
                    elif bone.name == "crl_Head__C":
                        bone.relative.rotation.pitch -= 90  # Changed from 40 to 13
                        bone.relative.rotation.roll += 20  # added new
                        bone.relative.rotation.yaw -= 0  # added new
                        new_pose.append((bone.name, bone.relative))
                    else:
                        pass
                        # new_pose.append((bone.name, bone.relative))
                control = carla.WalkerBoneControlIn()
                control.bone_transforms = new_pose
                self.walker.set_bones(control)
                self.walker.blend_pose(0.25)
                self.walker.on_street = True
                self.walker.icr = ICR.PLANNING_TO
                self.walker.son = SON.FORCING

            # LOOK RIGHT
            elif self.start_direction == 1:
                bones = self.walker.get_bones()
                new_pose = []
                for bone in bones.bone_transforms:
                    if bone.name == "crl_spine__C":  # Added new
                        bone.relative.rotation.roll += self.lean  # Added new
                        new_pose.append((bone.name, bone.relative))  # Added new
                    if bone.name == "crl_neck__C":
                        bone.relative.rotation.pitch -= 120  # Changed from 30 to 10
                        new_pose.append((bone.name, bone.relative))
                    elif bone.name == "crl_Head__C":
                        bone.relative.rotation.pitch += 90  # Changed from 40 to 13
                        bone.relative.rotation.roll -= 20  # added new
                        bone.relative.rotation.yaw -= 0  # added new
                        new_pose.append((bone.name, bone.relative))
                    else:
                        new_pose.append((bone.name, bone.relative))
                control = carla.WalkerBoneControlIn()
                control.bone_transforms = new_pose
                self.walker.set_bones(control)
                self.walker.blend_pose(0.25)
                self.walker.on_street = True
                self.walker.icr = ICR.INTERESTED
                self.walker.son = SON.YIELDING

            self.start_direction *= -1
            self.current_point += 1

        if self.current_point == len(self.uncertain_steps_points):
            self.done = True
            return "Done"
        # self.done = True
        # return "Done"
        return "Running"


class ControllerConfig:
    """Configuration object for controlling pedestrian and scenario parameters."""

    def __init__(self, ped_speed: float = 1.0, ped_distance: float = 30.0) -> None:
        """Initialize a configuration object for controlling pedestrian and scenario parameters.

        This class stores various parameters that define the behavior of a pedestrian
        within a specific scenario instance, such as speeds, distances for different
        actions, and interaction characteristics.

        Args:
            ped_speed (float): The base speed of the pedestrian in meters per second.
                Defaults to 1.0.
            ped_distance (float): A general distance parameter, often used as an initial
                distance for interactions or decisions. Its specific meaning can vary
                depending on the scenario. Defaults to 30.0.
        """
        self.ped_speed = ped_speed
        self.ped_distance = ped_distance
        self.spawning_distance = 0.0
        self.walking_distance = 0.0
        self.looking_distance = 0.0
        self.crossing_distance = 0.0
        self.reenter_distance = 0.0
        self.op_reenter_distance = 0.0
        self.char = "yielding"
        self.sprint_speed_multiplier = 1.0  # Default multiplier is 1 (no sprint)
        self.walking_distance_X = 0.0
        self.walking_distance_Y = 0.0
        self.walk_after_crossing_X = 0.0
        self.walk_after_crossing_Y = 0.0
        self.wait_duration = 0.0

    def __str__(self) -> str:
        """Return a string representation of the object.

        E.g.: `ControllerConfig(ped_speed=1.0, ped_distance=30.0, spawning_distance=0, ...)`
        """
        desc = "ControllerConfig("
        for attr, value in self.__dict__.items():
            desc += f"{attr}="
            if type(value) is float or type(value) is np.float64:
                desc += f"{value:.4f}, "
            else:
                desc += f"{value}, "
        return desc.rstrip(", ") + ")"


class ICR(Enum):
    """Represent the Intention to Claim the Road (ICR) of a pedestrian.

    This enum defines different levels of a pedestrian's intention to cross or
    occupy the road space, which can influence their behavior and interaction
    with vehicles.
    """

    VERY_LOW = 1  # Pedestrian has very little or no intention to cross.
    LOW = 2  # Pedestrian has low intention to cross, may be hesitant.
    INTERESTED = 3  # Pedestrian shows interest in crossing, may be observing.
    PLANNING_TO = 4  # Pedestrian is actively planning to cross, preparing to move.
    GOING_TO = 5  # Pedestrian is committed and moving to cross the road.


class SON(Enum):
    """Represent the Strategy of Negotiation (SON) of a pedestrian.

    This enum categorizes the general approach or strategy a pedestrian adopts
    when interacting or negotiating with a vehicle, particularly in situations
    of potential conflict for road space.
    """

    AVERTING = 1  # Pedestrian actively avoids conflict, e.g., by stopping or moving away.
    YIELDING = 2  # Pedestrian is prepared to give way to the vehicle.
    FORCING = 3  # Pedestrian intends to take priority and expects the vehicle to yield.


def l2_distance(pos1: carla.Location, pos2: carla.Location) -> float:
    """Calculate the 2D Euclidean distance between two CARLA locations.

    This function computes the straight-line distance between `pos1` and `pos2`
    in the XY plane, ignoring the Z (height) coordinate, using `math.hypot`
    for efficient calculation.

    Args:
        pos1 (carla.Location): The first CARLA location.
        pos2 (carla.Location): The second CARLA location.

    Returns:
        float: The 2D Euclidean distance between the two positions.
    """
    direction = pos1 - pos2
    return math.sqrt(direction.x**2 + direction.y**2)


def y_distance(pos1: carla.Location, pos2: carla.Location) -> float:
    """Calculate the difference in the Y-coordinates of two CARLA locations.

    This function returns the result of `pos2.y - pos1.y`. A positive value
    indicates that `pos2` is further along the positive Y-axis than `pos1`.
    This is often used to determine relative positioning along a specific axis,
    for example, if a pedestrian is in front of or behind a vehicle along its
    direction of travel if the Y-axis aligns with that direction.

    Args:
        pos1 (carla.Location): The first CARLA location.
        pos2 (carla.Location): The second CARLA location.

    Returns:
        float: The difference `pos2.y - pos1.y`.
    """
    return pos2.y - pos1.y


def l2_length(pos1: carla.Vector3D) -> float:
    """Calculate the 2D magnitude (length) of a CARLA vector.

    This function computes the length of the vector `pos1` in the XY plane,
    effectively treating it as a 2D vector by ignoring its Z component.
    This is equivalent to the L2 norm of the vector's (x, y) components.

    Args:
        pos1 (carla.Vector3D): The CARLA vector (or any object with x, y attributes)
            for which to calculate the 2D length.

    Returns:
        float: The 2D length (magnitude) of the vector.
    """
    direction = pos1
    return math.sqrt(direction.x**2 + direction.y**2)
