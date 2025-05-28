import math
import time
from enum import Enum

import carla
import numpy as np

from carla_icts2.config import logger


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

        logger.info("Bones called")

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


# class RaiseArmStationary:
#     """Raises Arm in a stationary gesture.

#     Improvement over `RaiseArm`, so that if the walker is at the `start_pos` (and not yet past
#     `end_pos`), it still applies the arm-raising pose.
#     """

#     def __init__(self, walker, start_pos, char, end_pos):
#         self.walker = walker
#         self.start_pos = start_pos
#         self.end_pos = end_pos
#         self.done = False
#         self.has_started_raising = False
#         if char == "forcing":
#             self.spine_roll = 60
#         else:
#             self.spine_roll = 20

#     def step(self):
#         if self.done:
#             return "Done"

#         current_walker_loc = self.walker.get_location()

#         # Check if we should stop raising the arm
#         if self.has_started_raising:  # Only check end_pos if arm is already up
#             direction_end = current_walker_loc - self.end_pos
#             direction_norm_end = math.sqrt(direction_end.x**2 + direction_end.y**2)
#             if direction_norm_end < 0.3:  # Slightly larger threshold for deactivation
#                 logger.info("RaiseArm: Reached end_pos, deactivating.")
#                 self.walker.blend_pose(0)  # Reset to default animation
#                 self.done = True
#                 return "Done"

#         # Check if we should start or continue raising the arm
#         # Condition: Walker is at or just past start_pos, and not yet done.
#         direction_start = current_walker_loc - self.start_pos
#         direction_norm_start = math.sqrt(direction_start.x**2 + direction_start.y**2)

#         # Trigger if close to start_pos OR if already started raising and not yet at end_pos
#         should_apply_pose = False
#         if not self.has_started_raising:
#             if direction_norm_start < 0.3:  # If close enough to start_pos
#                 logger.info("RaiseArm: At start_pos, initiating arm raise.")
#                 self.has_started_raising = True
#                 should_apply_pose = True
#             else:
#                 # Not yet at start_pos or too far past it without having started
#                 return "Running"  # Waiting to reach start_pos
#         elif self.has_started_raising and not self.done:  # Already started, keep arm up
#             should_apply_pose = True

#         if should_apply_pose:
#             logger.debug("RaiseArm: Applying arm-raised pose.")  # Changed to debug
#             bones = self.walker.get_bones()
#             new_pose = []
#             for bone in bones.bone_transforms:
#                 if bone.name == "crl_arm__R":
#                     bone.relative.rotation.pitch -= 45
#                     #     bone.relative.rotation.roll = 90
#                     #     # bone.relative.rotation.yaw = 0
#                     new_pose.append((bone.name, bone.relative))
#                 if bone.name == "crl_shoulder__R":
#                     bone.relative.rotation.pitch -= -1
#                     bone.relative.rotation.roll += 20
#                     # bone.relative.rotation.yaw = 90
#                     new_pose.append((bone.name, bone.relative))
#                 if bone.name == "crl_foreArm__R":
#                     bone.relative.rotation.pitch -= 10
#                     bone.relative.rotation.roll += 40
#                     # bone.relative.rotation.yaw = -45
#                     new_pose.append((bone.name, bone.relative))
#                 else:
#                     new_pose.append((bone.name, bone.relative))

#             control = carla.WalkerBoneControlIn()
#             control.bone_transforms = new_pose
#             self.walker.set_bones(control)
#             self.walker.blend_pose(0.7)  # Blend a bit faster for a snappier raise
#             return "Running"  # Pose applied, controller is active

#         return "Running"  # Should not be reached if logic is correct, but as fallback


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
    def __init__(self, ped_speed=1.0, ped_distance=30.0):
        self.ped_speed = ped_speed
        self.ped_distance = ped_distance
        self.spawning_distance = 0
        self.walking_distance = 0
        self.looking_distance = 0
        self.crossing_distance = 0
        self.reenter_distance = 0
        self.op_reenter_distance = 0
        self.char = "yielding"

        # --- NEW Parameters specifically for IConfig07 ---
        self.sprint_speed_multiplier = 1.0  # Default multiplier is 1 (no sprint)
        self.walking_distance_X = None
        self.walking_distance_Y = None
        self.walk_after_crossing_X = None
        self.walk_after_crossing_Y = None
        self.wait_duration = 0.0  # Default wait duration

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
    VERY_LOW = 1
    LOW = 2
    INTERESTED = 3
    PLANNING_TO = 4
    GOING_TO = 5


class SON(Enum):
    AVERTING = 1
    YIELDING = 2
    FORCING = 3


def l2_distance(pos1, pos2):
    direction = pos1 - pos2
    direction_norm = math.sqrt(direction.x**2 + direction.y**2)
    return direction_norm


def y_distance(pos1, pos2):
    return pos2.y - pos1.y


def l2_length(pos1):
    direction = pos1
    direction_norm = math.sqrt(direction.x**2 + direction.y**2)
    return direction_norm
