import math
from enum import Enum

import carla


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


# Works:
# class LookAcrossStreetLeft(object):
#     """
#     Controller to make the walker look left (North, Yaw=0 world)
#     assuming the walker has stopped at the curb facing East (Yaw=90 world).
#     Focuses mainly on Yaw rotation.
#     """

#     def __init__(self, walker, start_pos=None, duration=0.4):  # Slightly longer duration
#         self.walker = walker
#         self.start_pos = start_pos
#         self.done = False
#         self.duration = duration
#         self.start_time = None
#         self.state = "Idle"

#     def step(self):
#         if self.done:
#             return "Done"

#         current_loc = self.walker.get_location()

#         # Trigger condition
#         if (
#             self.state == "Idle"
#             and self.start_pos is not None
#             and l2_distance(current_loc, self.start_pos) <= 0.3
#         ):
#             self.state = "Looking"
#             self.start_time = time.time()
#             logger.debug("LookAcrossStreetLeft triggered (Yaw only).")

#             bones = self.walker.get_bones()
#             new_pose = []
#             # Target World Yaw = 0 (North). Walker World Yaw = 90 (East). Relative change = -90.
#             target_relative_yaw = -90
#             for bone in bones.bone_transforms:
#                 if bone.name == "crl_neck__C":
#                     bone.relative.rotation.pitch = 0  # Keep level
#                     bone.relative.rotation.roll = 0  # Keep level
#                     bone.relative.rotation.yaw = target_relative_yaw * 0.6  # Distribute turn
#                     new_pose.append((bone.name, bone.relative))
#                 elif bone.name == "crl_Head__C":
#                     bone.relative.rotation.pitch = 0
#                     bone.relative.rotation.roll = 0
#                     bone.relative.rotation.yaw = target_relative_yaw * 0.4  # Distribute turn
#                     new_pose.append((bone.name, bone.relative))
#                 # Optional minimal spine twist if needed
#                 # elif bone.name == "crl_spine01__C":
#                 #     bone.relative.rotation.roll = 0
#                 #     bone.relative.rotation.yaw = target_relative_yaw * 0.1
#                 #     new_pose.append((bone.name, bone.relative))
#                 else:
#                     new_pose.append((bone.name, bone.relative))  # Keep others default

#             control = carla.WalkerBoneControlIn()
#             control.bone_transforms = new_pose
#             self.walker.set_bones(control)
#             self.walker.blend_pose(self.duration)  # Blend over specified duration

#         elif self.state == "Looking":
#             if time.time() - self.start_time >= self.duration:
#                 self.state = "Done"
#                 self.done = True
#                 logger.debug("LookAcrossStreetLeft finished blending.")
#                 return "Done"

#         elif self.state == "Done":
#             return "Done"

#         return "Running"

# Works meh:
# import numpy as np


# class LookAcrossStreetLeft(object):
#     """
#     Controller to make the walker look left (North, World Yaw=0, World Pitch=0).
#     Calculates required relative rotation to achieve the target world orientation.
#     Focuses rotation primarily on the neck.
#     """

#     def __init__(
#         self, walker, start_pos=None, duration=0.6, target_world_yaw=0.0, target_world_pitch=0.0
#     ):
#         self.walker = walker
#         self.start_pos = start_pos
#         self.duration = duration
#         self.target_world_yaw = target_world_yaw
#         self.target_world_pitch = target_world_pitch  # Allow specifying target pitch
#         self.done = False
#         self.start_time = None
#         self.state = "Idle"
#         self._target_pose_set = False

#     def _get_bone_world_transform(self, bone_transforms, bone_name):
#         for bone_info in bone_transforms:
#             if bone_info.name == bone_name:
#                 return bone_info.world
#         return None

#     def _matrix_to_carla_rotation(self, matrix):
#         # (Keep the matrix_to_carla_rotation function from the previous version)
#         rot_mat = matrix[:3, :3]
#         sin_pitch = -rot_mat[2, 0]
#         sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
#         pitch = math.degrees(math.asin(sin_pitch))
#         cos_pitch = math.cos(math.radians(pitch))
#         if abs(cos_pitch) > 1e-6:
#             sin_yaw = rot_mat[1, 0] / cos_pitch
#             cos_yaw = rot_mat[0, 0] / cos_pitch
#             yaw = math.degrees(math.atan2(sin_yaw, cos_yaw))
#             sin_roll = rot_mat[2, 1] / cos_pitch
#             cos_roll = rot_mat[2, 2] / cos_pitch
#             roll = math.degrees(math.atan2(sin_roll, cos_roll))
#         else:
#             yaw = 0
#             sin_roll = -rot_mat[1][2]
#             cos_roll = rot_mat[1][1]
#             roll = math.degrees(math.atan2(sin_roll, cos_roll))
#         return carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

#     def _calculate_target_relative_rotation(self, bone_world_tf, parent_world_tf):
#         """Calculates the full relative rotation needed for the bone to achieve the target world yaw/pitch."""
#         if not bone_world_tf or not parent_world_tf:
#             return carla.Rotation()

#         # --- Target World Rotation: Target Yaw, Target Pitch, Current Roll ---
#         target_world_rotation = carla.Rotation(
#             pitch=self.target_world_pitch,  # Explicitly target pitch 0
#             yaw=self.target_world_yaw,
#             roll=bone_world_tf.rotation.roll,  # Keep current roll to avoid weird tilt
#         )
#         target_bone_world_tf = carla.Transform(bone_world_tf.location, target_world_rotation)

#         # --- Calculate Relative Transform Matrix---
#         try:
#             parent_world_inv_matrix = np.linalg.inv(parent_world_tf.get_matrix())
#             target_bone_world_matrix = target_bone_world_tf.get_matrix()
#             target_bone_relative_matrix = np.dot(parent_world_inv_matrix, target_bone_world_matrix)
#         except np.linalg.LinAlgError:
#             return carla.Rotation()

#         target_relative_rotation = self._matrix_to_carla_rotation(target_bone_relative_matrix)
#         return target_relative_rotation

#     def step(self):
#         if self.done:
#             return "Done"
#         current_loc = self.walker.get_location()

#         if self.state == "Idle":
#             if self.start_pos is not None and l2_distance(current_loc, self.start_pos) <= 0.3:
#                 self.state = "Looking"
#                 self.start_time = time.time()
#                 self._target_pose_set = False
#                 logger.info(
#                     f"LookAcrossStreetLeft triggered. Target Head World Yaw: {self.target_world_yaw:.1f}, Pitch: {self.target_world_pitch:.1f}"
#                 )
#                 return "Running"
#             else:
#                 return "Idle"

#         elif self.state == "Looking":
#             if not self._target_pose_set:
#                 bones_out = self.walker.get_bones()
#                 if not bones_out or not bones_out.bone_transforms:
#                     logger.warning("LookAcrossStreetLeft: Could not get bones.")
#                     self.state = "Done"
#                     self.done = True
#                     return "Done"

#                 bone_transforms_list = bones_out.bone_transforms
#                 new_pose = []
#                 try:
#                     spine01_world = self._get_bone_world_transform(
#                         bone_transforms_list, "crl_spine01__C"
#                     )
#                     neck_world = self._get_bone_world_transform(
#                         bone_transforms_list, "crl_neck__C"
#                     )
#                     head_world = self._get_bone_world_transform(
#                         bone_transforms_list, "crl_Head__C"
#                     )
#                     parent_for_neck = (
#                         spine01_world if spine01_world else self.walker.get_transform()
#                     )
#                     parent_for_head = neck_world if neck_world else parent_for_neck

#                     # Calculate the full target *relative* rotations needed for Yaw=0, Pitch=0
#                     target_neck_relative_rot = self._calculate_target_relative_rotation(
#                         neck_world, parent_for_neck
#                     )
#                     target_head_relative_rot = self._calculate_target_relative_rotation(
#                         head_world, parent_for_head
#                     )

#                     # Inversion might still be needed based on coordinate system mapping
#                     # Let's try *without* inversion first, now that pitch is also targeted.
#                     # If it looks right instead of left, add the negation back.
#                     final_neck_yaw = target_neck_relative_rot.yaw
#                     final_head_yaw = target_head_relative_rot.yaw
#                     logger.debug(
#                         f"Calculated Target Relative -> Neck Yaw: {final_neck_yaw:.1f}, Pitch: {target_neck_relative_rot.pitch:.1f}"
#                     )
#                     logger.debug(
#                         f"Calculated Target Relative -> Head Yaw: {final_head_yaw:.1f}, Pitch: {target_head_relative_rot.pitch:.1f}"
#                     )

#                     for bone in bone_transforms_list:
#                         current_relative_loc = bone.relative.location
#                         current_relative_rot = bone.relative.rotation  # Keep original roll

#                         if bone.name == "crl_neck__C":
#                             # Apply calculated target relative yaw and pitch, keep original roll
#                             new_rotation = carla.Rotation(
#                                 pitch=target_neck_relative_rot.pitch,  # Use calculated pitch
#                                 yaw=final_neck_yaw,
#                                 roll=current_relative_rot.roll,
#                             )
#                             new_pose.append(
#                                 (bone.name, carla.Transform(current_relative_loc, new_rotation))
#                             )
#                         elif bone.name == "crl_Head__C":
#                             new_rotation = carla.Rotation(
#                                 pitch=target_head_relative_rot.pitch,  # Use calculated pitch
#                                 yaw=final_head_yaw,
#                                 roll=current_relative_rot.roll,
#                             )
#                             new_pose.append(
#                                 (bone.name, carla.Transform(current_relative_loc, new_rotation))
#                             )
#                         else:
#                             new_pose.append((bone.name, bone.relative))

#                     control = carla.WalkerBoneControlIn()
#                     control.bone_transforms = new_pose
#                     self.walker.set_bones(control)
#                     self.walker.blend_pose(self.duration)
#                     self._target_pose_set = True
#                     logger.debug(
#                         "LookAcrossStreetLeft: Target pose (Yaw=0, Pitch=0 world target) blend started."
#                     )

#                 except Exception as e:
#                     logger.error(f"Error calculating/applying look pose: {e}", exc_info=True)
#                     self.state = "Done"
#                     self.done = True
#                     return "Done"

#             # Check blend duration
#             if time.time() - self.start_time >= self.duration:
#                 self.state = "Done"
#                 self.done = True
#                 try:
#                     bones_final = self.walker.get_bones()
#                     final_head_transform = self._get_bone_world_transform(
#                         bones_final.bone_transforms, "crl_Head__C"
#                     )
#                     if final_head_transform:
#                         logger.info(
#                             f"LookAcrossStreetLeft finished blending. Final Head World Yaw: {final_head_transform.rotation.yaw:.1f}, Pitch: {final_head_transform.rotation.pitch:.1f}"
#                         )
#                     else:
#                         logger.warning(
#                             "LookAcrossStreetLeft finished, could not get final head transform."
#                         )
#                 except Exception as e:
#                     logger.warning(f"Exception getting final head transform after blend: {e}")
#                 return "Done"
#             else:
#                 return "Running"


#         elif self.state == "Done":
#             return "Done"
#         return "Idle"
# Optional: Apply a fraction of the neck's *change* in relative yaw to the spine?
# elif bone.name == "crl_spine01__C":
#     spine_yaw_change = (
#         self._normalize_angle(
#             target_neck_relative_rot.yaw - current_relative_rot.yaw
#         )
#         * 0.1
#     )  # Small fraction
#     new_rotation = carla.Rotation(
#         pitch=current_relative_rot.pitch,
#         yaw=current_relative_rot.yaw + spine_yaw_change,
#         roll=current_relative_rot.roll,
#     )
#     new_pose.append(
#         (bone.name, carla.Transform(current_relative_loc, new_rotation))
#     )
class LookAcrossStreetLeft(object):
    """
    Controller to make the walker look left (North, Yaw=0 world)
    assuming the walker has stopped at the curb facing East (Yaw=90 world).
    Uses REDUCED relative yaw to avoid unnatural twisting.
    """

    def __init__(self, walker, start_pos=None, duration=0.4):
        self.walker = walker
        self.start_pos = start_pos
        self.done = False
        self.duration = duration
        self.start_time = None
        self.state = "Idle"

    def step(self):
        if self.done:
            return "Done"

        current_loc = self.walker.get_location()

        # Trigger condition
        if (
            self.state == "Idle"
            and self.start_pos is not None
            and l2_distance(current_loc, self.start_pos) <= 0.3
        ):
            self.state = "Looking"
            self.start_time = time.time()
            logger.debug("LookAcrossStreetLeft triggered (Reduced Yaw).")

            bones = self.walker.get_bones()
            new_pose = []
            # --- REDUCED YAW --- Aim for maybe -60 to -70 degrees relative total
            neck_relative_yaw = -45  # Reduced from -80 * 0.6
            head_relative_yaw = -20  # Reduced from -80 * 0.4 + -10
            # Total relative yaw is now approx -65 degrees

            for bone in bones.bone_transforms:
                if bone.name == "crl_neck__C":
                    bone.relative.rotation.pitch = 0  # Keep level
                    bone.relative.rotation.roll = 0  # Keep level
                    bone.relative.rotation.yaw = neck_relative_yaw  # Apply reduced yaw
                    new_pose.append((bone.name, bone.relative))
                elif bone.name == "crl_Head__C":
                    bone.relative.rotation.pitch = 0  # Keep level
                    bone.relative.rotation.roll = 0  # Keep level
                    bone.relative.rotation.yaw = head_relative_yaw  # Apply reduced yaw
                    new_pose.append((bone.name, bone.relative))
                # Remove spine twist for simplicity
                # elif bone.name == "crl_spine01__C":
                #     pass # Don't modify spine roll/yaw
                else:
                    # Ensure other bones maintain their default relative transform from the base animation
                    # If just appending bone.relative, it might capture an intermediate state.
                    # It's safer to not include bones we aren't actively changing unless
                    # we know their default relative transform. Let's only include neck/head.
                    if bone.name not in ["crl_neck__C", "crl_Head__C"]:
                        new_pose.append((bone.name, bone.relative))

            control = carla.WalkerBoneControlIn()
            # Only apply transforms for the bones we are controlling
            controlled_bones_pose = [
                (b[0], b[1]) for b in new_pose if b[0] in ["crl_neck__C", "crl_Head__C"]
            ]
            if not controlled_bones_pose:
                logger.warning("LookAcrossStreetLeft: No neck or head bones found in pose list!")
            else:
                control.bone_transforms = controlled_bones_pose
                self.walker.set_bones(control)
                self.walker.blend_pose(self.duration)

        elif self.state == "Looking":
            if time.time() - self.start_time >= self.duration:
                self.state = "Done"
                self.done = True
                logger.debug("LookAcrossStreetLeft finished blending.")
                return "Done"

        elif self.state == "Done":
            return "Done"

        return "Running"


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
                for bone in bones.bone_transforms:
                    # Modify Right Arm bones
                    if bone.name == "crl_arm__R":
                        bone.relative.rotation.pitch -= 70  # Raise arm forward/up
                        bone.relative.rotation.roll += 20  # Slight outward roll
                        new_pose.append((bone.name, bone.relative))
                    elif bone.name == "crl_shoulder__R":
                        bone.relative.rotation.pitch -= 10  # Adjust shoulder
                        new_pose.append((bone.name, bone.relative))
                    # elif bone.name == "crl_foreArm__R":
                    #     bone.relative.rotation.pitch += 0 # Keep forearm straight initially
                    #     new_pose.append((bone.name, bone.relative))
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
        # Has to be initialized due weired initial call
        self.spawning_distance = 0
        self.walking_distance = 0
        self.looking_distance = 0
        self.crossing_distance = 0
        self.reenter_distance = 0
        self.op_reenter_distance = 0
        self.char = "yielding"


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
