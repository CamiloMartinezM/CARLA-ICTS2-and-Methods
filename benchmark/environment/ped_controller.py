# ped_controller.py
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
