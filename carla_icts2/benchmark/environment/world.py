"""Author: Dikshant Gupta
Time: 23.03.21 14:27
"""

import math
import random
import sys
import time

import carla
import numpy as np

from carla_icts2.benchmark.environment.car_controller import CarController
from carla_icts2.benchmark.environment.ped_controller import (
    ICR,
    SON,
    ControllerConfig,
    InternalStateSetter,
    LeanForward,
    LookAcrossStreetLeft,
    LookBehindLeft,
    LookBehindLeftSpine,
    LookBehindRight,
    PathController,
    RaiseArm,
    RaiseHandBriefly,
    Relaxer,
    ResetPose,
    TurnHeadLeftWalk,
    TurnHeadRightBehind,
    TurnHeadRightBehindNoICR,
    TurnHeadRightWalk,
    UncertainSteps,
    l2_distance,
    l2_length,
    y_distance,
)
from carla_icts2.benchmark.environment.sensors import *
from carla_icts2.benchmark.environment.utils import find_weather_presets
from carla_icts2.config import logger


class World:
    def __init__(self, carla_world, hud, scenario, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            logger.error(f"RuntimeError: {error}")
            logger.error("  The server could not send the OpenDRIVE (.xodr) file:")
            logger.error("  Make sure it exists, has the same name of your town, and is correct.")
            sys.exit(1)
        self.hud = hud
        self.scenario = None
        self.player = None
        self.walker = None
        self.incoming_car = None
        self.parked_cars = None
        self.player_max_speed = None
        self.player_max_speed_fast = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.semseg_sensor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gama
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        # TODO CHECK THIS OUT
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            # carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All,
        ]

        self.car_blueprint = self.get_car_blueprint()
        self.ped_speed = None
        self.ped_distance = None
        self.drawn = False
        self.camera = True
        self.restart(scenario)
        self.world.on_tick(hud.on_world_tick)
        for _ in range(2):
            self.next_weather()
        self.random = False
        self.dummy_car = False
        self.debug = True

        # Variables needed to set up the spectator camera
        self.update_camera_func = None
        self.update_camera_args = None
        self.already_spawned_bev_camera = False

        # Define the functions to update the camera based on the provided specifications
        self.define_camera_based_on_specs(camera_specs=args.camera)

    def define_camera_based_on_specs(self, camera_specs: str) -> None:
        """Define the function to update the camera based on the provided specifications.

        Specifically, it defines the `self.update_camera_func` and `self.update_camera_args` attrs
        which will be called in the `tick` method.

        Args:
            camera_specs (str): A camera specification. Could be:
                - "pedestrian_pov"
                - "vehicle_pov"
                - "bev_static"
                - "bev_follow_vehicle"
                - "bev_follow_pedestrian"
        """
        if camera_specs == "pedestrian_pov":
            self.update_camera_func = self.update_walker_pov_camera
        elif camera_specs == "vehicle_pov":
            self.update_camera_func = self.update_player_pov_camera
        elif camera_specs == "bev_static":
            self.update_camera_func = self.update_bev_camera
            self.update_camera_args = {"follow_player": False, "follow_walker": False}
        elif camera_specs == "bev_follow_vehicle":
            self.update_camera_func = self.update_bev_camera
            self.update_camera_args = {"follow_player": True, "follow_walker": False}
        elif camera_specs == "bev_follow_pedestrian":
            self.update_camera_func = self.update_bev_camera
            self.update_camera_args = {"follow_player": False, "follow_walker": True}
        else:
            logger.warning(f"Unrecognized camera specification: {camera_specs}. Using default.")

    def update_bev_camera(
        self,
        *,
        follow_player: bool = False,
        follow_walker: bool = False,
    ) -> None:
        """Create and attach a Bird's Eye View (BEV) camera.

        Args:
            follow_player (bool): If True, the camera will follow the vehicle. Default is False.
            follow_walker (bool): If True, the camera will follow the walker. Default is False.
        """
        if follow_player and follow_walker:
            logger.error(
                "Cannot follow both player and walker at the same time. "
                "Setting to follow vehicle only.",
            )
            follow_player = True

        if follow_walker or not self.already_spawned_bev_camera:
            transform = self.walker.get_transform()

        if follow_player or not self.already_spawned_bev_camera:
            transform = self.player.get_transform()

        location = carla.Location(x=transform.location.x, y=transform.location.y, z=10)
        self.already_spawned_bev_camera = True
        self.world.get_spectator().set_transform(
            carla.Transform(location, carla.Rotation(yaw=180.0, pitch=-90.0)),
        )

    def update_player_pov_camera(self) -> None:
        """Create and attach a POV camera inside the car."""
        # offset = carla.Location(x=0.3, y=0.0, z=1.5)
        offset = carla.Location(x=0.6, y=-0.4, z=1.2)

        # Compute the offset in the vehicle's local frame.
        # Here we rotate the offset vector by the vehicle's yaw.
        # Using the forward and right vectors ensures the offset follows the car.
        forward = self.player.get_transform().get_forward_vector()
        right = self.player.get_transform().get_right_vector()

        offset_world = carla.Location(
            x=offset.x * forward.x + offset.y * right.x,
            y=offset.x * forward.y + offset.y * right.y,
            z=offset.z,
        )
        new_location = self.player.get_transform().location + offset_world

        # Match the rotation of the vehicle
        new_rotation = self.player.get_transform().rotation
        new_transform = carla.Transform(new_location, new_rotation)
        self.world.get_spectator().set_transform(new_transform)

    def get_head_world_transform(self) -> carla.Transform | None:
        """Retrieve the world transform of the pedestrian's head bone ('crl_Head__C').

        Args:
            pedestrian: The carla.Actor (walker) object.

        Returns:
            carla.Transform: The world transform of the head bone, or None if not found or bones
                cannot be retrieved.
        """
        try:
            bones_out = self.walker.get_bones()
            if not bones_out or not bones_out.bone_transforms:
                logger.warning(f"Could not retrieve valid bones for pedestrian {self.walker.id}")
                return None

            # Find the head bone transform in the list
            for bone_info in bones_out.bone_transforms:
                if bone_info.name == "crl_Head__C":
                    return bone_info.world  # Return the pre-computed world transform

            logger.warning(f"Head bone 'crl_Head__C' not found for pedestrian {self.walker.id}")
            return None

        except Exception as e:
            logger.error(
                f"Error getting bones for pedestrian {self.walker.id}: {e}", exc_info=True
            )
            return None

    def update_walker_pov_camera(self) -> None:
        """Create and attach a POV camera at the pedestrian's eye level."""
        # offset = carla.Location(x=0.0, y=0.0, z=1.7)
        eye_forward_offset = 0.4  # How far in front of the head bone origin
        eye_up_offset = 0.05  # How far above the head bone origin
        eye_right_offset = 0.0  # Sideways offset (usually 0)

        # --- Get Head Bone's World Transform ---
        head_transform = self.get_head_world_transform()

        if head_transform is None:
            # --- Fallback: Use the actor's base transform + fixed Z offset ---
            # This is less accurate but prevents errors if the head bone isn't found.
            logger.warning(
                f"Head bone 'crl_Head__C' not found for {self.walker.id}. Using fallback view."
            )
            pedestrian_transform = self.walker.get_transform()
            fallback_location = pedestrian_transform.location + carla.Location(
                z=1.7
            )  # Approx eye level from ground
            final_transform = carla.Transform(fallback_location, pedestrian_transform.rotation)
        else:
            # --- Use Head Bone Transform for Position and Orientation Basis ---

            # # This works to create a 3rd person view
            # # 1. Calculate Final Camera POSITION
            # # Apply the offset relative to the head bone's local coordinate system
            # final_location = head_transform.transform(offset)

            # # 2. Calculate Final Camera ROTATION based on head's forward direction
            # # Get the head bone's forward vector in world coordinates
            # forward_vector = head_transform.get_forward_vector()

            # # Calculate Yaw and Pitch from the forward vector to orient the camera
            # # Note: CARLA's coordinate system: X=Forward, Y=Left, Z=Up
            # # atan2(y, x) gives the angle from the positive X-axis
            # yaw = math.degrees(math.atan2(forward_vector.y, forward_vector.x)) - 90

            # # asin gives the angle with the XY plane. Negative Z for looking down (positive pitch).
            # # Clamp the argument to avoid domain errors due to floating point inaccuracies.
            # asin_arg = max(-1.0, min(1.0, forward_vector.z))
            # pitch = math.degrees(math.asin(asin_arg))

            # # Create the final rotation with Roll=0 to keep the camera upright
            # final_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)

            # final_transform = carla.Transform(final_location, final_rotation)

            # # This works to create a POV view
            # 1. Calculate Final Camera ROTATION based on head's forward direction
            # Get the head bone's forward vector in world coordinates
            head_forward_vector = head_transform.get_forward_vector()

            # Calculate Yaw and Pitch from the forward vector
            # -90° makes the camera look forward instead of to the side
            yaw = math.degrees(math.atan2(head_forward_vector.y, head_forward_vector.x)) - 90
            # ************************************************************

            # Clamp asin argument for safety
            asin_arg = max(-1.0, min(1.0, head_forward_vector.z))
            pitch = math.degrees(math.asin(asin_arg)) - 10  # Make it look slightly down

            # Ensure camera is upright relative to the world
            final_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)

            # 2. Calculate Final Camera POSITION based on the calculated rotation and head origin
            head_origin = head_transform.location

            # Get the direction vectors *from the calculated final rotation*
            camera_forward = final_rotation.get_forward_vector()
            camera_up = final_rotation.get_up_vector()
            camera_right = final_rotation.get_right_vector()

            # Calculate the world offset vector based on the desired eye position relative to the
            # head origin
            world_offset = (
                camera_forward * eye_forward_offset
                + camera_up * eye_up_offset
                + camera_right * eye_right_offset
            )

            # Add the world offset to the head bone's origin
            final_location = head_origin + world_offset

            final_transform = carla.Transform(final_location, final_rotation)

        # Update the spectator view
        self.world.get_spectator().set_transform(final_transform)

    def get_car_blueprint(self):
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute("role_name", self.actor_role_name)
        if blueprint.has_attribute("color"):
            color = blueprint.get_attribute("color").recommended_values[1]
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")
        # set the max speed
        if blueprint.has_attribute("speed"):
            self.player_max_speed = float(blueprint.get_attribute("speed").recommended_values[1])
            self.player_max_speed_fast = float(
                blueprint.get_attribute("speed").recommended_values[2],
            )
        else:
            print("No recommended values for 'speed' attribute")
        return blueprint

    def get_random_blueprint(self):
        vehicles = [
            "vehicle.audi.a2",
            "vehicle.audi.tt",
            "vehicle.chevrolet.impala",
            "vehicle.audi.etron",
        ]
        vehicle_type = random.choice(vehicles)
        blueprint = random.choice(self.world.get_blueprint_library().filter(vehicle_type))
        blueprint.set_attribute("role_name", self.actor_role_name)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")
        return blueprint

    def _draw_grid(self):
        width = 20
        world = self.world
        loc = self.walker.get_location()
        loc = self.walker.get_location() + carla.Location(0, 0, 0)
        upper = loc + carla.Location(0, -width, 0)
        right = loc + carla.Location(width, 0, 0)
        right_upper = loc + carla.Location(width, -width, 0)
        world.debug.draw_line(loc, upper, thickness=0.02)
        world.debug.draw_line(loc, right, thickness=0.02)
        world.debug.draw_line(upper, right_upper, thickness=0.02)
        world.debug.draw_line(right, right_upper, thickness=0.02)
        for i in range(1, width):
            offset_y = carla.Location(0, -i, 0)
            offset_x = carla.Location(i, 0, 0)
            world.debug.draw_line(loc + offset_y, right + offset_y, thickness=0.02)
            world.debug.draw_line(loc + offset_x, upper + offset_x, thickness=0.02)

    def _draw_point(self, p, color=carla.Color(r=0, g=255, b=255)):
        self.world.debug.draw_point(p, size=0.1, color=color, life_time=0)

    def get_point(self, offset):
        cur = self.walker.get_location()
        offset_x, offset_y = offset
        loc = carla.Location(cur.x + offset_x, cur.y - offset_y, 0.5)
        return loc

    def restart(self, scenario, conf=ControllerConfig()):
        self.scenario = scenario
        self.ped_speed = conf.ped_speed
        self.ped_distance = conf.ped_distance
        self.counter = 0

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        semseg_index = self.semseg_sensor.index if self.semseg_sensor is not None else 5
        cam_pos_index = (
            self.camera_manager.transform_index if self.camera_manager is not None else 5
        )
        semseg_pos_index = (
            self.semseg_sensor.transform_index if self.semseg_sensor is not None else 5
        )

        # Spawn the player.
        start = self.scenario[3]
        spawn_point = carla.Transform()
        spawn_point.location.x = start[0]
        spawn_point.location.y = start[1]
        spawn_point.location.z = 0.01
        spawn_point.rotation.yaw = start[2]
        if self.player is not None:
            self.destroy()
            self.player = self.world.try_spawn_actor(self.car_blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print("There are no spawn points available in your map/town.")
                print("Please add some Vehicle Spawn Point to your UE4 scene.")
                sys.exit(1)
            self.player = self.world.try_spawn_actor(self.car_blueprint, spawn_point)
            # self.world.wait_for_tick()
            self.modify_vehicle_physics(self.player)

        # Set up other agents
        scenario_type = self.scenario[0]
        obstacles = self.scenario[1]
        # print(obstacles[0][1])
        # TODO remove code duplication
        if scenario_type == "01_int":
            self.choice = None
            self.setup_01_int(obstacles, conf)
            # cam_transform = carla.Transform(
            #     carla.Location(
            #         spawn_point.location.x,
            #         spawn_point.location.y - 25,
            #         spawn_point.location.z + 7,
            #     ),
            #     carla.Rotation(-30, 270, 0),
            # )
            # self.world.get_spectator().set_transform(cam_transform)
            # p = self.player.get_location()
            # print(p)
            # self.world.debug.draw_point(p+carla.Location(0,-2,2), size=0.1, color=carla.Color(r=0,g=255,b=255), life_time=0)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))
        elif scenario_type == "02_int":
            self.choice = None
            self.stopped = False
            self.setup_02_int(obstacles, conf)
            # print(spawn_point.location)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 35,
                    spawn_point.location.z + 7,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))
        elif scenario_type == "03_int":
            self.choice = None
            self.stopped = False
            self.setup_03_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 20,
                    spawn_point.location.z + 7,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))
        if scenario_type == "04_int":
            self.choice = None
            self.setup_04_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 25,
                    spawn_point.location.z + 7,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            # p = self.player.get_location()
            # print(p)
            # self.world.debug.draw_point(p+carla.Location(0,-2,2), size=0.1, color=carla.Color(r=0,g=255,b=255), life_time=0)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))

        if scenario_type == "05_int":
            self.choice = None
            self.setup_05_int(obstacles, conf)
            # cam_transform = carla.Transform(
            #     carla.Location(
            #         spawn_point.location.x,
            #         spawn_point.location.y - 25,
            #         spawn_point.location.z + 7,
            #     ),
            #     carla.Rotation(-30, 270, 0),
            # )
            # self.world.get_spectator().set_transform(cam_transform)
            # p = self.player.get_location()
            # print(p)
            # self.world.debug.draw_point(p+carla.Location(0,-2,2), size=0.1, color=carla.Color(r=0,g=255,b=255), life_time=0)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))

        if scenario_type == "06_int":
            self.choice = None
            self.setup_06_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 25,
                    spawn_point.location.z + 7,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            # p = self.player.get_location()
            # print(p)
            # self.world.debug.draw_point(p+carla.Location(0,-2,2), size=0.1, color=carla.Color(r=0,g=255,b=255), life_time=0)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))

        elif scenario_type == "07_int":
            self.setup_07_int(obstacles, conf)
            # Set initial camera focus if needed
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 30,  # Adjust camera Y offset
                    spawn_point.location.z + 10,  # Adjust camera Z offset
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:  # Assuming self.random controls if velocity is set
                self.player.set_target_velocity(carla.Vector3D(0, -8, 0))  # Adjust target velocity

        elif scenario_type == "01_non_int":
            self.choice = None
            self.stopped = False
            self.setup_01_non_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y + -10,
                    spawn_point.location.z + 5,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))
        elif scenario_type == "02_non_int":
            self.choice = None
            self.stopped = False
            self.setup_02_non_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y + 5,
                    spawn_point.location.z + 15,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))
        elif scenario_type == "03_non_int":
            self.choice = None
            self.stopped = False
            self.setup_03_non_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 20,
                    spawn_point.location.z + 15,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))

        elif scenario_type == "04_non_int":
            self.choice = None
            self.stopped = False
            self.setup_04_non_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 20,
                    spawn_point.location.z + 15,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))

        elif scenario_type == "05_non_int":
            self.choice = None
            self.stopped = False
            self.setup_05_non_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 20,
                    spawn_point.location.z + 15,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))

        elif scenario_type == "06_non_int":
            self.choice = None
            self.stopped = False
            self.setup_06_non_int(obstacles, conf)
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y - 20,
                    spawn_point.location.z + 15,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
            if not self.random:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))

        elif scenario_type in [1, 2, 4, 5]:
            # Single pedestrian scenarios
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, self.ped_speed, 0), 1))
            cam_transform = carla.Transform(
                carla.Location(
                    spawn_point.location.x,
                    spawn_point.location.y + 5,
                    spawn_point.location.z + 15,
                ),
                carla.Rotation(-30, 270, 0),
            )
            self.world.get_spectator().set_transform(cam_transform)
        elif scenario_type == 6:
            # Single pedestrian scenarios
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(
                carla.WalkerControl(carla.Vector3D(0, -self.ped_speed, 0), 1),
            )
        elif scenario_type in [3, 7, 8]:
            # Single pedestrian scenarios with parked car
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
        elif scenario_type == 10:
            # Single pedestrian with incoming car
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(
                carla.WalkerControl(carla.Vector3D(0, -self.ped_speed, 0), 1),
            )
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
        elif scenario_type == 9:
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
        elif scenario_type == 11:
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.player.set_target_velocity(carla.Vector3D(0, 20 * 0.2778, 0))
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
            self.parked_cars = []
            car_spawn_point = obstacles[2][1]
            car_spawn_point.location.y -= 7
            for _ in range(12):
                car_spawn_point.location.y += 7
                parked_car = None
                while parked_car is None:
                    parked_car = self.world.try_spawn_actor(
                        self.get_random_blueprint(),
                        car_spawn_point,
                    )
                self.parked_cars.append(parked_car)
        elif scenario_type == 12:
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
            self.parked_cars = []
            parked_car = None
            while parked_car is None:
                parked_car = self.world.try_spawn_actor(obstacles[2][0], obstacles[2][1])
            self.parked_cars.append(parked_car)

        elif scenario_type == 0:
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])

        # Set up the sensors.
        ### Set walker flags ##
        self.walker.on_street = False

        # '05_int', '06_int' are handled in their setup
        if scenario_type in ["01_int", "02_int", "03_int", "04_int"]:
            self.walker.icr = ICR.LOW
            self.walker.son = SON.AVERTING

        else:
            self.walker.icr = ICR.INTERESTED
            self.walker.son = SON.YIELDING
        # self.walker.initial_son = "Test"
        #######################
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)

        if self.camera:
            print("Current cam_pos_index is", cam_pos_index)
            self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=True, force_respawn=True)
            actor_type = get_actor_display_name(self.player)
            self.hud.notification(actor_type)

        self.semseg_sensor = CameraManager(self.player, self.hud, self._gamma)
        self.semseg_sensor.transform_index = semseg_pos_index
        self.semseg_sensor.set_sensor(semseg_index, notify=False)

    def tick(self, clock):
        self.counter += 1
        self.hud.tick(self, clock)
        dist_walker = abs(self.player.get_location().y - self.walker.get_location().y)
        car_velocity = self.player.get_velocity()
        car_speed = np.sqrt(car_velocity.x**2 + car_velocity.y**2)
        if not self.drawn:
            self.drawn = True

        # Run the camera update function if it is defined. If it's not defined, then the default
        # camera will be used
        if self.update_camera_func is not None:
            if self.update_camera_args is not None:
                self.update_camera_func(**self.update_camera_args)
            else:
                self.update_camera_func()

        # if dist_walker < self.ped_distance:  # and car_speed > 0:
        if self.scenario[0] in [1, 2, 3]:
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(self.ped_speed, 0, 0), 1))
            if self.scenario[0] in [1, 3] and self.walker.get_location().x > 4.5:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
            if self.scenario[0] == 2 and self.walker.get_location().x > 95.0:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
        elif self.scenario[0] in [4, 5, 7, 8, 6]:
            self.walker.apply_control(
                carla.WalkerControl(carla.Vector3D(-self.ped_speed, 0, 0), 1),
            )
            if self.walker.get_location().x < -4.5 and self.scenario[0] in [4, 7, 8]:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
            if self.scenario[0] in [5, 6] and self.walker.get_location().x < 85.0:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
        elif self.scenario[0] == 10:
            self.walker.apply_control(
                carla.WalkerControl(carla.Vector3D(-self.ped_speed, 0, 0), 1),
            )
        elif self.scenario[0] == 9:
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, self.ped_speed, 0), 1))
        if self.scenario[0] == 10:
            flag = (
                0 < (self.walker.get_location().y - self.incoming_car.get_location().y) < 5
            ) and (self.walker.get_location().x > -4.4)
            if self.incoming_car.get_location().y > 250 or flag:
                self.incoming_car.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                self.incoming_car.set_target_velocity(
                    carla.Vector3D(0, 9, 0),
                )  # Set target velocity for experiment
        if self.scenario[0] == 11:
            # self.incoming_car.set_target_velocity(carla.Vector3D(0, -20 * 0.2778, 0))
            if self.incoming_car.get_location().y - self.player.get_location().y < 10:
                self.incoming_car.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                self.incoming_car.set_target_velocity(
                    carla.Vector3D(0, -self.ped_speed * 0.2778, 0),
                )
        if self.scenario[0] == 12:
            # pass
            self.incoming_car.set_target_velocity(carla.Vector3D(0, self.ped_speed * 0.2778, 0))
        if self.scenario[0] == "01_int":
            status = self.path_controller_1.step()
            self.look_behind_right.step()
            self.turn_head.step()
            if self.dummy_car:
                self.car_controller.step()

            if self.choice == "Left":
                self.look_behind_left.step()
                if status == "Done":
                    self.reset.step()
                    self.path_controller_3.step()
            elif self.choice == "Right":
                self.reset.step()
                if status == "Done":
                    self.path_controller_2.step()
            elif l2_distance(self.walker.get_location(), self.desc_p) < 0.1:
                distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2
                # distance = l2_distance(self.walker.get_location(), self.player.get_location())
                # print(distance)
                # self.compute_collision_point()
                if self.decision_trigger(distance, self.db):
                    self.choice = "Left"
                    self.walker.icr = ICR.VERY_LOW
                    self.walker.son = SON.AVERTING
                else:
                    self.choice = "Right"
                    self.walker.icr = ICR.GOING_TO
                    # self.walker.con = SON.YIELDING
            self.relaxer.step()
            self.iss_crossed.step()

        if self.scenario[0] == "02_int":
            # print(self.walker.get_control().speed)
            status = self.path_controller_1.step()
            self.turn_head.step()
            self.look_right.step()
            if self.dummy_car:
                self.car_controller.step()

            if self.choice == "Stop":
                if status == "Done" and not self.stopped:
                    self.set_walker_speed_relative(0.0)
                    self.stopped = True
                    if self.choice == "Stop":
                        self.path_controller_2.cur_speed = self.path_controller_1.cur_speed
                    else:
                        self.path_controller_2.cur_speed = self.path_controller_1.cur_speed
                    self.path_controller_2.speed_schedule = self.speed_schedule_stop
                elif status == "Done":
                    distance = y_distance(self.walker.get_location(), self.player.get_location())
                    if self.second_decider(distance, 5):
                        self.choice = "Cross"
                        self.walker.blend_pose(0)
                        self.path_controller_2.step()
                        self.walker.icr = ICR.GOING_TO
                        self.iss_crossed.step()
            elif self.choice == "Cross":
                if status == "Done":
                    self.walker.blend_pose(0)
                    self.path_controller_2.step()
                    self.iss_crossed.step()
            elif self.choice is None:
                if l2_distance(self.walker.get_location(), self.desc_p) < 0.1:
                    distance = (
                        y_distance(self.walker.get_location(), self.player.get_location()) - 2
                    )
                    # distance = l2_distance(self.walker.get_location(), self.player.get_location())
                    # print(distance)
                    # self.compute_collision_point()
                    if self.decision_trigger(distance, self.db):
                        self.choice = "Stop"
                        self.path_controller_1.set_walker_speed_relative(0.5)
                        self.look_right.relax_spine()
                        self.walker.icr = ICR.VERY_LOW
                        # self.walker.son = SON.AVERTING
                    else:
                        self.path_controller_1.set_walker_speed_relative(1.0752)
                        self.path_controller_2.cur_speed = self.path_controller_1.cur_speed
                        self.path_controller_2.speed_schedule = self.speed_schedule_cross
                        self.choice = "Cross"
                        self.walker.icr = ICR.GOING_TO

            relax = self.relaxer.step()
            if relax and self.choice is None:
                self.path_controller_1.speed_schedule = None
                self.path_controller_2.speed_schedule = None
                self.path_controller_1.cur_speed = self.ped_speed
                self.path_controller_2.cur_speed = self.ped_speed
                self.walker.icr = ICR.GOING_TO
                self.walker.son = SON.AVERTING
                self.choice = "Cross"
                # print("######################################################")
                # print("relax")
                # print("######################################################")

        if self.scenario[0] == "03_int":
            if self.dummy_car:
                self.car_controller.step()
            if self.init_char == "forcing":
                status = self.path_controller_1.step()
                self.turn_head.step()
                if self.choice == "Stop":
                    # print("Status", status, "Choice", self.choice, "Stopped", self.stopped)
                    if status == "Done" and not self.stopped:
                        self.path_controller_2.cur_speed = 0.0
                        self.path_controller_1.cur_speed = 0.0
                        self.set_walker_speed_relative(0.0)
                        self.stopped = True
                        # print("Stopped")
                    elif status == "Done":
                        distance = (
                            y_distance(self.walker.get_location(), self.player.get_location()) + 10
                        )
                        if self.second_decider(distance, 10):
                            self.walker.blend_pose(0)
                            self.path_controller_2.cur_speed = self.ped_speed
                            self.path_controller_2.step()
                            self.walker.icr = ICR.GOING_TO
                            self.choice = "Cross"
                elif self.choice == "Cross":
                    # self.walker.blend_pose(0)
                    self.path_controller_2.step()
                else:
                    if (
                        l2_distance(self.walker.get_location(), self.flip_p) < 0.1
                        and self.flip_choice is None
                    ):
                        distance = (
                            y_distance(self.walker.get_location(), self.player.get_location()) - 2
                        )

                        if self.decision_trigger(
                            distance,
                            self.slow_db,
                            without_speed=True,
                        ):  # distance >=self.slow_db[0] and distance <= self.slow_db[1]:
                            self.flip_choice = "Error"
                            self.set_walker_speed_relative(0.7)
                            self.path_controller_1.cur_speed = (
                                self.path_controller_1.cur_speed * 0.7
                            )
                            self.turn_head.relax_spine()
                            self.walker.icr = ICR.INTERESTED
                            self.walker.son = SON.YIELDING
                        else:
                            self.flip_choice = "StandardAcc"
                            # print(self.flip_choice)
                            self.set_walker_speed_relative(1.1)
                            self.path_controller_1.cur_speed = (
                                self.path_controller_1.cur_speed * 1.1
                            )
                            self.turn_head.lean_forward(1.2)
                            self.walker.icr = ICR.PLANNING_TO

                    if (
                        l2_distance(self.walker.get_location(), self.acc_p) < 0.1
                        and self.flip_choice == "Error"
                    ):
                        distance = (
                            y_distance(self.walker.get_location(), self.player.get_location()) - 2
                        )
                        if self.decision_trigger(
                            distance,
                            self.acc_db,
                            without_speed=True,
                        ):  # distance >=self.acc_db[0] and distance <= self.acc_db[1]:
                            self.path_controller_1.cur_speed = (
                                self.path_controller_1.cur_speed * 1.0 / 0.7 * 1.2
                            )
                            self.set_walker_speed_relative(1.0 / 0.7 * 1.2)
                            self.turn_head.lean_forward(1)
                            self.flip_choice = "Accelerated"

                            self.walker.icr = ICR.PLANNING_TO
                            self.walker.son = SON.FORCING
                        else:
                            self.flip_choice = "Keep"
                        # print(self.flip_choice)
                    if l2_distance(self.walker.get_location(), self.desc_p) < 0.1:
                        distance = (
                            y_distance(self.walker.get_location(), self.player.get_location()) - 2
                        )
                        # print("Desc_p")
                        if self.decision_trigger(
                            distance,
                            self.db,
                        ):  # distance >=self.db[0] and distance <= self.db[1]:
                            self.choice = "Stop"
                            self.cur_speed = self.path_controller_1.cur_speed
                            self.path_controller_1.cur_speed = (
                                self.path_controller_1.cur_speed * 0.8
                            )
                            self.path_controller_1.speed_schedule = self.speed_schedule_stop
                            self.path_controller_1.set_walker_speed_relative(0.8)
                            self.turn_head.relax_spine()
                            self.walker.icr = ICR.VERY_LOW
                            # self.walker.son = SON.AVERTING
                        else:
                            self.choice = "Cross"
                            self.walker.icr = ICR.GOING_TO
                        # print(distance, self.choice)
            else:
                status = self.path_controller_1.step()
                self.turn_head.step()
                if self.choice == "Stop":
                    # print("Status", status, "Choice", self.choice, "Stopped", self.stopped)
                    if status == "Done" and not self.stopped:
                        self.path_controller_2.cur_speed = 0.0
                        self.path_controller_1.cur_speed = 0.0
                        self.set_walker_speed_relative(0.0)
                        self.stopped = True
                        # print("Stopped")
                    elif status == "Done":
                        distance = (
                            y_distance(self.walker.get_location(), self.player.get_location()) + 10
                        )
                        if self.second_decider(distance, 20):  # distance < 0:
                            self.walker.blend_pose(0)
                            self.path_controller_2.cur_speed = self.ped_speed
                            self.path_controller_2.step()
                            self.walker.icr = ICR.GOING_TO
                            self.choice = "Cross"
                elif self.choice == "Cross":
                    # self.walker.blend_pose(0)
                    self.path_controller_2.step()
                elif l2_distance(self.walker.get_location(), self.desc_p) < 0.1:
                    distance = (
                        y_distance(self.walker.get_location(), self.player.get_location()) - 2
                    )
                    if self.decision_trigger(
                        distance,
                        self.db,
                    ):  # distance >=self.db[0] and distance <= self.db[1]:
                        self.choice = "Stop"
                        self.cur_speed = self.path_controller_1.cur_speed
                        self.path_controller_1.cur_speed = self.path_controller_1.cur_speed * 0.95
                        self.path_controller_1.speed_schedule = self.speed_schedule_stop
                        self.path_controller_1.set_walker_speed_relative(0.95)
                        self.turn_head.relax_spine()
                        self.walker.icr = ICR.VERY_LOW
                        # self.walker.son = SON.AVERTING
                    else:
                        self.choice = "Cross"
                        self.turn_head.lean_forward(1.2)
                        self.walker.icr = ICR.GOING_TO
            self.iss_crossed.step()
            relax = self.relaxer.step()
            if relax and self.choice is None:
                self.path_controller_1.speed_schedule = None
                self.path_controller_1.cur_speed = self.ped_speed
                self.path_controller_2.speed_schedule = None
                self.path_controller_2.cur_speed = self.ped_speed
                self.walker.son = SON.AVERTING

        if self.scenario[0] == "04_int":
            status = self.path_controller_1.step()
            if self.dummy_car:
                self.car_controller.step()
            # self.look_behind_right.step()
            self.turn_head.step()
            self.resetLD1.step()

            self.turn_head_second.step()
            if self.walker.initial_son == SON.FORCING:
                self.lean_forward.step()

            if status == "Done":
                # self.reset.step()
                status2 = self.path_controller_2.step()

                if status2 == "Done":
                    if self.choice == "Back":
                        # self.reset.step()

                        # self.reset.step()
                        # make walker run backwards
                        self.walker.set_transform(
                            carla.Transform(
                                self.walker.get_transform().location,
                                carla.Rotation(0, 0, 0),
                            ),
                        )
                        status3 = self.path_controller_4.step()
                        if status3 == "Done":
                            self.reset.step()
                            self.path_controller_1.cur_speed = 0.0
                            self.path_controller_2.cur_speed = 0.0
                            self.path_controller_3.cur_speed = 0.0
                            self.path_controller_4.cur_speed = 0.0
                            self.set_walker_speed_relative(0.0)
                    elif self.choice == "Continue":
                        # self.reset.step()
                        self.resetLD2.step()
                        status3 = self.path_controller_3.step()
                        if status3 == "Done":
                            self.reset.step()
                            self.path_controller_1.cur_speed = 0.0
                            self.path_controller_2.cur_speed = 0.0
                            self.path_controller_3.cur_speed = 0.0
                            self.path_controller_4.cur_speed = 0.0
                            self.set_walker_speed_relative(0.0)

                    # self.choice = "Continue"
                    # self.walker.icr = ICR.GOING_TO
                    elif l2_distance(self.walker.get_location(), self.desc_p) < 0.2:
                        distance = (
                            y_distance(self.walker.get_location(), self.player.get_location()) - 2
                        )
                        # distance = l2_distance(self.walker.get_location(), self.player.get_location())
                        # print(distance)
                        # self.compute_collision_point()
                        if self.decision_trigger(distance, self.db):
                            self.choice = "Back"
                            self.walker.icr = ICR.VERY_LOW
                            self.walker.son = SON.YIELDING
                        else:
                            self.choice = "Continue"
                            self.walker.icr = ICR.GOING_TO
                            self.walker.son = SON.FORCING

            self.relaxer.step()
            self.iss_crossed.step()
            self.starts_crossing.step()
            self.curd_stat.step()
        if self.scenario[0] == "05_int":
            status = self.path_controller_1.step()
            self.uncertain.step()
            if self.dummy_car:
                self.car_controller.step()

            if status == "Done":
                status2 = self.path_controller_2.step()
                if status2 == "Done":
                    if self.choice == "Continue":
                        self.lean_forward.step()
                        self.path_controller_3.step()
                    elif self.choice == "Stop":
                        self.path_controller_2.cur_speed = 0.0
                        self.path_controller_1.cur_speed = 0.0
                        self.set_walker_speed_relative(0.0)
                    elif l2_distance(self.walker.get_location(), self.desc_p) < 0.2:
                        distance = (
                            y_distance(self.walker.get_location(), self.player.get_location()) - 2
                        )
                        if self.decision_trigger(distance, self.db):
                            self.choice = "Stop"
                            self.walker.icr = ICR.VERY_LOW
                            self.walker.son = SON.YIELDING
                        else:
                            self.choice = "Continue"
                            self.walker.icr = ICR.GOING_TO
                            self.walker.son = SON.FORCING
            self.relaxer.step()
            self.iss_crossed.step()

        if self.scenario[0] == "06_int":
            status = self.path_controller_1.step()
            if self.dummy_car:
                self.car_controller.step()

            # self.look_behind_right.step()

            # self.turn_head.step()

            if status == "Done":
                if self.choice == "Avoid":
                    self.reset.step()
                    self.path_controller_3.step()
                elif self.choice == "Continue":
                    self.raise_arm.step()
                    self.reset.step()
                    self.path_controller_2.step()
                # if self.walker.initial_son == SON.FORCING:
                #     self.choice = "Continue"
                #     self.walker.icr = ICR.GOING_TO
                # elif self.walker.initial_son == SON.YIELDING:
                #     self.choice = "Avoid"
                #     self.walker.icr = ICR.PLANNING_TO

                elif l2_distance(self.walker.get_location(), self.desc_p) < 0.2:
                    distance = (
                        y_distance(self.walker.get_location(), self.player.get_location()) - 2
                    )
                    if self.decision_trigger(distance, self.db):
                        self.choice = "Avoid"
                        self.walker.icr = ICR.GOING_TO
                        self.walker.son = SON.YIELDING
                    else:
                        self.choice = "Continue"
                        self.walker.icr = ICR.GOING_TO
                        self.walker.son = SON.FORCING
            self.iss_crossed.step()
            self.iss_crossed_2.step()

            self.relaxer.step()

        elif self.scenario[0] == "07_int":
            # Scenario 7: Walk -> Stop -> Look -> Raise Hand -> Decide -> Sprint/Wait Logic
            current_time = time.time()
            hand_status = "Idle"  # Default status
            look_status = "Idle"  # Default status

            # 1. Walk to Curb Point
            if not self.at_curb:
                status1 = self.path_controller_1.step()
                if status1 == "Done":
                    self.at_curb = True
                    self.walker.apply_control(
                        carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0)
                    )  # Stop
                    logger.info("Scenario 07: Reached curb point, stopping.")
                    # Reset flags for next phases
                    self.look_started = False
                    self.look_finished = False
                    self.hand_raise_started = False
                    self.hand_raise_finished = False
                    self.decided_action = None

            # 2. Look Across Street (if at curb and not started/finished)
            elif self.at_curb and not self.look_started:
                look_status = self.look_across_street_left.step()
                if look_status != "Idle" and look_status != "Done":  # i.e., "Running"
                    self.look_started = True
                    logger.info("Scenario 07: Starting look across street.")
                elif look_status == "Done":  # Can happen if duration is tiny
                    self.look_started = True
                    self.look_finished = True
                    logger.info("Scenario 07: Look finished immediately.")

            # 3. Wait for Look to Finish
            elif self.at_curb and self.look_started and not self.look_finished:
                look_status = self.look_across_street_left.step()
                if look_status == "Done":
                    self.look_finished = True
                    logger.info("Scenario 07: Look finished.")
                    # Immediately try to start hand raise
                    self.hand_raise_started = False
                    self.hand_raise_finished = False

            # 4. Raise Hand (if look finished and hand raise not started/finished)
            elif self.at_curb and self.look_finished and not self.hand_raise_started:
                hand_status = self.raise_hand_briefly.step()
                if hand_status != "Idle" and hand_status != "Done":
                    self.hand_raise_started = True
                    logger.info("Scenario 07: Starting hand raise.")
                elif hand_status == "Done":
                    self.hand_raise_started = True
                    self.hand_raise_finished = True
                    self.wait_start_time = current_time  # Start decision timer
                    logger.info("Scenario 07: Hand raise finished immediately.")

            # 5. Wait for Hand Raise to Finish
            elif self.at_curb and self.hand_raise_started and not self.hand_raise_finished:
                hand_status = self.raise_hand_briefly.step()
                if hand_status == "Done":
                    self.hand_raise_finished = True
                    self.wait_start_time = current_time  # Start decision timer NOW
                    logger.info("Scenario 07: Hand raise finished.")

            # 6. Make Decision (only if at curb, look finished, hand raise finished, and no decision yet)
            elif (
                self.at_curb
                and self.look_finished
                and self.hand_raise_finished
                and self.decided_action is None
            ):
                distance_to_car = y_distance(self.curb_point, self.player.get_location()) - 2
                car_in_db = self.decision_trigger(distance_to_car, self.db)
                time_elapsed_decision = (
                    current_time - self.wait_start_time
                )  # Time since hand raise finished

                # Reset pose BEFORE deciding, so pedestrian looks neutral during decision/wait
                self.reset_pose.step()

                if self.char == "yielding":
                    if car_in_db:
                        if time_elapsed_decision < self.wait_duration:
                            self.decided_action = "Wait"
                            logger.info("Scenario 07: Decided to Wait (Yielding, Car Close).")
                            self.walker.icr = ICR.PLANNING_TO
                            self.walker.son = SON.YIELDING
                        else:
                            self.decided_action = "Sprint"  # Waited long enough
                            logger.info("Scenario 07: Yielding timeout, deciding to Sprint.")
                    else:  # Car is far
                        self.decided_action = "Sprint"
                        logger.info("Scenario 07: Yielding, car far, deciding to Sprint.")
                elif self.char == "forcing":
                    self.decided_action = "Sprint"  # Forcing always sprints
                    logger.info("Scenario 07: Forcing, deciding to Sprint.")

                # Apply sprint speed immediately if sprinting
                if self.decided_action == "Sprint":
                    sprint_speed = self.ped_speed * self.sprint_multiplier
                    self.path_controller_2.cur_speed = sprint_speed
                    logger.info(f"Scenario 07: Setting sprint speed: {sprint_speed:.2f} m/s")
                    self.walker.icr = ICR.GOING_TO
                    self.walker.son = SON.FORCING
                    # Start moving immediately by calling step controller
                    self.path_controller_2.step()

            # 7. Execute Action (Wait or Sprint)
            elif self.decided_action == "Wait":
                self.walker.apply_control(
                    carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0)
                )  # Stay stopped
                # Re-evaluate periodically
                distance_to_car = y_distance(self.curb_point, self.player.get_location()) - 2
                car_in_db = self.decision_trigger(distance_to_car, self.db)
                time_elapsed_decision = current_time - self.wait_start_time
                if not car_in_db or time_elapsed_decision >= self.wait_duration:
                    logger.info("Scenario 07: Wait finished or car passed, switching to sprint.")
                    self.decided_action = "Sprint"
                    sprint_speed = self.ped_speed * self.sprint_multiplier
                    self.path_controller_2.cur_speed = sprint_speed
                    logger.info(f"Scenario 07: Setting sprint speed: {sprint_speed:.2f} m/s")
                    self.walker.icr = ICR.GOING_TO
                    self.walker.son = SON.FORCING
                    # Start moving immediately
                    self.path_controller_2.step()

            elif self.decided_action == "Sprint":
                status2 = self.path_controller_2.step()
                if status2 == "Done":
                    logger.info("Scenario 07: Finished crossing (sprint path).")
                    self.decided_action = "CrossingDone"
                    # Start optional continuation path
                    self.path_controller_3.step()
                    self.walker.icr = ICR.VERY_LOW
                    self.walker.son = SON.AVERTING

            # 8. Optional Continuation after Crossing
            elif self.decided_action == "CrossingDone":
                status3 = self.path_controller_3.step()
                if status3 == "Done":
                    self.decided_action = "FinishedScenario"
                    self.walker.apply_control(
                        carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0)
                    )

            # Update dummy car controller
            if self.dummy_car and hasattr(self, "car_controller"):
                self.car_controller.step()

        if self.scenario[0] == "01_non_int":
            self.path_controller_1.step()
            self.look_behind_right.step()
            self.reset.step()
            self.iss_crossed.step()
            self.going_to.step()

        if self.scenario[0] == "02_non_int":
            self.path_controller_1.step()
            self.look_behind_left.step()
            self.reset.step()
            self.iss_crossed.step()
            self.going_to.step()

        if self.scenario[0] == "03_non_int":
            self.path_controller_1.step()
            self.lean_forward.step()
            self.iss_crossed.step()
            # self.look_behind_left.step()
            # self.reset.step()

        if self.scenario[0] == "04_non_int":
            self.path_controller_1.step()
            self.lean_forward.step()
            self.iss_crossed.step()
            self.look_right.step()
            self.reset.step()

        if self.scenario[0] == "05_non_int":
            self.path_controller_1.step()
            self.lean_forward.step()
            self.iss_crossed.step()
            self.look_left.step()
            self.reset.step()

        if self.scenario[0] == "06_non_int":
            self.path_controller_1.step()
            self.lean_forward.step()
            self.iss_crossed.step()
            self.look_right.step()
            self.reset.step()

    def second_decider(self, distance, dec_d=None):
        if self.random:
            if self.second_choice:
                simulation_step = 0.05
                self.waiting_c += 1
                return self.waiting_c * simulation_step > self.waiting_time
            self.waiting_time = np.random.random() * 2 + 1
            self.waiting_c = 0
            self.second_choice = True
        else:
            velocity = self.player.get_velocity()
            speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
            if dec_d is None:
                return distance < 0 or speed < 1  # less than 3.6kmh
            return distance + dec_d < 0 or (speed < 1 and distance > 2.5)

    def decision_trigger(self, distance, db, without_speed=False):
        if self.random:
            choice = np.random.choice(2)
            choices = [np.random.choice(2) for i in range(10)]
            return choice == 1
        velocity = self.player.get_velocity()
        speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
        return distance >= db[0] and distance <= db[1] and (speed > 1.5 or without_speed)

    def get_walker_state(self):
        loc = self.walker.get_location()
        x, y = loc.x, loc.y
        return (x, y, self.walker.icr, self.walker.son)

    def get_walker_state_full(self):
        """Get all pedestrian-related observables for the DBN."""
        loc = self.walker.get_location()
        x, y = loc.x, loc.y

        # Basic state information - directly available from walker
        icr_ped = self.walker.icr
        sn_ped = self.walker.son

        # Velocity and speed
        vel = self.walker.get_velocity()
        speed = (vel.x**2 + vel.y**2) ** 0.5

        # Precise head orientation based on active controllers
        ho_ped = "Ignoring"  # Default
        if (
            (
                hasattr(self, "look_behind_right")
                and hasattr(self.look_behind_right, "done")
                and self.look_behind_right.done
            )
            or (
                hasattr(self, "look_behind_left")
                and hasattr(self.look_behind_left, "done")
                and self.look_behind_left.done
            )
            or (
                hasattr(self, "turn_head")
                and hasattr(self.turn_head, "done")
                and self.turn_head.done
            )
        ):
            ho_ped = "Facing"

        # Calculate relative to car
        walker_rotation = self.walker.get_transform().rotation.yaw
        car_loc = self.player.get_location()
        walker_loc = self.walker.get_location()
        angle_to_car = (
            math.atan2(car_loc.y - walker_loc.y, car_loc.x - walker_loc.x) * 180 / math.pi
        )
        angle_diff = abs((walker_rotation - angle_to_car + 180) % 360 - 180)

        if angle_diff < 45:
            bo_ped = "Facing"
        elif angle_diff > 135:
            bo_ped = "Averting"
        else:
            bo_ped = "Ignoring"  # Default

        # Hip orientation based on specific animation controllers
        hio_ped = "Neutral"  # Default
        if (
            hasattr(self, "lean_forward")
            and hasattr(self.lean_forward, "done")
            and self.lean_forward.done
        ):
            hio_ped = "Leaning forward"

        # Discretize speed based on observed ranges in the config files
        if speed < 0.5:
            speed_class = "Stopped"
        elif speed < 1.2:
            speed_class = "Slow"
        elif speed < 1.8:
            speed_class = "Normal"
        elif speed < 2.2:
            speed_class = "Fast"
        else:
            speed_class = "VeryFast"

        return {
            "ICR_ped": icr_ped,
            "SN_ped": sn_ped,
            "HO_ped": ho_ped,
            "BO_ped": bo_ped,
            "HIO_ped": hio_ped,
            "S_ped": speed_class,
            "speed_ped_raw": speed,  # Store raw speed for acceleration calculation
        }

    def set_walker_speed_relative(self, per):
        control = self.walker.get_control()
        control.speed = per * control.speed
        self.walker.apply_control(control)

    def get_car_state(self):
        loc = self.player.get_location()
        x, y = loc.x, loc.y
        return (x, y)

    def get_car_state_full(self):
        """Get all car-related observables for the DBN."""
        loc = self.player.get_location()
        x, y = loc.x, loc.y

        # Velocity and speed
        vel = self.player.get_velocity()
        speed = (vel.x**2 + vel.y**2) ** 0.5

        # Discretize speed into classes - based on configuration ranges seen in config.py
        if speed < 1.0:
            speed_class = "Stopped"
        elif speed < 5.0:
            speed_class = "Slow"
        elif speed < 8.0:
            speed_class = "Medium"
        else:
            speed_class = "Fast"

        # Get acceleration if available
        accel = None
        accel_magnitude = 0
        is_accelerating = False
        is_decelerating = False
        lateral_movement = False

        if hasattr(self.player, "get_acceleration"):
            accel = self.player.get_acceleration()
            accel_magnitude = (accel.x**2 + accel.y**2) ** 0.5

            # Determine if car is accelerating or decelerating in its forward direction
            forward_vector = self.player.get_transform().get_forward_vector()

            # Calculate dot product to see if acceleration is along forward direction
            accel_dot_forward = accel.x * forward_vector.x + accel.y * forward_vector.y

            is_accelerating = accel_dot_forward > 0.1  # Threshold for acceleration
            is_decelerating = accel_dot_forward < -0.1  # Threshold for deceleration

            # Check for significant lateral movement (perpendicular to forward direction)
            lateral_accel = abs(accel.x * forward_vector.y - accel.y * forward_vector.x)
            lateral_movement = lateral_accel > 0.5  # Threshold for lateral movement

        # Get steering if available
        steering_value = 0
        if hasattr(self.player, "get_control"):
            control = self.player.get_control()
            if hasattr(control, "steer"):
                steering_value = abs(control.steer)

        # Car's strategy of negotiation - directly from car_controller if available
        # TODO: Introduce "Avoiding", as Strategy of Negotiation (SN) = {Avoiding, Yielding, Forcing}
        if hasattr(self, "car_controller") and hasattr(self.car_controller, "choice"):
            sn_car = "Yielding" if self.car_controller.choice else "Forcing"
        else:
            # Infer based on behavior
            if speed < 2.0 and hasattr(self.player, "get_acceleration"):
                accel = self.player.get_acceleration()
                accel_magnitude = (accel.x**2 + accel.y**2) ** 0.5
                sn_car = "Yielding" if accel_magnitude < 0 else "Forcing"
            else:
                sn_car = "Forcing" if speed > 6.0 else "Yielding"

        # Car's intention to claim the road (map from car_controller.py speed and yielding)
        if hasattr(self, "car_controller") and hasattr(self.car_controller, "speed"):
            target_speed = self.car_controller.speed
            if target_speed < 1.0 or (self.car_controller.choice and speed < 1.0):
                icr_car = "Very low"
            elif target_speed < 3.0 or (self.car_controller.choice and speed < 3.0):
                icr_car = "Low"
            elif target_speed < 6.0:
                icr_car = "Interested"
            elif target_speed < 9.0:
                icr_car = "Planning to"
            else:
                icr_car = "Going to"
        else:
            # Fallback to speed-based estimation
            if speed < 1.0:
                icr_car = "Very low"
            elif speed < 3.0:
                icr_car = "Low"
            elif speed < 6.0:
                icr_car = "Interested"
            elif speed < 9.0:
                icr_car = "Planning to"
            else:
                icr_car = "Going to"

        # Car orientation observables
        car_transform = self.player.get_transform()
        car_rotation = car_transform.rotation.yaw
        walker_loc = self.walker.get_location()
        car_loc = self.player.get_location()
        angle_to_walker = (
            math.atan2(walker_loc.y - car_loc.y, walker_loc.x - car_loc.x) * 180 / math.pi
        )
        angle_diff = abs((car_rotation - angle_to_walker + 180) % 360 - 180)

        if angle_diff < 45:
            ws_car = "Facing"
            cbo_car = "Facing"
        elif angle_diff > 135:
            ws_car = "Averting"
            cbo_car = "Averting"
        else:
            ws_car = "Ignoring"
            cbo_car = "Ignoring"

        return {
            "SN_car": sn_car,
            "ICR_car": icr_car,
            "WS_car": ws_car,
            "CBO_car": cbo_car,
            "S_car": speed_class,
            "speed_car_raw": speed,  # Store raw speed for acceleration calculation
        }

    def calculate_derived_observables(self, prev_data=None):
        """Calculate derived observables like distance, approaching status, and accelerations."""
        # Get current states
        walker_state = self.get_walker_state_full()
        car_state = self.get_car_state_full()

        # Calculate distance between car and pedestrian
        car_loc = self.player.get_location()
        walker_loc = self.walker.get_location()
        distance = l2_distance(car_loc, walker_loc)

        # Discretize distance - based on ranges seen in config files
        if distance < 3.0:
            distance_class = "VeryClose"
        elif distance < 8.0:
            distance_class = "Close"
        elif distance < 15.0:
            distance_class = "Medium"
        elif distance < 25.0:
            distance_class = "Far"
        else:
            distance_class = "VeryFar"

        # Calculate approaching status and acceleration
        if prev_data is not None:
            # Car approaching pedestrian?
            prev_distance = prev_data["D_raw"]
            a_car = "Yes" if distance < prev_distance else "No"

            # For pedestrian approaching car, use dot product calculation
            car_vel = self.player.get_velocity()
            walker_vel = self.walker.get_velocity()

            # Check if pedestrian is moving toward car using dot product calculation
            ped_speed = walker_state["speed_ped_raw"]
            if ped_speed > 0.1:
                # Calculate normalized velocity vector
                walker_dir = [walker_vel.x / ped_speed, walker_vel.y / ped_speed]

                # Calculate vector from pedestrian to car
                ped_to_car = [car_loc.x - walker_loc.x, car_loc.y - walker_loc.y]
                ped_to_car_len = (ped_to_car[0] ** 2 + ped_to_car[1] ** 2) ** 0.5

                if ped_to_car_len > 0:
                    # Normalize vector
                    ped_to_car = [ped_to_car[0] / ped_to_car_len, ped_to_car[1] / ped_to_car_len]

                    # Calculate dot product
                    dot_product = walker_dir[0] * ped_to_car[0] + walker_dir[1] * ped_to_car[1]
                    a_ped = "Yes" if dot_product > 0.3 else "No"
                else:
                    a_ped = "No"
            else:
                a_ped = "No"

            # Calculate accelerations
            acc_car = car_state["speed_car_raw"] - prev_data["speed_car_raw"]
            acc_ped = walker_state["speed_ped_raw"] - prev_data["speed_ped_raw"]

            # Discretize accelerations
            if acc_car > 0.2:
                acc_car_class = "Accelerating"
            elif acc_car < -0.2:
                acc_car_class = "Decelerating"
            else:
                acc_car_class = "Constant"

            if acc_ped > 0.1:
                acc_ped_class = "Accelerating"
            elif acc_ped < -0.1:
                acc_ped_class = "Decelerating"
            else:
                acc_ped_class = "Constant"
        else:
            # Default values for first frame
            a_car = "No"
            a_ped = "No"
            acc_car_class = "Constant"
            acc_ped_class = "Constant"

        # TODO: Calculate sense of security as the same as ICR_ped for now
        ssec = walker_state["ICR_ped"]

        # Combine all states into a single data point
        data = {
            "SN_car": car_state["SN_car"],
            "SN_ped": walker_state["SN_ped"],
            "ICR_car": car_state["ICR_car"],
            "ICR_ped": walker_state["ICR_ped"],
            "SSEC": ssec,
            "A_car": a_car,
            "WS_car": car_state["WS_car"],
            "CBO_car": car_state["CBO_car"],
            "ACC_car": acc_car_class,
            "S_car": car_state["S_car"],
            "D": distance_class,
            "D_raw": distance,  # Raw value for next frame calculation
            "HO_ped": walker_state["HO_ped"],
            "BO_ped": walker_state["BO_ped"],
            "HIO_ped": walker_state["HIO_ped"],
            "A_ped": a_ped,
            "ACC_ped": acc_ped_class,
            "S_ped": walker_state["S_ped"],
            "speed_car_raw": car_state["speed_car_raw"],  # Raw values for next frame calculation
            "speed_ped_raw": walker_state["speed_ped_raw"],
        }

        return data

    def get_p_from_vector(self, loc1, loc2, perc):
        vec = loc2 - loc1
        return loc1 + perc * vec

    def setup_04_non_int(
        self,
        obstacles,
        conf,
    ):
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        walking_distance = conf.walking_distance
        op_reenter_distance = conf.op_reenter_distance

        # spawn car:
        self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc + carla.Location(+2, 0, 0)
        # print(spawn_loc)
        # print(obstacles[0][1].location)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        street_dist = 10.5

        offsets_1 = [
            (street_dist, op_reenter_distance * walking_distance),
            (street_dist, op_reenter_distance * walking_distance + 50 * walking_distance),
        ]

        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )

        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))

        self.turning_point = self.get_p_from_vector(spawn_loc, path_1[0], 1 - looking_distance)
        self.look_right = LookBehindRight(self.walker, spawn_loc, conf.char)
        self.reset = ResetPose(self.walker, self.turning_point)
        self.lean_forward = LeanForward(self.walker, self.turning_point)
        # self.turning_point = self.get_p_from_vector( path_1[0], path_1[1], 0.05)

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_1[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                braking_point=None,
                speed=8,
                yielding=False,
            )

    def setup_05_non_int(
        self,
        obstacles,
        conf,
    ):
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        walking_distance = conf.walking_distance
        op_reenter_distance = conf.op_reenter_distance

        # spawn car:
        self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc - carla.Location(2, 0, 0)
        # print(spawn_loc)
        # print(obstacles[0][1].location)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        street_dist = 10.5

        offsets_1 = [
            (-street_dist, op_reenter_distance * walking_distance),
            (-street_dist, op_reenter_distance * walking_distance + 50 * walking_distance),
        ]

        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )

        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))

        # self.turning_point = self.get_p_from_vector( path_1[0], path_1[1], 0.05)

        self.turning_point = self.get_p_from_vector(spawn_loc, path_1[0], 1 - looking_distance)
        self.look_left = LookBehindLeft(self.walker, spawn_loc)
        self.reset = ResetPose(self.walker, self.turning_point)
        self.lean_forward = LeanForward(self.walker, self.turning_point)
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                braking_point=None,
                speed=8,
                yielding=False,
            )

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_1[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

    def setup_06_non_int(
        self,
        obstacles,
        conf,
    ):
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        walking_distance = conf.walking_distance
        op_reenter_distance = conf.op_reenter_distance

        # spawn car:
        self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc - carla.Location(2, 0, 0)
        # print(spawn_loc)
        # print(obstacles[0][1].location)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        street_dist = 17.5

        offsets_1 = [
            (-street_dist, op_reenter_distance * walking_distance),
            (-street_dist, op_reenter_distance * walking_distance + 50 * walking_distance),
        ]

        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )

        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))

        self.turning_point = self.get_p_from_vector(spawn_loc, path_1[0], 1 - looking_distance)
        self.look_right = LookBehindRight(self.walker, spawn_loc, conf.char)
        self.reset = ResetPose(self.walker, self.turning_point)
        self.lean_forward = LeanForward(self.walker, self.turning_point)
        # self.turning_point = self.get_p_from_vector( path_1[0], path_1[1], 0.05)

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_1[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                braking_point=None,
                speed=5,
                yielding=False,
            )

    def setup_03_non_int(
        self,
        obstacles,
        conf,
    ):
        # print("03_non_int")
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        walking_distance = conf.walking_distance
        op_reenter_distance = conf.op_reenter_distance

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc + carla.Location(+2, 0, 0)
        # print(spawn_loc)
        # print(obstacles[0][1].location)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        street_dist = -10.5
        offsets_1 = [
            (0, +walking_distance),
            (street_dist, +walking_distance + op_reenter_distance),
            (street_dist, +walking_distance + op_reenter_distance + 50),
        ]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )

        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))

        self.turning_point = self.get_p_from_vector(spawn_loc, path_1[0], looking_distance)
        self.lean_forward = LeanForward(self.walker, self.turning_point)
        # self.turning_point = self.get_p_from_vector( path_1[0], path_1[1], 0.05)
        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_1[1],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                braking_point=None,
                speed=4,
                yielding=False,
            )

    def setup_02_non_int(self, obstacles, conf):
        # print("02_non_int")
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        walking_distance = conf.walking_distance
        op_reenter_distance = conf.op_reenter_distance

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc + carla.Location(-1, 0, 0)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        street_dist = +10.5
        offsets_1 = [
            (0, -walking_distance),
            (street_dist, -walking_distance - op_reenter_distance),
            (street_dist, -walking_distance - op_reenter_distance - 5),
        ]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )

        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))

        self.turning_point = self.get_p_from_vector(spawn_loc, path_1[0], looking_distance)
        self.look_behind_left = LookBehindLeftSpine(
            self.walker,
            self.turning_point,
            char="forcing",
        )

        self.turning_point = self.get_p_from_vector(path_1[0], path_1[1], 0.1)
        self.reset = ResetPose(self.walker, self.turning_point)
        self.going_to = InternalStateSetter(
            self.walker,
            path_1[0],
            icr=ICR.GOING_TO,
            son=SON.FORCING,
        )

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_1[1],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                braking_point=None,
                speed=3,
                yielding=False,
            )

    def setup_01_non_int(self, obstacles, conf):
        # print("01_non_int")
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        walking_distance = conf.walking_distance
        op_reenter_distance = conf.op_reenter_distance

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc + carla.Location(1, 0, 0)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        street_dist = -10.5
        offsets_1 = [
            (0, -walking_distance),
            (street_dist, -walking_distance - op_reenter_distance),
            (street_dist, -walking_distance - op_reenter_distance - 5),
        ]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )

        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))

        self.turning_point = self.get_p_from_vector(spawn_loc, path_1[0], looking_distance)
        self.look_behind_right = LookBehindRight(
            self.walker,
            self.turning_point,
            char="forcing",
            scenario="01_non_int",
        )
        self.turning_point = self.get_p_from_vector(path_1[0], path_1[1], 0.1)
        self.reset = ResetPose(self.walker, self.turning_point)
        self.going_to = InternalStateSetter(
            self.walker,
            path_1[0],
            icr=ICR.GOING_TO,
            son=SON.FORCING,
        )

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_1[1],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                braking_point=None,
                speed=3,
                yielding=False,
            )

    def setup_03_int(self, obstacles, conf):
        # print("03")
        self.flip_choice = None
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        self.db = [0, 20] if conf.char == "yielding" else [0, 20]
        self.slow_db = [20, 38]
        self.acc_db = [20, 38]
        self.init_char = conf.char
        self.second_choice = False

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        self.walker.on_street = False
        street_x = 95
        offsets_1 = [(street_x - spawn_loc.x, 0)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))

        offsets_2 = [(-21, 0)]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed,
        )
        if conf.char == "forcing":
            self.desc_p = self.get_p_from_vector(spawn_loc, path_1[0], 0.88)
        else:
            self.desc_p = self.get_p_from_vector(spawn_loc, path_1[0], 0.8)
        if self.debug:
            self._draw_point(self.desc_p)

        self.flip_p = self.get_p_from_vector(spawn_loc, path_1[0], 0.4)

        if self.debug:
            self._draw_point(self.flip_p, carla.Color(0, 0, 255))

        self.acc_p = self.get_p_from_vector(spawn_loc, path_1[0], 0.55)
        if self.debug:
            self._draw_point(self.acc_p, carla.Color(0, 255, 0))
            self._draw_db(db=self.slow_db)
            self._draw_db(db=self.acc_db, color=carla.Color(255, 0, 0))
            self._draw_db(self.db, color=carla.Color(0, 0, 255))
        self.turn_head = TurnHeadLeftWalk(
            self.walker,
            start_pos=self.get_p_from_vector(spawn_loc, path_1[0], looking_distance),
            char=conf.char,
        )
        self.relaxer = Relaxer(self.walker, self.player, self.flip_p)
        self.speed_schedule_stop = [
            (self.get_p_from_vector(spawn_loc, path_1[0], per), 0.85) for per in [0.87, 0.92]
        ]
        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            self.path_2[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        player_loc = self.player.get_location()
        breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
        car_behave = conf.char == "forcing"
        car_to_desc = self.player.get_location().y - self.desc_p.y - self.db[1]
        ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
        ped_time = ped_to_desc / self.ped_speed
        speed = car_to_desc / ped_time
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                breaking_point,
                speed=speed if car_behave else speed,
                yielding=car_behave,
            )

    def setup_02_int(self, obstacles, conf):
        # print("02")
        spawning_distance = conf.spawning_distance
        walking_distance = conf.walking_distance
        looking_distance = conf.looking_distance
        crossing_distance = conf.crossing_distance
        op_reenter_distance = conf.op_reenter_distance
        street_delta = 3 if conf.char == "yielding" else 5
        self.db = [-1, 15] if conf.char == "yielding" else [-1, 10]
        # mult =  1.0 if conf.char == "yielding"  else 1.1*1.1*1.1
        self.second_choice = False

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        self.walker.on_street = False

        offsets_1 = [(0, walking_distance), (street_delta, walking_distance + crossing_distance)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        offsets_2 = [
            (9.5, walking_distance + crossing_distance + op_reenter_distance),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 2),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 10),
        ]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=0, g=255, b=0) if self.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed,
        )

        turn_p = self.get_point((0, looking_distance * walking_distance))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)

        self.look_right = TurnHeadRightWalk(self.walker, path_1[0], conf.char)
        self.reset = ResetPose(self.walker)

        vec = path_1[1] - path_1[0]
        self.desc_p = path_1[0] + 0.9 * vec

        self.path_controller_1.speed_schedule = [
            (path_1[0] + per * vec, 0.93) for per in [0.0, 0.2, 0.4]
        ]

        vec_2 = self.path_2[0] - path_1[1]
        self.speed_schedule_stop = [(path_1[1] + per * vec_2, 1.355) for per in [0.0, 0.05, 0.075]]

        self.speed_schedule_cross = [(path_1[1] + per * vec_2, 1.075) for per in [0.0, 0.05]]
        if self.debug:
            self._draw_db()
            self._draw_point(self.desc_p)
            self._draw_grid()

        # self._draw_db_circle()
        # self.world.debug.draw_point(path_1[0] + 0.2 * vec, size=0.1, color=carla.Color(r=0,g=255,b=255), life_time=0)
        self.relaxer = Relaxer(self.walker, self.player, path_1[0] + 0.2 * vec)
        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            self.path_2[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        player_loc = self.player.get_location()
        breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
        car_behave = conf.char == "forcing"
        car_to_desc = self.player.get_location().y - self.desc_p.y - self.db[1]
        ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
        ped_time = ped_to_desc / self.ped_speed
        speed = car_to_desc / ped_time
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                breaking_point,
                speed=speed if car_behave else speed - 1,
                yielding=car_behave,
            )

    def setup_01_int(self, obstacles, conf):
        # print("Setup 01_int")
        spawning_distance = conf.spawning_distance
        walking_distance = conf.walking_distance
        looking_distance = conf.looking_distance
        crossing_distance = conf.crossing_distance
        reenter_distance = conf.reenter_distance
        op_reenter_distance = conf.op_reenter_distance
        street_delta = 3 if conf.char == "yielding" else 5
        self.db = [-1, 15] if conf.char == "yielding" else [-1, 20]  # TODO
        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1

        # print(spawning_distance)
        # print(carla.Location(0,-spawning_distance,0))
        # print(obstacles[0][1].location )
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc  # + carla.Location(-5,0,0)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        print(self.walker)
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        offsets_1 = [(0, walking_distance), (street_delta, walking_distance + crossing_distance)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        offsets_2 = [
            (9.5, walking_distance + crossing_distance + op_reenter_distance),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 2),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 20),
        ]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=0, g=255, b=0) if self.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed * mult,
        )

        reenter = walking_distance + crossing_distance + reenter_distance
        offsets_3 = [(0, reenter), (0, reenter + 5)]
        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.path_controller_3 = PathController(self.world, self.walker, path_3, self.ped_speed)

        turn_p = self.get_point((0, looking_distance * walking_distance))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)

        self.look_behind_right = LookBehindRight(self.walker, path_1[0], conf.char)
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)

        vec = path_1[1] - path_1[0]
        self.desc_p = path_1[0] + 0.95 * vec
        # self.db = [2,10]
        if conf.char == "forcing":
            self.path_controller_1.speed_schedule = [
                (path_1[0] + per * vec, 1.1) for per in [0.0, 0.2, 0.4]
            ]

        # self._draw_db_circle()
        if self.debug:
            self.world.debug.draw_point(
                path_1[0] + 0.2 * vec,
                size=0.1,
                color=carla.Color(r=0, g=255, b=255),
                life_time=0,
            )
            self._draw_grid()
            self._draw_db()
        self.relaxer = Relaxer(self.walker, self.player, path_1[0] + 0.2 * vec)

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            self.path_2[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        player_loc = self.player.get_location()
        breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
        car_behave = conf.char == "forcing"
        car_to_desc = self.player.get_location().y - self.desc_p.y - self.db[1]
        ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
        ped_time = ped_to_desc / self.ped_speed
        speed = car_to_desc / ped_time
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                breaking_point,
                speed=speed - 1 if car_behave else speed,
                yielding=car_behave,
            )

    def setup_04_int(self, obstacles, conf):
        spawning_distance = conf.spawning_distance
        walking_distance = conf.walking_distance
        looking_distance1 = conf.looking_distance1
        looking_distance2 = conf.looking_distance2
        crossing_distanceX = conf.crossing_distanceX
        crossing_distanceY = conf.crossing_distanceY
        walk_back_distance = conf.walk_back_distance
        char = conf.char

        street_delta = 3 if conf.char == "yielding" else 5
        if self.dummy_car:
            self.db = [-1, 20] if conf.char == "yielding" else [-1, 20]
        else:
            self.db = [-1, 15] if conf.char == "yielding" else [-1, 20]

        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc  # + carla.Location(-5,0,0)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        # The first path: To the edge of the curb
        offsets_1 = [(0, walking_distance), (1, walking_distance + crossing_distanceY)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=100, b=0) if self.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # The second path: Crossing the street to red marker
        offsets_2 = [(crossing_distanceX, walking_distance + crossing_distanceY)]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed * mult,
        )

        # The third path: fully crossing the street to the other side
        offsets_3 = [
            (12, walking_distance + crossing_distanceY),
            (12, walking_distance + crossing_distanceY + 5),
        ]
        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.path_controller_3 = PathController(self.world, self.walker, path_3, self.ped_speed)

        # The fourth path: Taking a step back
        offsets_4 = [
            (
                crossing_distanceX - walk_back_distance - 0.2 * crossing_distanceX,
                walking_distance + crossing_distanceY,
            ),
        ]
        path_4 = self._compute_plans(
            offsets_4,
            base_loc,
            color=carla.Color(r=255, g=100, b=0) if self.debug else None,
        )
        self.path_controller_4 = PathController(self.world, self.walker, path_4, self.ped_speed)

        # turn_p = self.get_point((0, looking_distance1 + walking_distance))
        self.turn_head = TurnHeadRightBehindNoICR(self.walker, path_1[1])

        # self.look_behind_right = LookBehindRight(self.walker, path_1[1], conf.char)
        self.turning_point = self.get_p_from_vector(path_1[0], path_1[1], 0.1)
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)

        second_turn_p = self.get_point(
            (crossing_distanceX - looking_distance2, walking_distance + crossing_distanceY),
        )

        self.turn_head_second = TurnHeadRightBehindNoICR(self.walker, second_turn_p)

        reset_ld1_p = self.get_point(
            (1 + looking_distance1, walking_distance + crossing_distanceY),
        )
        self.resetLD1 = ResetPose(self.walker, reset_ld1_p)
        self.resetLD2 = ResetPose(self.walker, self.path_2[0])

        vec = path_1[1] - path_1[0]
        self.desc_p = self.path_2[0]
        # self.db = [2,10]
        # if conf.char == "forcing":
        #     self.path_controller_1.speed_schedule = [(path_1[0] + per * vec, 1.1) for per in [0.0, 0.2, 0.4]]

        # self._draw_db_circle()
        if self.debug:
            self.world.debug.draw_point(
                reset_ld1_p,
                color=carla.Color(r=0, g=255, b=255),
                life_time=0,
            )
            self.world.debug.draw_point(
                second_turn_p,
                color=carla.Color(r=0, g=255, b=255),
                life_time=0,
            )
            # self.world.debug.draw_point(path_1[0] + 0.2 * vec, size=0.1, color=carla.Color(r=0, g=255, b=255),
            #                             life_time=0)
            self._draw_grid()
            self._draw_db()
        self.relaxer = Relaxer(self.walker, self.player, self.path_2[0])

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_3[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )
        self.starts_crossing = InternalStateSetter(
            self.walker,
            path_1[-1],
            icr=ICR.GOING_TO,
            son=self.walker.initial_son,
        )
        self.lean_forward = LeanForward(self.walker, path_1[-1])
        self.curd_stat = InternalStateSetter(
            self.walker,
            path_1[0],
            icr=ICR.PLANNING_TO,
            son=self.walker.initial_son,
        )

        player_loc = self.player.get_location()
        breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
        car_behave = conf.char == "forcing"
        car_to_desc = self.player.get_location().y - self.desc_p.y - self.db[1]
        ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
        ped_time = ped_to_desc / self.ped_speed
        speed = car_to_desc / ped_time
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                breaking_point,
                speed=speed - 2 if car_behave else speed - 1,
                yielding=car_behave,
            )

    ## ACTUAL EDIT
    def setup_05_int(self, obstacles, conf):
        spawning_distance = conf.spawning_distance
        walking_distance_X = conf.walking_distance_X
        walking_distance_Y = conf.walking_distance_Y
        uncertain_steps = conf.uncertain_steps

        self.walker.icr = ICR.INTERESTED
        self.walker.son = SON.FORCING if conf.char == "forcing" else SON.YIELDING

        crossing_distance = conf.crossing_distance
        if self.dummy_car:
            self.db = [-1, 30] if conf.char == "yielding" else [-1, 20]
        else:
            self.db = [-1, 15] if conf.char == "yielding" else [-1, 20]

        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc  # + carla.Location(-5,0,0)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        # Walk to the curb
        offsets_1 = [(walking_distance_X, -walking_distance_Y)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=100, b=0) if self.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # Walk to the middle of the road
        offsets_2 = [(walking_distance_X + crossing_distance, -walking_distance_Y)]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed * mult,
        )

        delta = crossing_distance / (uncertain_steps + 1)
        points = []

        for uncert_p in range(uncertain_steps):
            points.append(
                self.get_point((walking_distance_X + (uncert_p + 1) * delta, -walking_distance_Y)),
            )

        # for x in points:
        #     self.world.debug.draw_point(x, size=0.1, color=carla.Color(r=255, g=255, b=255),
        #                                 life_time=0)

        # Uncertain generator
        self.uncertain = UncertainSteps(self.walker, points, conf.char)

        # reenter = walking_distance + crossing_distance + reenter_distance
        # Continue Crossing Road
        offsets_3 = [(12, -walking_distance_Y), (12, 0)]
        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.path_controller_3 = PathController(self.world, self.walker, path_3, self.ped_speed)

        turn_p = self.get_point((0, walking_distance_Y))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)

        self.look_behind_right = LookBehindRight(self.walker, path_1[0], conf.char)
        self.turning_point = self.get_p_from_vector(path_1[0], path_1[0], 0.1)
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)

        # vec = path_1[0] - path_1[0]
        self.desc_p = self.path_2[0]
        # self.db = [2,10]

        ped_speed = conf.ped_speed
        if conf.char == "forcing":
            for p in range(len(points)):
                self.path_controller_2.speed_schedule = [
                    (points[p], conf.ped_speed * 1.5 if p % 2 == 0 else conf.ped_speed * 1.0),
                ]
        else:
            for p in range(len(points)):
                self.path_controller_2.speed_schedule = [
                    (points[p], conf.ped_speed * 1.0 if p % 2 == 0 else conf.ped_speed * 0.5),
                ]

        # self._draw_db_circle()
        if self.debug:
            # self.world.debug.draw_point(path_1[0] + 0.2 * vec, size=0.1, color=carla.Color(r=0, g=255, b=255),
            #                             life_time=0)
            self._draw_grid()
            self._draw_db()
        self.relaxer = Relaxer(self.walker, self.player, path_1[0])

        self.lean_forward = LeanForward(self.walker, self.desc_p)

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_3[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        player_loc = self.player.get_location()
        breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
        car_behave = conf.char == "forcing"
        car_to_desc = self.player.get_location().y - self.desc_p.y - self.db[1]
        ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
        ped_time = ped_to_desc / self.ped_speed
        speed = car_to_desc / ped_time
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                breaking_point,
                speed=speed - 1 if car_behave else speed - 0.5,
                yielding=car_behave,
            )

    def setup_06_int(self, obstacles, conf):
        # print(conf.char)

        self.walker.icr = ICR.GOING_TO
        self.walker.son = SON.FORCING if conf.char == "forcing" else SON.YIELDING

        spawning_distance = conf.spawning_distance
        crossing_distance = conf.crossing_distance
        car_avoid_X = conf.car_avoid_X
        car_avoid_Y = conf.car_avoid_Y

        street_delta = 3 if conf.char == "yielding" else 5

        if self.dummy_car:
            self.db = [-1, 5 + car_avoid_Y] if conf.char == "yielding" else [-1, 10]
        else:
            self.db = [-1, car_avoid_Y + 2] if conf.char == "yielding" else [-1, 5]

        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1

        # print(spawning_distance)
        # print(carla.Location(0,-spawning_distance,0))
        # print(obstacles[0][1].location )
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc  # + carla.Location(-5,0,0)
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        # Walk to the middel of the road
        offsets_1 = [(crossing_distance, 0)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # Walk to the other side of the road
        offsets_2 = [(12, 0), (12, 20)]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=0, g=255, b=0) if self.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed * mult,
        )

        # Avoid the car
        offsets_3 = [
            (crossing_distance + car_avoid_X, -car_avoid_Y),
            (12, -car_avoid_Y),
            (12, -car_avoid_Y + 20),
        ]
        self.path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.path_controller_3 = PathController(
            self.world,
            self.walker,
            self.path_3,
            self.ped_speed,
        )

        turn_p = self.get_point((0, 0))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)

        self.look_behind_right = LookBehindRight(self.walker, path_1[0], conf.char)
        self.turning_point = self.get_p_from_vector(path_1[0], path_1[0], 0.1)
        self.raise_arm = RaiseArm(
            self.walker,
            path_1[0],
            "forcing",
            self.get_p_from_vector(path_1[0], self.path_2[0], 0.5),
        )
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)

        vec = path_1[0] - path_1[0]
        self.desc_p = path_1[0] + 0.95 * vec
        # self.db = [2,10]
        if conf.char == "forcing":
            self.path_controller_2.speed_schedule = [
                (path_1[0] + per * path_1[0] - carla.Location(1, 0, 0), 10.9)
                for per in [0.0, 0.2, 0.4]
            ]
        if conf.char == ("yielding"):
            self.path_controller_1.speed_schedule = [
                (path_1[0] - per * path_1[0] - carla.Location(1, 0, 0), 0.8)
                for per in [0.0, 0.2, 0.4]
            ]
        # self._draw_db_circle()
        if self.debug:
            # self.world.debug.draw_point(path_1[0] + 0.2 * vec, size=0.1, color=carla.Color(r=0, g=255, b=255),
            #                             life_time=0)
            self._draw_grid()
            self._draw_db()
        self.relaxer = Relaxer(self.walker, self.player, path_1[0] + 0.2 * vec)

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            self.path_3[1],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )
        self.iss_crossed_2 = InternalStateSetter(
            self.walker,
            self.path_2[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        player_loc = self.player.get_location()
        breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
        car_behave = conf.char == "forcing"
        car_to_desc = self.player.get_location().y - self.desc_p.y - self.db[1]
        ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
        ped_time = ped_to_desc / self.ped_speed
        speed = car_to_desc / ped_time
        if self.dummy_car:
            self.car_controller = CarController(
                self.player,
                breaking_point,
                speed=speed - 1 if car_behave else speed - 0.5,
                yielding=car_behave,
            )

    def setup_07_int(self, obstacles, conf):
        """Setup for Interactive Scenario 7: Based on 05_int + Hand Raise"""
        logger.info("Setting up Scenario 07_int: Based on 05_int + Hand Raise")
        # --- Parameters ---
        spawning_distance = conf.spawning_distance
        # Use parameter names consistent with IConfig05/IConfig07
        walking_distance_X = conf.walking_distance_X
        walking_distance_Y = conf.walking_distance_Y
        crossing_distance = conf.crossing_distance  # Reusing this name for simplicity
        self.sprint_multiplier = conf.sprint_speed_multiplier
        self.wait_duration = conf.wait_duration
        self.char = conf.char
        self.ped_speed = conf.ped_speed

        self.db = (
            [-1, 15] if self.char == "yielding" else [-1, 20]
        )  # Decision Box from 05_int (adjust if needed)
        mult = 1.0 if self.char == "yielding" else 1.1 * 1.1 * 1.1  # Speed multiplier from 05_int

        # --- Spawn & Base Location (Identical to 05_int) ---
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc  # Spawn at base_loc
        spawn_rotation = obstacles[0][
            1
        ].rotation  # Use rotation from scenario def (usually 90.0 for 05_int)
        # Ensure correct starting Yaw if scenario def doesn't provide it reliably
        # spawn_rotation.yaw = 90.0 # Explicitly face East like 05_int start

        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, spawn_rotation),
        )
        if self.walker is None:
            logger.error("Failed to spawn walker for scenario 07_int")
            return
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        # --- Path 1: Walk Towards Curb (Identical to 05_int's first path) ---
        offsets_1 = [(walking_distance_X, -walking_distance_Y)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=100, b=0) if self.debug else None,
        )
        if not path_1:
            logger.error("Scenario 07_int: Path 1 calculation failed.")
            return
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        self.curb_point = path_1[-1]  # Point where pedestrian reaches near curb

        # --- Path 2: Cross Street (Sprint or Normal) ---
        # Crosses by crossing_distance in +X direction from curb_point
        target_across_loc = self.curb_point + carla.Location(int(crossing_distance), 0, 0)
        offsets_2 = [(target_across_loc.x - base_loc.x, -(target_across_loc.y - base_loc.y))]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.debug else None,
        )
        # Initial speed is base speed * mult (from 05_int logic), might be overridden by sprint
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed * mult,
        )

        # --- Path 3: Optional Continue (Similar to 05_int) ---
        offsets_3 = [
            (walking_distance_X + crossing_distance + 5, -walking_distance_Y),  # Further +X
            (walking_distance_X + crossing_distance + 5, -walking_distance_Y - 5),
        ]  # Walk along other side
        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.path_controller_3 = PathController(
            self.world, self.walker, path_3, self.ped_speed
        )  # Optional

        # --- Initialize Controllers & State ---
        self.look_across_street_left = LookAcrossStreetLeft(
            self.walker, self.curb_point, duration=0.5
        )  # Look triggered at curb
        self.raise_hand_briefly = RaiseHandBriefly(
            self.walker, self.curb_point, raise_duration=0.2, hold_duration=0.4
        )  # Hand raise triggered after look
        self.reset_pose = ResetPose(self.walker)  # Generic reset

        self.at_curb = False
        self.look_started = False
        self.look_finished = False
        self.hand_raise_started = False
        self.hand_raise_finished = False
        self.wait_start_time = None
        self.decided_action = None  # "Wait", "Sprint", "CrossNormal", "CrossingDone"

        # --- Debugging & Initial State ---
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(0, 255, 255))
            self._draw_point(self.curb_point, color=carla.Color(255, 255, 0))
            if self.path_2:
                self._draw_point(self.path_2[0], color=carla.Color(255, 0, 255))
            self.desc_p = self.curb_point  # Decision point is at the curb
            self._draw_db(self.db, color=carla.Color(0, 0, 255))

        self.walker.icr = ICR.INTERESTED
        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.walker.son = self.walker.initial_son

        # --- Setup Dummy Car (Similar to 05_int/previous 07_int) ---
        if self.dummy_car:
            player_loc = self.player.get_location()
            car_yielding = self.char == "yielding"
            estimated_car_speed_mps = 8.0  # Example speed
            self.car_controller = CarController(
                self.player,
                braking_point=carla.Location(player_loc.x, self.curb_point.y + self.db[0] - 5, 0.5)
                if car_yielding
                else None,
                speed=estimated_car_speed_mps,
                yielding=car_yielding,
            )
            logger.debug(
                f"Car controller initialized with speed: {estimated_car_speed_mps:.1f} m/s"
            )

    def _compute_plans(self, offsets, position, color=None):
        plan = []
        cur = position  # self.walker.get_location()
        for offset_x, offset_y in offsets:
            loc = carla.Location(cur.x + offset_x, cur.y - offset_y, 0.5)
            plan.append(loc)
            if color is not None:
                self.world.debug.draw_point(loc, size=0.1, color=color, life_time=0)
        return plan

    def _draw_circle(self, loc, radius):
        for i in range(0, 360, 2):
            x = radius * math.cos(math.radians(i))
            y = radius * math.sin(math.radians(i))

            self.world.debug.draw_point(
                loc + carla.Location(-x, y, 0),
                size=0.05,
                color=carla.Color(255, 165, 0),
                life_time=0,
            )

    def _draw_db(self, db=None, color=carla.Color(0, 255, 0)):
        if db is None:
            db = self.db
        left = carla.Location(83, self.desc_p.y + db[0], 0.5)
        right = carla.Location(103, self.desc_p.y + db[0], 0.5)
        self.world.debug.draw_line(left, right, thickness=0.05, color=color)
        left = carla.Location(83, self.desc_p.y + db[1], 0.5)
        right = carla.Location(103, self.desc_p.y + db[1], 0.5)
        self.world.debug.draw_line(left, right, thickness=0.05, color=color)

    def compute_collision_point(self):
        walker_loc = self.walker.get_location()
        goal_loc = self.path_2[0]
        walker_dir = goal_loc - walker_loc
        car_loc = self.player.get_location()
        walker_vel = self.walker.get_velocity()
        walker_vel = walker_dir * l2_length(walker_vel) / l2_length(walker_dir)
        car_vel = self.player.get_velocity()

        self.world.debug.draw_line(
            walker_loc,
            walker_loc + 2 * walker_vel,
            thickness=0.05,
            color=carla.Color(255, 255, 255),
        )
        self.world.debug.draw_line(
            car_loc,
            car_loc + 2 * car_vel,
            thickness=0.05,
            color=carla.Color(255, 255, 255),
        )

    def _draw_db_circle(self):
        self._draw_circle(self.desc_p, self.db[0])
        self._draw_circle(self.desc_p, self.db[1])

    def setup_01_int_vanilla(self, spawn_point, obstacles):
        self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        self._draw_grid()
        offsets_1 = [(0,), (5, 10)]
        path_1 = self._compute_plans(offsets=offsets_1, color=carla.Color(r=255, g=0, b=0))
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        offsets_2 = [(15, 12)]
        self.path_2 = self._compute_plans(offsets=offsets_2, color=carla.Color(r=0, g=255, b=0))
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed,
        )

        offsets_3 = [(1.5, 15), (0, 25)]
        path_3 = self._compute_plans(offsets=offsets_3, color=carla.Color(r=0, g=0, b=255))
        self.path_controller_3 = PathController(self.world, self.walker, path_3, self.ped_speed)

        self.look_behind_right = LookBehindRight(self.walker, path_1[0])
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)
        turn_p = self.get_point((0, 4))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)
        vec = path_1[1] - path_1[0]
        self.desc_p = path_1[0] + 0.85 * vec
        self.db = [3, 15]
        self._draw_db()
        self.world.debug.draw_point(
            self.desc_p,
            size=0.1,
            color=carla.Color(r=0, g=255, b=255),
            life_time=0,
        )
        self.world.debug.draw_point(
            turn_p,
            size=0.1,
            color=carla.Color(r=0, g=255, b=255),
            life_time=0,
        )
        self.choice = None

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.world.set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification("LayerMap selected: %s" % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification("Unloading map layer: %s" % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification("Loading map layer: %s" % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

    def render(self, display):
        self.camera_manager.render(display)
        # self.semseg_sensor.render(display)
        # self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

        self.semseg_sensor.sensor.destroy()
        self.semseg_sensor.sensor = None
        self.semseg_sensor.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor if self.camera else None,
            self.semseg_sensor.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.walker is not None:
            self.walker.destroy()
        # if self.incoming_car is not None and self.scenario[0] in [10, 3, 7, 8]:
        if self.incoming_car is not None and self.incoming_car.is_alive:
            self.incoming_car.destroy()
        if self.scenario[0] in [11, 12]:
            if self.parked_cars is not None:
                for car in self.parked_cars:
                    car.destroy()

        # Destroy POV camera if it exists
        if hasattr(self, "pov_camera") and self.pov_camera is not None:
            self.pov_camera.destroy()
