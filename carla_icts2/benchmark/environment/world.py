"""Author: Dikshant Gupta
Time: 23.03.21 14:27
"""

import math
import random
import sys
import time
import traceback

import carla
import numpy as np

from carla_icts2.benchmark.environment.car_controller import CarController
from carla_icts2.benchmark.environment.ped_controller import (
    ICR,
    SON,
    ControllerConfig,
    InternalStateSetter,
    LeanForward,
    LeanForwardAndLook,
    LookAcrossStreetLeft,
    LookBehindLeft,
    LookBehindLeftSpine,
    LookBehindRight,
    PathController,
    RaiseArm,
    RaiseArmStationary,
    Relaxer,
    ResetPose,
    SimplifiedWave,
    TurnHeadLeftWalk,
    TurnHeadRightBehind,
    TurnHeadRightBehindNoICR,
    TurnHeadRightWalk,
    UncertainSteps,
    WaveHand,
    l2_distance,
    l2_length,
    y_distance,
)
from carla_icts2.benchmark.environment.sensors import *
from carla_icts2.benchmark.environment.utils import find_weather_presets
from carla_icts2.benchmark.scenarios.scenarios import SCENARIO_MAP, BaseScenario
from carla_icts2.config import logger


class World:
    def __init__(self, carla_world: carla.World, hud, scenario, args):
        self.world = carla_world
        self.actor_role_name = args.rolename

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            logger.error(f"RuntimeError: {error}")
            logger.error("  The server could not send the OpenDRIVE (.xodr) file:")
            logger.error("  Make sure it exists, has the same name of your town, and is correct.")
            sys.exit(1)

        # --- Scenario Management ---
        self.current_scenario: BaseScenario | None = None
        self.current_scenario_id: str | None = None
        self.current_config: ControllerConfig | None = None
        # -------------------------

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
        self.random = False
        self.dummy_car = False
        self.debug = args.debug

        # Initial restart with the provided tuple
        scenario_id, initial_config = scenario
        self.restart(scenario_id, initial_config)

        self.world.on_tick(hud.on_world_tick)

        for _ in range(2):
            self.next_weather()

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
        elif camera_specs == "pedestrian_frontal":
            self.update_camera_func = self.update_pedestrian_frontal_camera
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

        if not self.walker:
            raise ValueError("Found walker to be None here")  # Ensure walker is defined

        if not self.player:
            raise ValueError("Found player to be None here")  # Ensure player is defined

        if follow_walker or not self.already_spawned_bev_camera:
            transform = self.walker.get_transform()

        if follow_player or not self.already_spawned_bev_camera:
            transform = self.player.get_transform()

        if not transform:
            # This shouldn't happen
            return

        location = carla.Location(x=transform.location.x, y=transform.location.y, z=10)
        self.already_spawned_bev_camera = True
        self.world.get_spectator().set_transform(
            carla.Transform(location, carla.Rotation(yaw=180.0, pitch=-90.0)),
        )

    def update_player_pov_camera(self) -> None:
        """Create and attach a POV camera inside the car."""
        # x (+) brings it forward
        # y (-) brings it to the left
        offset = carla.Location(x=1, y=-0.4, z=1.2)

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

    def update_pedestrian_frontal_camera(self) -> None:
        """Set the spectator to a frontal view of the pedestrian."""
        if self.walker is None or not self.walker.is_alive:
            return

        pedestrian_transform = self.walker.get_transform()

        # Use defaults from CAMERA_CONFIGS if available, otherwise hardcode
        # TODO: It's better practice to pass these from run_config.yaml eventually
        distance_in_front = 2.5
        height_offset = 1.6
        camera_yaw_offset = 180.0
        camera_pitch = -10.0

        # Calculate offset in pedestrian's local frame
        # Pedestrian's +X is forward. Camera needs to be at +X relative to ped.
        local_offset = carla.Location(x=distance_in_front, y=0, z=height_offset)

        # Transform this local offset to world coordinates based on pedestrian's transform
        camera_location_world = pedestrian_transform.transform(local_offset)

        # Camera rotation: pedestrian's yaw + 180 degrees to face them, plus pitch
        camera_rotation_world = carla.Rotation(
            pitch=pedestrian_transform.rotation.pitch
            + camera_pitch,  # Add pitch relative to ped's pitch
            yaw=pedestrian_transform.rotation.yaw + camera_yaw_offset,
            roll=pedestrian_transform.rotation.roll,
        )

        final_camera_transform = carla.Transform(camera_location_world, camera_rotation_world)
        self.world.get_spectator().set_transform(final_camera_transform)

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

    # TODO: Compare to original
    def restart(self, scenario_id: str, conf: ControllerConfig):
        logger.info(f"Restarting world for scenario: {scenario_id}")
        self.current_scenario_id = scenario_id
        self.current_config = conf
        self.counter = 0  # Reset tick counter

        # Destroy previous scenario actors before creating new ones
        if self.current_scenario:
            self.current_scenario.destroy()
            self.current_scenario = None
            self.walker = None  # Ensure walker reference is cleared

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        semseg_index = (
            self.semseg_sensor.index
            if self.semseg_sensor is not None
            else 5  # Default semseg index
        )
        cam_pos_index = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 1  # Default POV index
        )
        semseg_pos_index = (
            self.semseg_sensor.transform_index
            if self.semseg_sensor is not None
            else 1  # Default POV index
        )

        # Instantiate the correct scenario class
        ScenarioClass = SCENARIO_MAP.get(scenario_id)
        if not ScenarioClass:
            raise ValueError(f"Scenario ID '{scenario_id}' not found in SCENARIO_MAP.")

        self.current_scenario = ScenarioClass(self, conf)

        if not self.current_scenario:
            raise ValueError(
                f"Scenario ID '{scenario_id}' found in SCENARIO_MAP, but unable to instantiate it"
            )

        # Spawn the player vehicle
        start_transform = self.current_scenario.get_start_transform()
        if self.player is not None and self.player.is_alive:
            self.player.destroy()  # Destroy previous player cleanly
            self.player = None

        # Retry spawning player
        spawn_attempts = 0
        while self.player is None and spawn_attempts < 5:
            self.player = self.world.try_spawn_actor(self.car_blueprint, start_transform)
            if self.player:
                self.modify_vehicle_physics(self.player)
                logger.info(f"Player spawned successfully for scenario {scenario_id}")
            else:
                logger.warning(
                    f"Attempt {spawn_attempts + 1}: Failed to spawn player, retrying...",
                )
                self.world.wait_for_tick()  # Wait a tick before retrying
                spawn_attempts += 1

        if self.player is None:
            logger.error(f"Failed to spawn player after {spawn_attempts} attempts. Exiting.")
            sys.exit(1)

        # Set up the scenario actors and controllers
        # This method is now responsible for spawning the walker
        # and assigning it to self.walker
        try:
            self.current_scenario.setup()
        except Exception as e:
            # Get the full traceback as a string
            tb_str = traceback.format_exc()

            logger.error(
                f"Error during scenario setup for {scenario_id}: {e}\nFull traceback:\n{tb_str}",
                exc_info=True,  # Includes full traceback
                stack_info=True,  # Includes stack info from current point
            )
            self.destroy()  # Clean-up if unable to start
            return  # Prevent further execution if setup failed

        # Check if walker was spawned correctly by setup
        if not self.walker or not self.walker.is_alive:
            logger.error(
                f"Walker was not spawned correctly during setup for scenario {scenario_id}",
            )
            return

        # Set initial walker state (ICR, SON) - Get from scenario object
        initial_icr, initial_son = self.current_scenario.get_initial_walker_state()
        self.walker.icr = initial_icr
        self.walker.son = initial_son
        # self.walker.initial_son = initial_son # Keep if needed elsewhere

        # TODO: Set initial player velocity if required by scenario type (example)
        # This logic could also be moved into the scenario's setup method
        if not self.random and scenario_id.endswith("_int"):
            self.player.set_target_velocity(carla.Vector3D(0, -6, 0))
        elif not self.random and scenario_id.endswith("_non_int"):
            # Adjust velocity based on specific non-int scenarios if needed
            if scenario_id in ["01_non_int", "02_non_int", "03_non_int"]:
                self.player.set_target_velocity(carla.Vector3D(0, -6, 0))  # Example
            elif scenario_id in ["04_non_int", "05_non_int", "06_non_int"]:
                self.player.set_target_velocity(
                    carla.Vector3D(0, -5, 0),
                )  # Example different speed

        # Ensure player exists before setting up sensors
        if self.player is None:
            logger.error("Player is None after spawn attempt, cannot setup sensors.")
            return

        # Destroy old sensors before creating new ones
        self.destroy_sensors(destroy_managers=False)  # Keep managers, just destroy sensor actors

        # Set up the sensors (attached to the possibly new player actor)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)

        # Re-initialize Camera Managers (if they exist) or create them
        # This re-attaches them to the potentially new player actor
        if self.camera:
            if self.camera_manager is None:
                self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
                self.camera_manager.transform_index = cam_pos_index
                self.camera_manager.set_sensor(cam_index, notify=True)
            else:
                # Update parent actor and respawn sensor
                self.camera_manager._parent = self.player
                self.camera_manager.set_sensor(
                    self.camera_manager.index, notify=True, force_respawn=True
                )
            actor_type = get_actor_display_name(self.player)
            self.hud.notification(actor_type)

        if self.semseg_sensor is None:
            self.semseg_sensor = CameraManager(self.player, self.hud, self._gamma)
            self.semseg_sensor.transform_index = semseg_pos_index
            self.semseg_sensor.set_sensor(semseg_index, notify=False)
        else:
            # Update parent actor and respawn sensor
            self.semseg_sensor._parent = self.player
            self.semseg_sensor.set_sensor(
                self.semseg_sensor.index, notify=False, force_respawn=True
            )

        logger.info(f"World restart complete for scenario {scenario_id}")



    # TODO: Compare to original
    def tick(self, clock):
        self.counter += 1
        self.hud.tick(self, clock)
        # dist_walker = abs(self.player.get_location().y - self.walker.get_location().y) # Calculate inside scenario if needed
        # car_velocity = self.player.get_velocity() # Calculate inside scenario if needed
        # car_speed = np.sqrt(car_velocity.x**2 + car_velocity.y**2) # Calculate inside scenario if needed

        # One-time drawing logic (if still required)
        if not self.drawn and self.debug:
            # self._draw_grid() # Example debug drawing
            self.drawn = True

        # Update spectator camera
        if self.update_camera_func is not None:
            if self.update_camera_args is not None:
                self.update_camera_func(**self.update_camera_args)
            else:
                self.update_camera_func()

        # --- Execute scenario-specific logic ---
        if self.current_scenario:
            try:
                self.current_scenario.tick()
            except Exception as e:
                logger.error(
                    f"Error during tick for scenario {self.current_scenario_id}: {e}",
                    exc_info=True,
                )
                # Decide how to handle tick errors (e.g., stop scenario, log and continue?)
                # self.current_scenario = None # Stop processing this scenario
        else:
            # logger.warning("Tick called but no current scenario is active.")
            # Apply default behavior if no scenario is running (e.g., simple forward motion?)
            pass  # Or apply a default control if needed

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
        # --- Parameters ---
        spawning_distance = conf.spawning_distance

        # Use parameter names consistent with IConfig05/IConfig07
        walking_distance_X = conf.walking_distance_X
        walking_distance_Y = conf.walking_distance_Y
        crossing_distance = conf.crossing_distance

        # Distance walked across the street after sprinting (i.e., same direction as the sprint)
        walk_after_crossing_X = conf.walk_after_crossing_X

        # Distance walked perpendicular to the street after crossing (i.e., along the curb)
        walk_after_crossing_Y = conf.walk_after_crossing_Y

        self.sprint_multiplier = conf.sprint_speed_multiplier
        self.wait_duration = conf.wait_duration
        self.char = conf.char
        self.ped_speed = conf.ped_speed

        self.db = [-1, 15] if self.char == "yielding" else [-1, 20]  # Decision Box from 05_int
        mult = 1.0 if self.char == "yielding" else 1.1 * 1.1 * 1.1  # Speed multiplier from 05_int

        # if self.debug:
        logger.debug(
            f"Setup Scenario 07_int: Spawning distance: {spawning_distance:.2f}, "
            f"Crossing distance: {crossing_distance:.2f}, "
            f"Walking distance X: {walking_distance_X:.2f}, "
            f"Walking distance Y: {walking_distance_Y:.2f}, "
            f"Walk after crossing X: {walk_after_crossing_X:.2f}, "
            f"Walk after crossing Y: {walk_after_crossing_Y:.2f}, "
            f"Wait duration: {self.wait_duration:.2f}, "
            f"Character (ped): {self.char}, "
            f"Car yielding: {self.char != 'yielding'}, "
            f"Pedestrian speed: {self.ped_speed:.2f}",
        )

        # --- Spawn & Base Location (Identical to 05_int) ---
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc  # Spawn at base_loc

        # Use rotation from scenario def (usually 90.0 for 05_int)
        # Ensure correct starting Yaw if scenario def doesn't provide it reliably
        # spawn_rotation.yaw = 90.0 # Explicitly face East like 05_int start
        spawn_rotation = obstacles[0][1].rotation

        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, spawn_rotation),
        )

        if self.walker is None:
            logger.error("Failed to spawn walker for scenario 07_int")
            return

        if self.player is None:
            logger.error("There is no player for scenario 07_int")
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
            (
                walking_distance_X + crossing_distance + walk_after_crossing_X,
                -walking_distance_Y,
            ),  # Further +X
            (
                walking_distance_X + crossing_distance + walk_after_crossing_X,
                -walking_distance_Y - walk_after_crossing_Y,
            ),
        ]  # Walk along other side

        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(r=0, g=0, b=255) if self.debug else None,
        )
        self.path_controller_3 = PathController(
            self.world,
            self.walker,
            path_3,
            self.ped_speed,
        )

        # Calculate absolute world location for the end of Path 4
        # Path 4 starts where Path 3 (sprint) ends, which is sprint_target_loc
        # Then it moves walk_after_crossing_X further in X and walk_after_crossing_Y further in -Y
        self.main_ped_final_destination = target_across_loc + carla.Location(
            walk_after_crossing_X,
            walk_after_crossing_Y + 1,
            0,
        )

        # --- SPAWN SECOND PEDESTRIAN ---
        self.walker2 = None  # Initialize
        # Different model
        ped2_blueprint = self.world.get_blueprint_library().filter("walker.pedestrian.0002")[0]

        # Make them non-collidable for simplicity
        if ped2_blueprint.has_attribute("is_invincible"):
            ped2_blueprint.set_attribute("is_invincible", "true")

        # Spawn at the main pedestrian's final destination
        spawn_loc_ped2 = self.main_ped_final_destination

        # Make ped2 face towards the main pedestrian's approach direction (e.g., -X)
        spawn_rotation_ped2 = carla.Rotation(yaw=270.0)  # Facing West (-X)

        self.walker2 = self.world.try_spawn_actor(
            ped2_blueprint,
            carla.Transform(spawn_loc_ped2, spawn_rotation_ped2),
        )
        if self.walker2:
            # Stand still
            self.walker2.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0),
            )
            logger.info(f"Spawned second pedestrian at {spawn_loc_ped2}")
        else:
            logger.error(f"Failed to spawn second pedestrian at {spawn_loc_ped2}")

        # --- Initialize Controllers & State ---
        self.look_across_street_left_controller = LookAcrossStreetLeft(
            self.walker,
            self.curb_point,
            duration=1.5,
        )  # Look triggered at curb

        # Define start and end for RaiseArm
        # Start raising arm slightly after curb_point (where look finishes)
        # End raising arm just before sprint decision or as sprint starts.
        self.raise_arm_start_trigger_point = self.curb_point
        self.raise_arm_end_trigger_point = self.path_2[0]  # Start of sprint path

        # WaveHand controller
        self.wave_hand_controller = SimplifiedWave(
            self.walker,
            debug=self.debug,
        )

        # LeanForward controller
        self.lean_forward_controller = LeanForwardAndLook(
            self.walker,
            start_pos=self.curb_point,  # Triggered at curb after waving
            look_to="custom",
            head_turn_neck_pitch=0,
            head_turn_head_pitch=2,
            head_turn_head_roll=100,
        )

        self.reset_pose_controller = ResetPose(self.walker)  # Still useful after sprint/wait

        # Initialize state flags
        self.at_curb = False
        self.look_started = False
        self.look_finished = False
        self.wave_started = False
        self.wave_total_duration = 2.0  # seconds
        self.wave_finished = False
        self.lean_forward_started = False
        self.lean_forward_finished = False
        self.decision_phase_started = False
        self.wait_start_time = None
        self.decided_action = None
        self.sprint_finished = False
        self.meeting_started = False

        # --- Debugging & Initial State ---
        if self.debug:
            self._draw_point(spawn_loc, color=carla.Color(0, 255, 255))
            self._draw_point(self.curb_point, color=carla.Color(255, 255, 0))
            if self.path_2:
                self._draw_point(self.path_2[0], color=carla.Color(255, 0, 255))
            self.desc_p = self.curb_point  # Decision point is at the curb
            self._draw_db(self.db, color=carla.Color(255, 0, 255))

        self.walker.icr = ICR.INTERESTED
        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.walker.son = self.walker.initial_son

        # InternalStateSetters (Adjust trigger points if necessary)
        self.iss_at_curb = InternalStateSetter(
            self.walker,
            self.curb_point,
            ICR.PLANNING_TO,
            self.walker.son,
        )
        self.iss_at_curb = InternalStateSetter(
            self.walker,
            self.curb_point,
            ICR.PLANNING_TO,
            self.walker.son,
        )
        self.iss_sprinting = InternalStateSetter(
            self.walker,
            target_across_loc,
            ICR.GOING_TO,
            SON.FORCING,
        )
        if self.walker2:
            self.iss_meeting = InternalStateSetter(
                self.walker,
                spawn_loc_ped2,
                ICR.VERY_LOW,
                SON.AVERTING,
            )
        else:
            self.iss_meeting = None  # Important if walker2 fails to spawn

        # --- Setup Dummy Car (Similar to 05_int/previous 07_int) ---
        if self.dummy_car:
            player_loc = self.player.get_location()
            car_yielding = self.char != "yielding"
            estimated_car_speed_mps = 6.0  # Example speed
            self.car_controller = CarController(
                self.player,
                braking_point=carla.Location(
                    player_loc.x,
                    self.curb_point.y + self.db[0] - 11,
                    0.5,
                )
                if car_yielding
                else None,
                speed=estimated_car_speed_mps,
                yielding=car_yielding,
            )
            logger.debug(
                f"Car controller initialized with speed: {estimated_car_speed_mps:.1f} m/s",
            )

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

    def destroy_sensors(self, *, destroy_managers: bool = True) -> None:
        """Destroy sensors and optionally the managers."""
        if self.debug:
            logger.debug(f"Destroying sensors (destroy_managers={destroy_managers})...")

        sensors_to_destroy = [
            (self.camera_manager.sensor if self.camera_manager else None),
            (self.semseg_sensor.sensor if self.semseg_sensor else None),
            (self.collision_sensor.sensor if self.collision_sensor else None),
            (self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None),
            (self.gnss_sensor.sensor if self.gnss_sensor else None),
            (self.imu_sensor.sensor if self.imu_sensor else None),
            (self.radar_sensor.sensor if self.radar_sensor else None),  # Added radar
        ]
        for sensor_actor in sensors_to_destroy:
            if sensor_actor is not None and sensor_actor.is_alive:
                if sensor_actor.is_listening:
                    sensor_actor.stop()
                sensor_actor.destroy()

        # Nullify references even if destruction failed
        if self.camera_manager:
            self.camera_manager.sensor = None
        if self.semseg_sensor:
            self.semseg_sensor.sensor = None
        if self.collision_sensor:
            self.collision_sensor.sensor = None
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.sensor = None
        if self.gnss_sensor:
            self.gnss_sensor.sensor = None
        if self.imu_sensor:
            self.imu_sensor.sensor = None
        if self.radar_sensor:
            self.radar_sensor.sensor = None

        if destroy_managers:
            self.camera_manager = None
            self.semseg_sensor = None
            self.collision_sensor = None
            self.lane_invasion_sensor = None
            self.gnss_sensor = None
            self.imu_sensor = None
            self.radar_sensor = None

    def destroy(self) -> None:
        """Destroy the scenario and all associated actors."""
        if self.radar_sensor is not None:
            self.toggle_radar()

        self.destroy_sensors(destroy_managers=True)

        # Try to clean up spawned actors
        if self.current_scenario:
            self.current_scenario.destroy()
        self.current_scenario = None

        if self.player is not None and self.player.is_alive:
            self.player.destroy()
            self.player = None

        if self.walker is not None and self.walker.is_alive:
            self.walker.destroy()
            self.walker = None

        # Check if walker2 exists and destroy it
        if hasattr(self, "walker2") and self.walker2 is not None and self.walker2.is_alive:
            self.walker2.destroy()
            self.walker2 = None

        if self.incoming_car is not None and self.incoming_car.is_alive:
            self.incoming_car.destroy()

        # Destroy POV camera if it exists
        if hasattr(self, "pov_camera") and self.pov_camera is not None:
            self.pov_camera.destroy()
