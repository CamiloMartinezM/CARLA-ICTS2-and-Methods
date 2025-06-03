# ruff: noqa: D102
"""Module containing the base scenario class and specific scenario implementations."""

import abc
from typing import TYPE_CHECKING, Any

import carla
import numpy as np

from carla_icts2.benchmark.environment.car_controller import CarController

# Import necessary controller classes here
from carla_icts2.benchmark.environment.ped_controller import (
    ICR,
    SON,
    ControllerConfig,
    InternalStateSetter,
    LeanForward,
    LookBehindLeft,
    LookBehindLeftSpine,
    LookBehindRight,
    PathController,
    RaiseArm,
    Relaxer,
    ResetPose,
    TurnHeadLeftWalk,
    TurnHeadRightBehind,
    TurnHeadRightBehindNoICR,
    TurnHeadRightWalk,
    UncertainSteps,
    l2_distance,
    y_distance,
)  # Local import
from carla_icts2.benchmark.environment.utils import (
    find_weather_presets,
)  # Keep utils if needed by scenarios
from carla_icts2.config import logger

if TYPE_CHECKING:
    from carla_icts2.benchmark.environment.world import World  # Avoid circular import


class BaseScenario(abc.ABC):
    """Abstract Base Class for all scenarios."""

    def __init__(self, world: "World", config: ControllerConfig):
        """Initialize the scenario.

        Args:
            world: The main World object containing actors and environment access.
            config: The specific ControllerConfig for this scenario instance.
        """
        self.my_world = world
        self.world = self.my_world.world  # Carla World instance
        self.config = config
        self.choice = None

    @abc.abstractmethod
    def get_scenario_id(self) -> str:
        """Return the unique string identifier for the scenario (e.g., '01_int')."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_spawn_details(
        self,
    ) -> tuple[Any, Any, tuple[float, float, float], tuple[float, float, float]]:
        """Return the details needed for spawning.

        Returns
        -------
        tuple: A tuple containing:
            - scenario_id (str): Unique identifier for the scenario.
            - obstacles (list): List of tuples containing actor blueprints and spawn transforms.
            - end_coords (tuple[float, float, float]): Coords for end location `(x, y, yaw)`.
            - start_coords (tuple[float, float, float]): Coords for start location `(x, y, yaw)`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def setup(self) -> None:
        """Set up the specific actors, paths, and controllers for this scenario.

        This method MUST spawn the primary walker and assign it to `self.world.walker`.
        It should also spawn other relevant actors like incoming/parked cars.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tick(self) -> None:
        """Execute one simulation step's logic specific to this scenario.

        This involves calling `step()` on the relevant controllers.
        """
        raise NotImplementedError

    def nullify_references(self) -> None:
        """Nullify references to actors to avoid memory leaks."""
        self.walker = None

    def get_obstacle_blueprints(self) -> list:
        """Get the list of obstacle blueprints and transforms."""
        _id, obstacles, _end, _start = self.get_spawn_details()
        return obstacles  # List of [blueprint, transform]

    def get_start_transform(self) -> carla.Transform:
        """Get the starting transform for the ego vehicle."""
        _id, _obstacles, _end, start_coords = self.get_spawn_details()
        spawn_point = carla.Transform()
        spawn_point.location.x = start_coords[0]
        spawn_point.location.y = start_coords[1]
        spawn_point.location.z = 0.01
        spawn_point.rotation.yaw = start_coords[2]
        return spawn_point

    def get_end_location(self) -> tuple[float, float, float]:
        """Get the target end location coordinates for the ego vehicle."""
        _id, _obstacles, end_coords, _start = self.get_spawn_details()
        return end_coords  # Should be (x, y, yaw) tuple

    # TODO: Check from original
    def _compute_plans(self, offsets, base_location, color=None):
        """Helper function copied from World class to compute paths."""
        plan = []
        cur = base_location
        for offset_x, offset_y in offsets:
            loc = carla.Location(cur.x + offset_x, cur.y - offset_y, 0.5)
            plan.append(loc)
            if color is not None and self.world.debug:
                self.world.debug.draw_point(loc, size=0.1, color=color, life_time=0)
        return plan

    def _get_p_from_vector(self, loc1, loc2, perc):
        """Helper function copied from World class."""
        vec = loc2 - loc1
        return loc1 + perc * vec

    def _draw_point(self, p, color=carla.Color(r=0, g=255, b=255)):
        """Helper debug draw function."""
        if self.world.debug:
            self.world.debug.draw_point(p, size=0.1, color=color, life_time=0)

    def _draw_db(self, db=None, color=carla.Color(0, 255, 0)):
        """Helper debug draw function."""
        if not self.world.debug or not hasattr(self, "desc_p") or self.desc_p is None:
            return

        if db is None:
            if self.db is None:
                return
            db = self.db

        left = carla.Location(83, self.desc_p.y + db[0], 0.5)  # Assuming desc_p is defined
        right = carla.Location(103, self.desc_p.y + db[0], 0.5)
        self.world.debug.draw_line(left, right, thickness=0.05, color=color, life_time=0)
        left = carla.Location(83, self.desc_p.y + db[1], 0.5)
        right = carla.Location(103, self.desc_p.y + db[1], 0.5)
        self.world.debug.draw_line(left, right, thickness=0.05, color=color, life_time=0)

    def set_walker_speed_relative(self, per):
        """Helper function copied from World."""
        if self.walker:
            control = self.walker.get_control()
            control.speed = per * control.speed
            self.walker.apply_control(control)

    def decision_trigger(self, distance, db, without_speed=False) -> bool:
        """Helper function copied from World."""
        if self.my_world.random:
            choice = np.random.choice(2)
            # choices = [np.random.choice(2) for i in range(10)] # This wasn't used
            return choice == 1

        if self.my_world.player is None:
            logger.error("Decision trigger called with player=None")
            return False

        velocity = self.my_world.player.get_velocity()
        speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5

        # Ensure db is not None before comparison
        if db is None:
            logger.warning("Decision trigger called with db=None")
            return False

        return distance >= db[0] and distance <= db[1] and (speed > 1.5 or without_speed)

    def second_decider(self, distance, dec_d=None):
        """Helper function copied from World."""
        # Note: This relies on state (self.second_choice, self.waiting_c, self.waiting_time)
        # which needs to be initialized in the specific scenario's __init__ or setup if used.
        # Add default initializations here or ensure subclasses handle it.
        if not hasattr(self, "second_choice"):
            self.second_choice = None
        if not hasattr(self, "waiting_c"):
            self.waiting_c = 0
        if not hasattr(self, "waiting_time"):
            self.waiting_time = 0

        if self.my_world.random:
            if self.second_choice:
                simulation_step = 0.05
                self.waiting_c += 1
                return self.waiting_c * simulation_step > self.waiting_time
            self.waiting_time = np.random.random() * 2 + 1
            self.waiting_c = 0
            self.second_choice = True
            return False  # Return False on the first call in random mode
        else:
            velocity = self.my_world.player.get_velocity()
            speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
            if dec_d is None:
                return distance < 0 or speed < 1  # less than 3.6kmh
            return distance + dec_d < 0 or (speed < 1 and distance > 2.5)

    def destroy(self) -> None:
        """Clean up actors specific to this scenario."""
        actors_to_destroy = []
        if hasattr(self, "walker") and self.walker and self.walker.is_alive:
            actors_to_destroy.append(self.walker)

        if hasattr(self, "incoming_car") and self.incoming_car and self.incoming_car.is_alive:
            actors_to_destroy.append(self.incoming_car)

        if hasattr(self, "parked_cars") and self.parked_cars:
            actors_to_destroy.extend([car for car in self.parked_cars if car and car.is_alive])

        # Use client batch destruction
        if actors_to_destroy:
            for actor in actors_to_destroy:
                if actor.is_alive:
                    actor.destroy()

        # Nullify references to avoid memory leaks
        self.nullify_references()


class Scenario01Int(BaseScenario):
    """Implementation for Interactive Scenario 01."""

    # def __init__(self, world: "World", config: ControllerConfig) -> None:
    #     super().__init__(world, config)

    def get_scenario_id(self) -> str:
        return "01_int"

    def get_spawn_details(
        self,
    ) -> tuple[Any, Any, tuple[float, float, float], tuple[float, float, float]]:
        start = (92.5, 300, -90)
        end = (92.5, 200, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 85
        walker_spawn_point.location.y = 300
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [walker_bp, walker_spawn_point]
        obstacles.append(walker)

        return "01_int", obstacles, end, start

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        # Override default if necessary, based on original world.py logic
        return ICR.LOW, SON.AVERTING  # Example, adjust as needed

    def setup(self) -> None:
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")
        obstacles = self.get_obstacle_blueprints()
        conf = self.config  # Use the passed config

        # --- Start of logic copied from World.setup_01_int ---
        spawning_distance = conf.spawning_distance
        walking_distance = conf.walking_distance
        looking_distance = conf.looking_distance
        crossing_distance = conf.crossing_distance
        reenter_distance = conf.reenter_distance
        op_reenter_distance = conf.op_reenter_distance
        self.ped_speed = conf.ped_speed

        street_delta = 3 if conf.char == "yielding" else 5
        self.db = [-1, 15] if conf.char == "yielding" else [-1, 20]
        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1

        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc

        # Spawn walker and assign to self.world.walker
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        if not self.walker:
            logger.error(f"Failed to spawn walker for {self.get_scenario_id()}")
            # Handle error appropriately, maybe raise exception or set a flag
            return

        self.my_world.walker = self.walker  # IMPORTANT: Assign to world instance

        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()  # Allow walker to settle?

        self.walker.on_street = False  # Initialize walker state

        offsets_1 = [(0, walking_distance), (street_delta, walking_distance + crossing_distance)]
        path_1 = self._compute_plans(offsets_1, base_loc)
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        offsets_2 = [
            (9.5, walking_distance + crossing_distance + op_reenter_distance),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 2),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 20),
        ]
        path_2 = self._compute_plans(offsets_2, base_loc)
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            path_2,
            self.ped_speed * mult,
        )

        reenter = walking_distance + crossing_distance + reenter_distance
        offsets_3 = [(0, reenter), (0, reenter + 5)]
        path_3 = self._compute_plans(offsets_3, base_loc)
        self.path_controller_3 = PathController(
            self.world,
            self.walker,
            path_3,
            self.ped_speed,
        )

        # Note: get_point needs fixing if it relies on self.walker which might not be set when get_point is called
        # Let's assume base_loc can be used for reference before walker spawns if needed
        # Or refactor get_point to take the reference location
        # For now, assuming setup order is okay.
        turn_p = base_loc + carla.Location(
            0, -(looking_distance * walking_distance), 0.5
        )  # Approximate get_point
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)

        self.look_behind_right = LookBehindRight(self.walker, path_1[0], conf.char)
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)

        vec = path_1[1] - path_1[0]
        self.desc_p = path_1[0] + 0.95 * vec
        if conf.char == "forcing":
            self.path_controller_1.speed_schedule = [
                (path_1[0] + per * vec, 1.1) for per in [0.0, 0.2, 0.4]
            ]

        self._draw_db()  # Debug drawing

        relaxer_start_point = path_1[0] + 0.2 * vec
        self.relaxer = Relaxer(self.walker, self.my_world.player, relaxer_start_point)

        self.walker.initial_son = SON.YIELDING if conf.char == "yielding" else SON.FORCING
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_2[0],  # Use the actual path variable
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        if self.my_world.dummy_car:
            player_loc = self.my_world.player.get_location()
            breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
            car_behave = conf.char == "forcing"
            car_to_desc = self.my_world.player.get_location().y - self.desc_p.y - self.db[1]
            # Need walker location *after* potential tick
            ped_to_desc = carla.Location.distance(self.walker.get_location(), self.desc_p)
            ped_time = ped_to_desc / self.ped_speed if self.ped_speed > 0 else float("inf")
            speed = car_to_desc / ped_time if ped_time != float("inf") else 8  # Default speed
            # Store car controller on world or scenario? Let's try world for now
            self.my_world.car_controller = CarController(
                self.my_world.player,
                breaking_point,
                speed=max(3, speed - 1 if car_behave else speed),  # Ensure min speed
                yielding=car_behave,
            )
        # --- End of copied logic ---

    def tick(self) -> None:
        # --- Start of logic copied from World.tick ---
        if not self.walker or not self.path_controller_1:  # Check if setup was successful
            return

        status = self.path_controller_1.step()
        self.look_behind_right.step()
        self.turn_head.step()

        # Access car_controller from my_world
        if self.my_world.dummy_car and self.my_world.car_controller:
            self.my_world.car_controller.step()

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
            distance = (
                y_distance(self.walker.get_location(), self.my_world.player.get_location()) - 2
            )
            if self.decision_trigger(distance, self.db):
                self.choice = "Left"
                self.walker.icr = ICR.VERY_LOW
                self.walker.son = SON.AVERTING
            else:
                self.choice = "Right"
                self.walker.icr = ICR.GOING_TO
                # self.walker.son = SON.YIELDING # Original code didn't set SON here
        self.relaxer.step()
        self.iss_crossed.step()
        # --- End of copied logic ---


# Create a mapping for easy lookup
SCENARIO_MAP = {
    "01_int": Scenario01Int,
    # "01_non_int": Scenario01NonInt,
    # "02_int": Scenario02Int,
    # "03_int": Scenario03Int,
    # "02_non_int": Scenario02NonInt,
    # "03_non_int": Scenario03NonInt,
    # "04_int": Scenario04Int,
    # "05_int": Scenario05Int,
    # "06_int": Scenario06Int,
    # "04_non_int": Scenario04NonInt,
    # "05_non_int": Scenario05NonInt,
    # "06_non_int": Scenario06NonInt,
    # Add mappings for scenarios 1-12 if they need specific logic different from non-int/int
    # "01": Scenario01, # Needs Scenario01 class definition
}
