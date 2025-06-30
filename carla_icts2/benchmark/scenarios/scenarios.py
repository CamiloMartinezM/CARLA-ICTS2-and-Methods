# ruff: noqa: D102
"""Module containing the base scenario class and specific scenario implementations."""

import abc
import math
import time
from typing import TYPE_CHECKING, Any

import carla

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
    LookBehindRight,
    PathController,
    RaiseArm,
    Relaxer,
    ResetPose,
    SimplifiedWave,
    TurnHeadLeftWalk,
    TurnHeadRightBehind,
    TurnHeadRightBehindNoICR,
    TurnHeadRightWalk,
    UncertainSteps,
    l2_distance,
    l2_length,
    y_distance,
)
from carla_icts2.config import RNG, logger
from carla_icts2.utils.exceptions import (
    PlayerNotAliveError,
    ScenarioSetupError,
    WalkerNotAliveError,
)

if TYPE_CHECKING:
    from carla_icts2.benchmark.environment.world import World  # Avoid circular import


class BaseScenario(abc.ABC):
    """Abstract Base Class for all scenarios."""

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        """Initialize the scenario.

        Args:
            world: The main World object containing actors and environment access.
            config: The specific ControllerConfig for this scenario instance.
        """
        self.my_world = world
        self.world = self.my_world.world  # Carla World instance
        self.config = config

        # Initialize scenario-specific attributes
        self.scenario_id: str = ""  # Unique identifier for the scenario
        self.choice = None
        self.db = None  # Decision box, to be set in subclasses or setup
        self.dummy_car = self.my_world.dummy_car  # Reference to avoid verbosity

        # Walker and player references
        self.walker = None
        self.player = self.my_world.player

    def get_scenario_id(self) -> str:
        """Return the unique string identifier for the scenario (e.g., '01_int')."""
        if not self.scenario_id:
            raise NotImplementedError("scenario_id must be set in the subclass's __init__ method.")

        return self.scenario_id

    @abc.abstractmethod
    def get_spawn_details(
        self,
    ) -> tuple[Any, Any, tuple[float, float, float], tuple[float, float, float]]:
        """Return the details needed for spawning.

        Returns:
            (tuple[Any, Any, tuple[float, float, float], tuple[float, float, float]]):
            A tuple containing:
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

    def get_point(self, offset: tuple[float, float]) -> carla.Location:
        """Get a CARLA location based on an offset from the walker's current position.

        This function calculates a new CARLA location by adding the provided x and y offsets
        to the walker's current location.

        Args:
            offset (tuple[float, float]): A tuple containing the x and y offsets to apply.

        Returns:
            carla.Location: The new CARLA location after applying the offset.
        """
        if not self.walker or not self.walker.is_alive:
            raise WalkerNotAliveError(self.scenario_id)

        cur = self.walker.get_location()
        offset_x, offset_y = offset
        return carla.Location(cur.x + offset_x, cur.y - offset_y, 0.5)

    def compute_collision_point(self) -> None:
        """Estimate and draw the future velocity vectors for the walker and ego vehicle.

        This debug function calculates the walker's intended direction towards its next
        goal (from `self.path_controller_2`) and scales its current velocity vector along
        this direction. It also gets the ego vehicle's current velocity.
        It then draws these scaled velocity vectors as lines in the CARLA simulator,
        originating from the current positions of the walker and the vehicle.
        This can help visualize if their future paths might intersect.

        Note:
        - This function assumes `self.walker`, `self.path_controller_2`, and `self.player` are
          valid.
        - `l2_length` is assumed to be a helper function that computes the magnitude of a
          `carla.Vector3D`.
        - The drawn lines represent velocity vectors scaled by a factor of 2
          (for visibility).
        """
        if not self.my_world.debug:  # Check debug flag
            return

        if not self.walker or not self.walker.is_alive:
            raise WalkerNotAliveError(self.scenario_id)
        if not self.player or not self.player.is_alive:
            raise PlayerNotAliveError(self.scenario_id)

        # self.assert_walker_is_alive(raise_error=True)
        # self.assert_player_is_alive(raise_error=True)

        if not hasattr(self, "path_2") or not self.path_2:
            path_2 = (
                self.path_controller_2.path[0]
                if hasattr(self, "path_controller_2")
                and self.path_controller_2
                and self.path_controller_2.path
                else None
            )
            if not path_2:
                logger.warning(
                    "compute_collision_point prerequisites not met (walker, path_controller_2).",
                )
                return
        else:
            path_2 = self.path_2[0]

        walker_loc = self.walker.get_location()

        # Goal is the first point in the walker's second planned path segment
        goal_loc = path_2

        walker_dir = goal_loc - walker_loc  # Vector from walker to goal
        car_loc = self.player.get_location()
        walker_vel = self.walker.get_velocity()
        walker_vel = walker_dir * l2_length(walker_vel) / l2_length(walker_dir)
        car_vel = self.player.get_velocity()

        # Draw the walker's estimated future velocity vector (scaled by 2)
        self.world.debug.draw_line(
            walker_loc,
            walker_loc + 2 * walker_vel,
            thickness=0.05,
            color=carla.Color(255, 255, 255),
        )

        # Draw the car's current velocity vector (scaled by 2)
        self.world.debug.draw_line(
            car_loc,
            car_loc + 2 * car_vel,
            thickness=0.05,
            color=carla.Color(255, 255, 255),
        )

    def _compute_plans(
        self,
        offsets: list[tuple[int | float, int | float]],
        base_location: carla.Location,
        color: carla.Color | None = None,
    ) -> list[carla.Location]:
        """Compute a sequence of CARLA locations (a plan) based on offsets from a start location.

        This function iteratively calculates new locations by adding x and y offsets to the
        `base_location`. The y-offset is subtracted, which is a common convention in some
        coordinate systems or specific scenario setups. If debugging is enabled in the world and a
        color is provided, it also draws a debug point at each computed location in the CARLA
        simulator.

        Args:
            offsets (list[tuple[float, float]]): A list of (x_offset, y_offset) tuples.
                Each tuple defines the displacement from the `base_location` for a new point.
            base_location (carla.Location): The starting CARLA location from which offsets are
                applied.
            color (carla.Color | None): If provided and `self.world.debug` is True,
                a debug point of this color will be drawn at each computed location.
                Defaults to None.

        Returns:
            list[carla.Location]: A list of `carla.Location` objects representing the computed
                plan.
        """
        plan = []

        # The 'cur' variable is not updated in the loop in the original code,
        # meaning all offsets are relative to the initial 'base_location'.
        # This behavior is preserved.
        cur = base_location
        for offset_x, offset_y in offsets:
            # Z-coordinate is fixed at 0.5.
            # Y-offset is subtracted as per the original implementation.
            loc = carla.Location(cur.x + offset_x, cur.y - offset_y, 0.5)
            plan.append(loc)
            if color is not None and self.world.debug:
                self.world.debug.draw_point(loc, size=0.1, color=color, life_time=0)
        return plan

    def _get_p_from_vector(
        self,
        loc1: carla.Location,
        loc2: carla.Location,
        perc: float,
    ) -> carla.Location:
        """Calculate an intermediate point along the vector defined by two CARLA locations.

        This function computes a point that lies at a specified percentage `perc` along the
        straight line segment connecting `loc1` to `loc2`. A `perc` value of 0.0 would return
        `loc1`, and a `perc` value of 1.0 would return `loc2`.

        Args:
            loc1 (carla.Location): The starting CARLA location of the vector.
            loc2 (carla.Location): The ending CARLA location of the vector.
            perc (float): The percentage along the vector from `loc1` to `loc2`
                where the intermediate point should be calculated (e.g., 0.5 for midpoint).

        Returns:
            carla.Location: The calculated intermediate `carla.Location`.
        """
        vec = loc2 - loc1  # Calculate the vector from loc1 to loc2
        return loc1 + perc * vec  # Scale the vector by perc and add to loc1

    def _draw_circle(self, loc: carla.Location, radius: float) -> None:
        """Draw a debug circle in the CARLA simulator at a specified location and radius.

        This function iterates through angles from 0 to 360 degrees (in 2-degree steps)
        to calculate points on the circumference of a circle. It then draws small
        debug points at these locations. The circle is drawn on the XY plane (z=0 relative
        to the input `loc`). This is useful for visualizing circular areas or ranges.
        The points are only drawn if `self.my_world.debug` is True.

        Args:
            loc (carla.Location): The center CARLA location of the circle.
            radius (float): The radius of the circle in meters.
        """
        if not self.my_world.debug:  # Check debug flag
            return
        for i in range(0, 360, 2):  # Iterate every 2 degrees for a smoother circle
            # Calculate x and y coordinates on the circle
            x = radius * math.cos(math.radians(i))
            y = radius * math.sin(math.radians(i))

            # Create the point location relative to 'loc'.
            # Note: Original code uses -x, y. This means the circle starts drawing
            # from the left side (negative x relative to center) and proceeds counter-clockwise.
            # Z-coordinate of the drawn points will be the same as loc.z.
            point_to_draw = loc + carla.Location(-x, y, 0)

            self.world.debug.draw_point(  # Use self.world (carla.World) for debug drawing
                point_to_draw,
                size=0.05,  # Size of each debug point
                color=carla.Color(255, 165, 0),  # Orange color
                # Point persists until next tick's clear, or indefinitely if 0 and not cleared
                life_time=0,
            )

    def _draw_db_circle(self) -> None:
        """Draw two debug circles representing the boundaries of a "decision box" (DB).

        This function visualizes the decision box, typically used for pedestrian
        decision-making, as two concentric circles centered at `self.desc_p` (a decision point).
        The radii of the circles are taken from `self.db[0]` (inner radius) and
        `self.db[1]` (outer radius). This is useful for debugging scenario logic
        that depends on these circular decision zones. The circles are only drawn if
        `self.my_world.debug` is True and `self.desc_p` and `self.db` are properly set.

        Raises:
            AttributeError: If `self.desc_p` or `self.db` is not set on the instance.
            IndexError: If `self.db` does not contain at least two elements.
        """
        if not self.my_world.debug:  # Check debug flag
            return

        if not hasattr(self, "desc_p") or self.desc_p is None:
            logger.warning("_draw_db_circle called but self.desc_p is not set.")
            return
        if not hasattr(self, "db") or self.db is None or len(self.db) < 2:
            logger.warning("_draw_db_circle called but self.db is not properly set.")
            return

        # Draw the inner circle using the first radius in self.db
        self._draw_circle(self.desc_p, self.db[0])
        # Draw the outer circle using the second radius in self.db
        self._draw_circle(self.desc_p, self.db[1])

    def _draw_grid(self) -> None:
        """Draw a debug grid around the current primary walker's location.

        This function visualizes a square grid in the CARLA simulator, centered
        (or starting from) the walker's current position. The grid lines are drawn
        on the XY plane. This can be useful for understanding spatial relationships
        and distances relative to the walker during scenario debugging.
        The grid is only drawn if `self.my_world.debug` is True and `self.walker` is valid.

        The grid has a fixed `width` of 20 units (meters).
        - It draws the outer bounding box of the 20x20 grid.
        - It then draws horizontal and vertical lines within this box, creating 1x1 cells.
        The y-coordinates are inverted in the drawing logic
        (e.g., `loc + carla.Location(0, -width, 0)`).
        """
        if not self.my_world.debug:  # Check debug flag
            return
        if not self.walker or not self.walker.is_alive:
            logger.warning("_draw_grid called but self.walker is not valid.")
            return

        width = 20  # Defines the size of the square grid (20x20 meters)
        loc = self.walker.get_location()  # Get current walker location as the origin/corner

        # Define the four corners of the outer grid boundary
        # Note: y-offsets are negative, effectively inverting y-axis for these points
        # Starting point (bottom-left from walker's perspective if y is inverted)
        upper_left_corner = loc
        upper_right_corner = loc + carla.Location(width, 0, 0)  # Extends +width in X

        # Extends -width in Y (effectively "down" on map)
        lower_left_corner = loc + carla.Location(0, -width, 0)
        lower_right_corner = loc + carla.Location(width, -width, 0)  # Corner opposite to loc

        # Draw the outer boundary lines of the grid
        self.world.debug.draw_line(
            upper_left_corner,
            lower_left_corner,
            thickness=0.02,
            life_time=0,
        )  # Left vertical
        self.world.debug.draw_line(
            upper_left_corner,
            upper_right_corner,
            thickness=0.02,
            life_time=0,
        )  # Top horizontal
        self.world.debug.draw_line(
            lower_left_corner,
            lower_right_corner,
            thickness=0.02,
            life_time=0,
        )  # Bottom horizontal
        self.world.debug.draw_line(
            upper_right_corner,
            lower_right_corner,
            thickness=0.02,
            life_time=0,
        )  # Right vertical

        # Draw the inner grid lines
        for i in range(1, width):  # Iterate from 1 to width-1 to draw lines inside
            # Horizontal lines: iterate along y-axis (inverted)
            offset_y_current = carla.Location(0, -i, 0)
            start_horizontal_line = upper_left_corner + offset_y_current
            end_horizontal_line = upper_right_corner + offset_y_current
            self.world.debug.draw_line(
                start_horizontal_line,
                end_horizontal_line,
                thickness=0.02,
                life_time=0,
            )

            # Vertical lines: iterate along x-axis
            offset_x_current = carla.Location(i, 0, 0)
            start_vertical_line = upper_left_corner + offset_x_current
            end_vertical_line = lower_left_corner + offset_x_current
            self.world.debug.draw_line(
                start_vertical_line,
                end_vertical_line,
                thickness=0.02,
                life_time=0,
            )

    def _draw_point(
        self,
        p: carla.Location,
        color: carla.Color | None = carla.Color(r=0, g=255, b=255),  # noqa: B008
    ) -> None:
        """Draw a debug point in the CARLA simulator at a specified location.

        This function is a utility for visualizing points of interest during scenario development
        or debugging. The point is only drawn if `self.world.debug` is True.

        Args:
            p (carla.Location): The CARLA location where the debug point should be drawn.
            color (carla.Color | None): The color of the debug point.
                Defaults to a cyan-like color (RGB: 0, 255, 255).
        """
        if self.my_world.debug:
            self.world.debug.draw_point(p, size=0.1, color=color, life_time=0)

    def _draw_db(
        self,
        db: list[float] | None = None,
        color: carla.Color | None = carla.Color(0, 255, 0),  # noqa: B008
    ) -> None:
        """Draw debug lines representing a "decision box" in the CARLA simulator.

        This function visualizes a rectangular region, typically used to define an area where a
        pedestrian makes a decision. The box is defined by two y-coordinates relative to
        `self.desc_p.y` (a decision point's y-coordinate, assumed to be set on the instance) and
        fixed x-coordinates (83 and 103). The lines are only drawn if `self.world.debug` is True
        and `self.desc_p` is set on the scenario instance.

        Args:
            db (list[float] | None): A list or tuple of two floats `[y_offset1, y_offset2]`
                defining the y-offsets from `self.desc_p.y` for the decision box lines.
                If None, it uses `self.db` from the scenario instance. Defaults to None.
            color (carla.Color | None): The color of the debug lines.
                Defaults to green (RGB: 0, 255, 0).
        """
        if not self.world.debug or not hasattr(self, "desc_p") or self.desc_p is None:
            return

        current_db = db
        if current_db is None:
            if self.db is None:  # self.db is an attribute of the scenario instance
                return
            current_db = self.db

        # Ensure current_db has at least two elements
        if len(current_db) < 2:
            logger.warning(f"Decision box `db` has fewer than 2 elements: {current_db}")
            return

        # Fixed x-coordinates for the lines, z is fixed at 0.5
        x_min, x_max, z_val = 83.0, 103.0, 0.5

        # First line of the decision box
        y1 = self.desc_p.y + current_db[0]
        left1 = carla.Location(x_min, y1, z_val)
        right1 = carla.Location(x_max, y1, z_val)
        self.world.debug.draw_line(left1, right1, thickness=0.05, color=color, life_time=0)

        # Second line of the decision box
        y2 = self.desc_p.y + current_db[1]
        left2 = carla.Location(x_min, y2, z_val)
        right2 = carla.Location(x_max, y2, z_val)
        self.world.debug.draw_line(left2, right2, thickness=0.05, color=color, life_time=0)

    def set_walker_speed_relative(self, per: float) -> None:
        """Set the speed of the scenario's walker relative to its current speed.

        This function modifies the walker's target speed by multiplying its current speed component
        in the `carla.WalkerControl` by the given percentage `per`.

        Args:
            per (float): The percentage to scale the walker's current speed.
                For example, 0.5 will halve the speed, 2.0 will double it.
        """
        if not self.walker or not self.walker.is_alive:
            raise WalkerNotAliveError(self.scenario_id)

        control = self.walker.get_control()
        control.speed = per * control.speed
        self.walker.apply_control(control)

    def decision_trigger(
        self,
        distance: float,
        db: list[float | int] | None,
        *,
        without_speed: bool = False,
    ) -> bool:
        """Determine if to trigger decision based on distance, speed, and decision boundaries.

        In random mode (`self.world.random` is True), the decision is made randomly.
        Otherwise, it checks if the given `distance` falls within the decision box `db`
        boundaries. Additionally, it considers the ego vehicle's speed: the trigger
        occurs if the speed is greater than 1.5 m/s, or if `without_speed` is True
        (ignoring the speed check).

        Args:
            distance (float): The distance to the relevant object or point for decision making.
            db (Optional[List[float]]): A list or tuple of two floats `[min_dist, max_dist]`
                defining the decision box boundaries. If None, the function will log a
                warning and return False.
            without_speed (bool): If True, the vehicle's speed check (must be > 1.5 m/s)
                is bypassed. Defaults to False.

        Returns:
            bool: True if the decision is triggered, False otherwise.
        """
        if self.my_world.random:
            choice = RNG.choice(2)
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

        if len(db) < 2:
            logger.warning(f"Decision box `db` has fewer than 2 elements: {db}")
            return False

        return db[0] <= distance <= db[1] and (speed > 1.5 or without_speed)

    def second_decider(self, distance: float, dec_d: float | None = None) -> bool:
        """Implement a secondary decision logic based on distance, speed, or random timed behavior.

        This function is used for more complex decision-making in scenarios.
        If `self.world.random` is True, it implements a timed random decision:
        on the first call for a random event, it sets a random waiting time and returns False.
        On subsequent calls, it checks if the accumulated simulation time (`self.waiting_c` * step)
        has exceeded `self.waiting_time`.

        If not in random mode, it checks conditions based on the `distance` to an object
        and the ego vehicle's speed.
        - If `dec_d` is None: returns True if `distance < 0` (object is behind) or if vehicle speed
          is < 1 m/s.
        - If `dec_d` is provided: returns True if `distance + dec_d < 0` or
          if (vehicle speed < 1 m/s AND `distance > 2.5`).

        Note: This method relies on instance attributes `self.second_choice`, `self.waiting_c`,
        and `self.waiting_time`, which should be initialized (e.g., in the scenario's `__init__`
        or `setup` method) before this function is called in random mode.

        Args:
            distance (float): The current distance relevant to the decision.
            dec_d (Optional[float]): An additional distance delta used in one of the
                deterministic decision conditions. Defaults to None.

        Returns:
            bool: True if the secondary decision criteria are met, False otherwise.
        """
        # Ensure instance attributes are initialized if not already (e.g., in __init__)
        if not hasattr(self, "second_choice"):
            self.second_choice = None
        if not hasattr(self, "waiting_c"):
            self.waiting_c = 0
        if not hasattr(self, "waiting_time"):
            self.waiting_time = 0.0

        if self.my_world.random:
            if self.second_choice:  # True if a random event is in progress
                simulation_step = 0.05  # Assumed simulation step time
                self.waiting_c += 1
                return self.waiting_c * simulation_step > self.waiting_time

            # Start a new random event
            self.waiting_time = RNG.random() * 2.0 + 1.0
            self.waiting_c = 0
            self.second_choice = True  # Mark that a random event has started
            return False  # Decision not met on the first call of a new random event

        if self.my_world.player is None:
            logger.error("Second decider called but self.world.player is None.")
            return False  # Or some default safe behavior

        velocity = self.my_world.player.get_velocity()
        speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5

        if dec_d is None:
            return distance < 0 or speed < 1  # Vehicle speed < 1 m/s (approx 3.6 km/h)

        # Vehicle speed < 1 m/s AND pedestrian is at least 2.5m away
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

    def __str__(self) -> str:
        """Return a string representation of the scenario."""
        class_attrs = ", ".join(
            f"{key}={value!r}"
            for key, value in self.__dict__.items()
            if not key.startswith("_")
            and "scenario_id" not in key
            and "my_world" not in key
            and "world" not in key
            and "config" not in key
        )
        return (
            f"BaseScenario(\n\tscenario_id={self.get_scenario_id()},\n"
            f"\tWorld={self.my_world},\n"
            f"\tConfig={self.config},\n"
            f"\tAttrs=" + "{" + class_attrs + "}\n" + "" + ")\n"
        )


class Scenario01Int(BaseScenario):
    """Implementation for Interactive Scenario 01."""

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        super().__init__(world, config)

        self.scenario_id = "01_int"  # Unique identifier for this scenario

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

        return self.scenario_id, obstacles, end, start

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        # Override default if necessary, based on original world.py logic
        return ICR.LOW, SON.AVERTING  # TODO: Adjust based on scenario logic

    def setup(self) -> None:
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")
        obstacles = self.get_obstacle_blueprints()
        conf = self.config  # Use the passed config

        if self.my_world.player is None:
            logger.error("Player actor is not set in the world. Cannot set up scenario.")
            return

        spawning_distance = conf.spawning_distance
        walking_distance = conf.walking_distance
        looking_distance = conf.looking_distance
        crossing_distance = conf.crossing_distance
        reenter_distance = conf.reenter_distance
        op_reenter_distance = conf.op_reenter_distance
        self.ped_speed = conf.ped_speed

        street_delta = 3 if conf.char == "yielding" else 5
        self.db = [-1.0, 15.0] if conf.char == "yielding" else [-1.0, 20.0]
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
            return

        self.my_world.walker = self.walker  # IMPORTANT: Assign to world instance

        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        self.walker.on_street = False  # Initialize walker state

        # --- First path setup ---
        offsets_1 = [(0.0, walking_distance), (street_delta, walking_distance + crossing_distance)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.my_world.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # --- Second path setup ---
        offsets_2 = [
            (9.5, walking_distance + crossing_distance + op_reenter_distance),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 2),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 20),
        ]
        path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=0, g=255, b=0) if self.my_world.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            path_2,
            self.ped_speed * mult,
        )

        # --- Third path setup ---
        reenter = walking_distance + crossing_distance + reenter_distance
        offsets_3 = [(0.0, reenter), (0.0, reenter + 5.0)]
        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(r=0, g=0, b=255) if self.my_world.debug else None,
        )
        self.path_controller_3 = PathController(
            self.world,
            self.walker,
            path_3,
            self.ped_speed,
        )

        # Note: get_point needs fixing if it relies on self.walker which might not be set when
        # get_point is called
        # Let's assume base_loc can be used for reference before walker spawns if needed
        # Or refactor get_point to take the reference location
        # For now, assuming setup order is okay.
        turn_p = self.get_point((0, looking_distance * walking_distance))
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
            path_2[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        if self.dummy_car:
            player_loc = self.my_world.player.get_location()
            breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
            car_behave = conf.char == "forcing"
            car_to_desc = self.my_world.player.get_location().y - self.desc_p.y - self.db[1]

            ped_to_desc = carla.Location.distance(self.walker.get_location(), self.desc_p)
            ped_time = ped_to_desc / self.ped_speed if self.ped_speed > 0 else float("inf")

            # TODO: Introduce default speed in scenario config
            speed = car_to_desc / ped_time if ped_time != float("inf") else 8.0

            self.my_world.car_controller = CarController(
                self.my_world.player,
                breaking_point,
                speed=speed - 1 if car_behave else speed,  # TODO: Ensure a minimum speed
                yielding=car_behave,
            )

    def tick(self) -> None:
        if not self.walker or not self.path_controller_1:  # Check if setup was successful
            return

        # --- Pedestrian's Initial Movement and Actions ---
        # The pedestrian attempts to follow the first defined path (path_1).
        # path_1 typically involves walking parallel to curb and then turning towards the street.
        # The `status` variable will indicate if the pedestrian has completed this path segment.
        status = self.path_controller_1.step()

        # The pedestrian executes a "look behind right" behavior.
        # This is usually triggered when the pedestrian reaches the start of path_1 (curb edge).
        # The `conf.char` (character: "yielding" or "forcing") influences intensity of this look.
        self.look_behind_right.step()

        # The pedestrian executes a "turn head" behavior.
        # This is triggered at `turn_p` (a point `looking_distance` along the initial walk parallel
        # to the curb). It simulates the pedestrian looking towards oncoming traffic before
        # committing to cross.
        self.turn_head.step()

        # Access car_controller from my_world
        if self.dummy_car and self.my_world.car_controller:
            self.my_world.car_controller.step()

        # --- Pedestrian Decision Making and Path Execution ---
        # The `self.choice` variable stores the pedestrian's decision (e.g., "Left", "Right", or
        # None if undecided).
        if self.choice == "Left":
            # If the pedestrian has decided to go "Left" (typically meaning to turn back or avoid
            # crossing directly):
            # Execute a "look behind left" behavior, potentially with increased intensity (mult=2).
            self.look_behind_left.step()
            if status == "Done":
                # If the first path segment (path_1) is completed:
                # Reset the pedestrian's pose (e.g., clear any specific head/body turns).
                self.reset.step()

                # Start following the third defined path (path_3), which represents returning to
                # the original curb or a safer position.
                self.path_controller_3.step()

        elif self.choice == "Right":
            # If the pedestrian has decided to go "Right" (typically meaning to continue crossing):
            # Reset the pedestrian's pose.
            self.reset.step()
            if status == "Done":
                # If the first path segment (path_1) is completed:
                # Start following the second defined path (path_2), which represents crossing the
                # street. The speed might be modified by `mult`.
                self.path_controller_2.step()

        elif l2_distance(self.walker.get_location(), self.desc_p) < 0.1:
            # If no decision has been made yet (`self.choice` is None) AND
            # pedestrian is very close (within 0.1 meters) to the decision point (`self.desc_p`):
            # `self.desc_p` is a point 95% along the segment of path_1 that turns towards street.

            if not self.my_world.player:
                logger.error("Player actor is not set in the world. Cannot tick scenario.")
                return

            # Calculate the distance to the ego vehicle, adjusted by -2 meters.
            # This `distance` is used as input for the `decision_trigger`.
            distance = (
                y_distance(self.walker.get_location(), self.my_world.player.get_location()) - 2
            )

            # Use the `decision_trigger` function to determine the choice.
            # This function considers calculated `distance`, decision box boundaries (`self.db`),
            # and the ego vehicle's speed.
            if self.decision_trigger(distance, self.db):
                # If the decision trigger is met (e.g., car is within a certain "yielding" zone):
                self.choice = "Left"  # Pedestrian decides to yield/avoid.
                # Update pedestrian's internal cognitive state:
                self.walker.icr = ICR.VERY_LOW
                self.walker.son = SON.AVERTING
            else:
                # If the decision trigger is not met (e.g., car is far or pedestrian is "forcing"):
                self.choice = "Right"  # Pedestrian decides to cross/continue.
                # Update pedestrian's internal cognitive state:
                self.walker.icr = ICR.GOING_TO
                self.walker.son = SON.FORCING  # TODO: Check if this is true

        # --- Continuous Behaviors ---
        self.relaxer.step()

        # The pedestrian's internal state (ICR and SON) is updated by `InternalStateSetter`.
        # This is triggered when the pedestrian reaches the start of path_2 (`path_2[0]`).
        # It sets ICR to VERY_LOW and SON to AVERTING, typically after successfully crossing.
        self.iss_crossed.step()


class Scenario02Int(BaseScenario):
    """Implementation for Interactive Scenario 02.

    In this scenario, the pedestrian starts on the sidewalk, walks parallel to the
    curb, and then turns to cross the street. The key interaction happens as the
    pedestrian approaches a decision point in the road. Based on the ego vehicle's
    position and speed, the pedestrian decides whether to continue crossing or to stop
    and yield. The pedestrian's initial character ('forcing' or 'yielding')
    influences the decision boundaries and behavior, such as deceleration rates.
    """

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        """Initialize scenario-specific attributes."""
        super().__init__(world, config)
        self.scenario_id = "02_int"

        # Initialize controllers and paths to None for clarity
        self.path_controller_1: PathController | None = None
        self.path_controller_2: PathController | None = None
        self.turn_head: TurnHeadRightBehind | None = None
        self.look_right: TurnHeadRightWalk | None = None
        self.reset: ResetPose | None = None
        self.relaxer: Relaxer | None = None
        self.iss_crossed: InternalStateSetter | None = None

        # Path and decision point attributes
        self.path_2: list[carla.Location] = []
        self.desc_p: carla.Location | None = None
        self.speed_schedule_stop: list[tuple[carla.Location, float]] = []
        self.speed_schedule_cross: list[tuple[carla.Location, float]] = []

    def get_spawn_details(
        self,
    ) -> tuple[str, list, tuple[float, float, float], tuple[float, float, float]]:
        """Return the spawn details for the ego vehicle and obstacles."""
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

        return self.scenario_id, obstacles, end, start

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        """Return the initial cognitive state of the walker."""
        # This scenario starts with the pedestrian interested in crossing.
        return ICR.INTERESTED, SON.YIELDING if self.config.char == "yielding" else SON.FORCING

    def setup(self) -> None:
        """Set up the scenario, spawning actors and initializing controllers."""
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")
        obstacles = self.get_obstacle_blueprints()
        conf = self.config

        if self.player is None:
            raise PlayerNotAliveError(self.scenario_id)

        # --- Parameter extraction from config ---
        spawning_distance = conf.spawning_distance
        walking_distance = conf.walking_distance
        looking_distance = conf.looking_distance
        crossing_distance = conf.crossing_distance
        op_reenter_distance = conf.op_reenter_distance
        self.ped_speed = conf.ped_speed

        street_delta = 3 if conf.char == "yielding" else 5
        self.db = [-1.0, 15.0] if conf.char == "yielding" else [-1.0, 10.0]

        # --- Spawn Walker ---
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        if not self.walker:
            logger.error(f"Failed to spawn walker for {self.get_scenario_id()}")
            return

        self.my_world.walker = self.walker
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()  # Allow walker to settle
        self.walker.on_street = False

        # --- Path Planning ---
        # Path 1: Walk to curb and turn towards street
        offsets_1 = [(0.0, walking_distance), (street_delta, walking_distance + crossing_distance)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.my_world.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # Path 2: Continue crossing the street
        offsets_2 = [
            (9.5, walking_distance + crossing_distance + op_reenter_distance),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 2),
            (10.5, walking_distance + crossing_distance + op_reenter_distance + 10),
        ]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=0, g=255, b=0) if self.my_world.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed,
        )

        # --- Controller Initialization ---
        turn_p = self.get_point((0, looking_distance * walking_distance))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)
        self.look_right = TurnHeadRightWalk(self.walker, path_1[0], conf.char)
        self.reset = ResetPose(self.walker)

        # --- Decision Point and Speed Schedules ---
        vec = path_1[1] - path_1[0]
        self.desc_p = path_1[0] + 0.9 * vec

        # Schedules are for path_controller_1, to be applied on decision
        self.path_controller_1.speed_schedule = [
            (path_1[0] + per * vec, 0.93) for per in [0.0, 0.2, 0.4]
        ]
        vec_2 = self.path_2[0] - path_1[1]
        self.speed_schedule_stop = [(path_1[1] + per * vec_2, 1.355) for per in [0.0, 0.05, 0.075]]
        self.speed_schedule_cross = [(path_1[1] + per * vec_2, 1.075) for per in [0.0, 0.05]]

        self.relaxer = Relaxer(self.walker, self.player, path_1[0] + 0.2 * vec)

        # --- Cognitive State Initialization ---
        initial_icr, initial_son = self.get_initial_walker_state()
        self.walker.icr = initial_icr
        self.walker.son = initial_son
        self.walker.initial_son = initial_son
        self.iss_crossed = InternalStateSetter(
            self.walker,
            self.path_2[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        # --- Debug Drawing ---
        if self.my_world.debug:
            self._draw_db()
            self._draw_point(self.desc_p)
            self._draw_grid()

        # --- Dummy Car Controller ---
        if self.dummy_car:
            if self.desc_p is None:
                logger.error("desc_p cannot be None at this point")
            else:
                player_loc = self.player.get_location()
                breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
                car_behave = conf.char == "forcing"
                car_to_desc = player_loc.y - self.desc_p.y - self.db[1]
                ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
                ped_time = ped_to_desc / self.ped_speed if self.ped_speed > 0 else float("inf")
                speed = car_to_desc / ped_time if ped_time != float("inf") else 8.0

                self.my_world.car_controller = CarController(
                    self.player,
                    breaking_point,
                    speed=speed if car_behave else speed - 1,
                    yielding=car_behave,
                )

    def tick(self) -> None:
        """Execute one simulation step for Scenario 02."""
        # Update references
        self.walker = self.my_world.walker
        self.player = self.my_world.player

        # If any attributes are missing, raise a specific error
        if (
            not self.walker
            or not self.player
            or not self.path_controller_1
            or not self.path_controller_2
            or not self.desc_p
            or not self.turn_head
            or not self.look_right
            or not self.reset
            or not self.relaxer
            or not self.iss_crossed
        ):
            raise ScenarioSetupError(self.scenario_id)

        # --- Pedestrian's Initial Movement and Actions ---
        status = self.path_controller_1.step()
        self.turn_head.step()
        self.look_right.step()

        if self.dummy_car and self.my_world.car_controller:
            self.my_world.car_controller.step()

        # --- Decision Logic ---
        if self.choice == "Left":
            self.reset.step()
            if status == "Done":
                self.path_controller_1.set_done()
                self.walker.apply_control(
                    carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0),
                )
        elif self.choice == "Right":
            self.reset.step()
            if status == "Done":
                self.path_controller_2.step()
        elif y_distance(self.walker.get_location(), self.desc_p) < 0.1:
            distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2

            if self.decision_trigger(distance, self.db):
                self.choice = "Left"
                self.walker.icr = ICR.LOW
                self.walker.son = SON.YIELDING
                self.path_controller_1.speed_schedule = self.speed_schedule_stop
            else:
                self.choice = "Right"
                self.walker.icr = ICR.GOING_TO
                self.walker.son = self.walker.initial_son
                self.path_controller_1.speed_schedule = self.speed_schedule_cross

        self.relaxer.step()
        self.iss_crossed.step()

        # Update references
        self.my_world.player = self.player
        self.my_world.walker = self.walker


class Scenario03Int(BaseScenario):
    """Implementation for Interactive Scenario 03.

    This scenario features a pedestrian starting on the sidewalk and walking directly
    towards the street, perpendicular to the curb. The interaction is characterized
    by the pedestrian's continuous movement towards the road, with decisions and
    behavioral changes (like acceleration or deceleration) happening based on the
    ego vehicle's proximity at specific trigger points. The pedestrian signals their
    intent by looking towards the oncoming car. The decision to ultimately cross or
    stop is made at a decision point on the road.
    """

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        """Initialize scenario-specific attributes."""
        super().__init__(world, config)
        self.scenario_id = "03_int"

        # Initialize controllers and paths to None
        self.path_controller_1: PathController | None = None
        self.path_controller_2: PathController | None = None
        self.turn_head: TurnHeadLeftWalk | None = None
        self.relaxer: Relaxer | None = None
        self.iss_crossed: InternalStateSetter | None = None

        # Scenario state and configuration attributes
        self.slow_db: list[float] = []
        self.acc_db: list[float] = []
        self.flip_choice: str | None = None
        self.init_char: str = ""
        self.second_choice: bool = False
        self.desc_p: carla.Location | None = None
        self.flip_p: carla.Location | None = None
        self.acc_p: carla.Location | None = None
        self.speed_schedule_stop: list[tuple[carla.Location, float]] = []

    def get_spawn_details(
        self,
    ) -> tuple[str, list, tuple[float, float, float], tuple[float, float, float]]:
        """Return the spawn details for the ego vehicle and obstacles."""
        start = (92.5, 300, -90)
        end = (92.5, 200, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 107
        walker_spawn_point.location.y = 300
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 180.0
        walker = [walker_bp, walker_spawn_point]
        obstacles.append(walker)

        return self.scenario_id, obstacles, end, start

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        """Return the initial cognitive state of the walker."""
        # The pedestrian's initial state is determined by the 'char' config
        return ICR.INTERESTED, SON.YIELDING if self.config.char == "yielding" else SON.FORCING

    def setup(self) -> None:
        """Set up the scenario, spawning actors and initializing controllers."""
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")
        obstacles = self.get_obstacle_blueprints()
        conf = self.config

        if self.player is None:
            raise PlayerNotAliveError(self.scenario_id)

        # --- Parameter extraction from config ---
        spawning_distance = conf.spawning_distance
        looking_distance = conf.looking_distance
        self.ped_speed = conf.ped_speed
        self.init_char = conf.char

        self.db = [0.0, 20.0]  # if conf.char == "yielding" else [0.0, 20.0]
        self.slow_db = [20.0, 38.0]
        self.acc_db = [20.0, 38.0]
        self.second_choice = False

        # --- Spawn Walker ---
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        if not self.walker:
            logger.error(f"Failed to spawn walker for {self.get_scenario_id()}")
            return

        self.my_world.walker = self.walker
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()
        self.walker.on_street = False

        # --- Path Planning ---
        # Path 1: Walk directly towards the street
        street_x = 95
        offsets_1 = [(street_x - spawn_loc.x, 0.0)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(r=255, g=0, b=0) if self.my_world.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # Path 2: Path after crossing (if needed, here it's short)
        offsets_2 = [(-21.0, 0.0)]
        path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(r=0, g=255, b=0) if self.my_world.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            path_2,
            self.ped_speed,
        )

        # --- Trigger Points and Speed Schedules ---
        decision_point_perc = 0.88 if conf.char == "forcing" else 0.8
        self.desc_p = self._get_p_from_vector(spawn_loc, path_1[0], decision_point_perc)
        self.flip_p = self._get_p_from_vector(spawn_loc, path_1[0], 0.4)
        self.acc_p = self._get_p_from_vector(spawn_loc, path_1[0], 0.55)

        self.speed_schedule_stop = [
            (self._get_p_from_vector(spawn_loc, path_1[0], per), 0.85) for per in [0.87, 0.92]
        ]

        # --- Controller Initialization ---
        self.turn_head = TurnHeadLeftWalk(
            self.walker,
            start_pos=self._get_p_from_vector(spawn_loc, path_1[0], looking_distance),
            char=conf.char,
        )
        self.relaxer = Relaxer(self.walker, self.player, self.flip_p)

        # --- Cognitive State Initialization ---
        initial_icr, initial_son = self.get_initial_walker_state()
        self.walker.icr = initial_icr
        self.walker.son = initial_son
        self.walker.initial_son = initial_son
        self.iss_crossed = InternalStateSetter(
            self.walker,
            path_2[0],
            icr=ICR.VERY_LOW,
            son=SON.AVERTING,
        )

        # --- Debug Drawing ---
        if self.my_world.debug:
            self._draw_point(spawn_loc, color=carla.Color(255, 0, 0))
            self._draw_point(self.desc_p)
            self._draw_point(self.flip_p, carla.Color(0, 0, 255))
            self._draw_point(self.acc_p, carla.Color(0, 255, 0))
            self._draw_db(db=self.slow_db)
            self._draw_db(db=self.acc_db, color=carla.Color(255, 0, 0))
            self._draw_db(self.db, color=carla.Color(0, 0, 255))

        # --- Dummy Car Controller ---
        if self.dummy_car:
            if self.desc_p is None:
                logger.error("desc_p cannot be None at this point")
            else:
                player_loc = self.player.get_location()
                breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
                car_behave = conf.char == "forcing"
                car_to_desc = player_loc.y - self.desc_p.y - self.db[1]
                ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
                ped_time = ped_to_desc / self.ped_speed if self.ped_speed > 0 else float("inf")
                speed = car_to_desc / ped_time if ped_time != float("inf") else 4.0

                self.my_world.car_controller = CarController(
                    self.player,
                    breaking_point,
                    speed=speed,
                    yielding=car_behave,
                )

    def tick(self) -> None:
        """Execute one simulation step for Scenario 03."""
        # Update references
        self.player = self.my_world.player
        self.walker = self.my_world.walker

        if (
            not self.walker
            or not self.player
            or not self.path_controller_1
            or not self.path_controller_2
            or not self.desc_p
            or not self.turn_head
            or not self.relaxer
            or not self.iss_crossed
            or not self.flip_p
            or not self.acc_p
        ):
            raise ScenarioSetupError(self.scenario_id)

        if self.dummy_car:
            self.my_world.car_controller.step()

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
                        self.path_controller_1.cur_speed = self.path_controller_1.cur_speed * 0.7
                        self.turn_head.relax_spine()
                        self.walker.icr = ICR.INTERESTED
                        self.walker.son = SON.YIELDING
                    else:
                        self.flip_choice = "StandardAcc"
                        # print(self.flip_choice)
                        self.set_walker_speed_relative(1.1)
                        self.path_controller_1.cur_speed = self.path_controller_1.cur_speed * 1.1
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
                        self.path_controller_1.cur_speed = self.path_controller_1.cur_speed * 0.8
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
                distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2
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

        # Update references
        self.my_world.player = self.player
        self.my_world.walker = self.walker


class Scenario04Int(BaseScenario):
    """Implementation for Interactive Scenario 04.

    This scenario simulates a pedestrian who may have misjudged the vehicle's speed.
    The pedestrian walks along the curb, turns to cross, and then, while already on
    the road, performs a second check. Based on the vehicle's proximity at this
    critical moment, the pedestrian decides to either continue crossing (potentially
    accelerating) or take a step back to clear the vehicle's lane, representing
    a last-second correction of their initial decision.
    """

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        """Initialize scenario-specific attributes."""
        super().__init__(world, config)
        self.scenario_id = "04_int"

        # Initialize controllers and paths to None
        self.path_controller_1: PathController | None = None
        self.path_controller_2: PathController | None = None
        self.path_controller_3: PathController | None = None
        self.path_controller_4: PathController | None = None
        self.turn_head: TurnHeadRightBehindNoICR | None = None
        self.look_behind_left: LookBehindLeft | None = None
        self.reset: ResetPose | None = None
        self.turn_head_second: TurnHeadRightBehindNoICR | None = None
        self.resetLD1: ResetPose | None = None
        self.resetLD2: ResetPose | None = None
        self.relaxer: Relaxer | None = None
        self.lean_forward: LeanForward | None = None
        self.curd_stat: InternalStateSetter | None = None
        self.starts_crossing: InternalStateSetter | None = None
        self.iss_crossed: InternalStateSetter | None = None

        # Scenario state and configuration attributes
        self.desc_p: carla.Location | None = None

    def get_spawn_details(
        self,
    ) -> tuple[str, list, tuple[float, float, float], tuple[float, float, float]]:
        """Return the spawn details for the ego vehicle and obstacles."""
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

        return self.scenario_id, obstacles, end, start

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        """Return the initial cognitive state of the walker."""
        return ICR.PLANNING_TO, SON.YIELDING if self.config.char == "yielding" else SON.FORCING

    def setup(self) -> None:
        """Set up the scenario, spawning actors and initializing controllers."""
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")
        obstacles = self.get_obstacle_blueprints()
        conf = self.config

        if self.player is None:
            raise PlayerNotAliveError(self.scenario_id)

        # --- Parameter extraction from config ---
        spawning_distance = conf.spawning_distance
        walking_distance = conf.walking_distance
        looking_distance1 = conf.looking_distance1
        looking_distance2 = conf.looking_distance2
        crossing_distanceX = conf.crossing_distanceX
        crossing_distanceY = conf.crossing_distanceY
        walk_back_distance = conf.walk_back_distance
        self.ped_speed = conf.ped_speed

        # Set decision box based on character
        self.db = [-1.0, 15.0] if conf.char == "yielding" else [-1.0, 20.0]
        if self.dummy_car:
            self.db = [-1.0, 20.0]

        # --- Spawn Walker ---
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        if not self.walker:
            logger.error(f"Failed to spawn walker for {self.get_scenario_id()}")
            return

        self.my_world.walker = self.walker
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        # --- Path Planning ---
        # Path 1: Walk to curb and turn slightly
        offsets_1 = [(0, walking_distance), (1, walking_distance + crossing_distanceY)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(255, 100, 0) if self.my_world.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # Path 2: Crossing the street to decision point
        offsets_2 = [(crossing_distanceX, walking_distance + crossing_distanceY)]
        path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(255, 0, 0) if self.my_world.debug else None,
        )
        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1
        self.path_controller_2 = PathController(
            self.world, self.walker, path_2, self.ped_speed * mult
        )

        # Path 3: Fully crossing the street after a positive decision
        offsets_3 = [
            (12.0, walking_distance + crossing_distanceY),
            (12.0, walking_distance + crossing_distanceY + 5.0),
        ]
        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(0, 0, 255) if self.my_world.debug else None,
        )
        self.path_controller_3 = PathController(self.world, self.walker, path_3, self.ped_speed)

        # Path 4: Taking a step back after a negative decision
        offsets_4 = [
            (
                crossing_distanceX - walk_back_distance - 0.2 * crossing_distanceX,
                walking_distance + crossing_distanceY,
            ),
        ]
        path_4 = self._compute_plans(
            offsets_4,
            base_loc,
            color=carla.Color(255, 100, 0) if self.my_world.debug else None,
        )
        self.path_controller_4 = PathController(self.world, self.walker, path_4, self.ped_speed)

        # --- Controller Initialization ---
        self.turn_head = TurnHeadRightBehindNoICR(self.walker, path_1[1])
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
        self.resetLD2 = ResetPose(self.walker, path_2[0])

        self.desc_p = path_2[0]  # The decision point is the end of the first crossing segment
        self.relaxer = Relaxer(self.walker, self.player, self.desc_p)
        self.lean_forward = LeanForward(self.walker, path_1[-1])

        # --- Cognitive State Initialization ---
        initial_icr, initial_son = self.get_initial_walker_state()
        self.walker.icr = initial_icr
        self.walker.son = initial_son
        self.walker.initial_son = initial_son
        self.iss_crossed = InternalStateSetter(self.walker, path_3[0], ICR.VERY_LOW, SON.AVERTING)
        self.starts_crossing = InternalStateSetter(
            self.walker,
            path_1[-1],
            ICR.GOING_TO,
            self.walker.initial_son,
        )
        self.curd_stat = InternalStateSetter(
            self.walker,
            path_1[0],
            ICR.PLANNING_TO,
            self.walker.initial_son,
        )

        # --- Debug Drawing ---
        if self.my_world.debug:
            self._draw_grid()
            self._draw_db()
            self._draw_point(reset_ld1_p, color=carla.Color(0, 255, 255))
            self._draw_point(second_turn_p, color=carla.Color(0, 255, 255))

        # --- Dummy Car Controller ---
        if self.dummy_car:
            if self.desc_p is None:
                logger.error("desc_p cannot be None at this point")
            else:
                player_loc = self.player.get_location()
                breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
                car_behave = conf.char == "forcing"
                car_to_desc = player_loc.y - self.desc_p.y - self.db[1]
                ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
                ped_time = ped_to_desc / self.ped_speed if self.ped_speed > 0 else float("inf")
                speed = car_to_desc / ped_time if ped_time != float("inf") else 8.0

                self.my_world.car_controller = CarController(
                    self.player,
                    breaking_point,
                    speed=speed - 2 if car_behave else speed - 1,
                    yielding=car_behave,
                )

    def tick(self) -> None:
        """Execute one simulation step for Scenario 04."""
        # Update references
        self.player = self.my_world.player
        self.walker = self.my_world.walker

        # Check if all required attributes are initialized
        if (
            self.walker is None
            or self.player is None
            or self.path_controller_1 is None
            or self.path_controller_2 is None
            or self.path_controller_3 is None
            or self.path_controller_4 is None
            or self.turn_head is None
            or self.look_behind_left is None
            or self.reset is None
            or self.turn_head_second is None
            or self.resetLD1 is None
            or self.resetLD2 is None
            or self.relaxer is None
            or self.lean_forward is None
            or self.curd_stat is None
            or self.starts_crossing is None
            or self.iss_crossed is None
            or self.desc_p is None
        ):
            raise ScenarioSetupError(self.scenario_id)

        status = self.path_controller_1.step()
        if self.dummy_car:
            self.my_world.car_controller.step()

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

        # Update world references at the end of the tick
        self.my_world.player = self.player
        self.my_world.walker = self.walker


class Scenario05Int(BaseScenario):
    """Implementation for Interactive Scenario 05.

    This scenario is designed to simulate a pedestrian who appears uncertain or
    confused, giving mixed signals to the driver. The pedestrian walks towards the
    curb and then begins to cross, but their movement across the road is interspersed
    with moments of looking back and forth and oscillating speed. This makes it
    challenging for a path prediction model to determine the pedestrian's true
    intent (to cross or to yield). The final decision to fully cross or stop is
    made at a decision point in the lane.
    """

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        """Initialize scenario-specific attributes."""
        super().__init__(world, config)
        self.scenario_id = "05_int"

        # Initialize controllers and paths to None
        self.path_controller_1: PathController | None = None
        self.path_controller_2: PathController | None = None
        self.path_controller_3: PathController | None = None
        self.uncertain: UncertainSteps | None = None
        self.turn_head: TurnHeadRightBehind | None = None
        self.look_behind_right: LookBehindRight | None = None
        self.look_behind_left: LookBehindLeft | None = None
        self.reset: ResetPose | None = None
        self.relaxer: Relaxer | None = None
        self.lean_forward: LeanForward | None = None
        self.iss_crossed: InternalStateSetter | None = None

        # Scenario state and configuration attributes
        self.path_2: list[carla.Location] = []
        self.desc_p: carla.Location | None = None

    def get_spawn_details(
        self,
    ) -> tuple[str, list, tuple[float, float, float], tuple[float, float, float]]:
        """Return the spawn details for the ego vehicle and obstacles."""
        start = (92.5, 300, -90)
        end = (92.5, 200, -90)
        obstacles = []
        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 85
        walker_spawn_point.location.y = 280
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 90.0
        walker = [walker_bp, walker_spawn_point]
        obstacles.append(walker)
        return self.scenario_id, obstacles, end, start

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        """Return the initial cognitive state of the walker."""
        return ICR.INTERESTED, SON.FORCING if self.config.char == "forcing" else SON.YIELDING

    def setup(self) -> None:
        """Set up the scenario, spawning actors and initializing controllers."""
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")
        obstacles = self.get_obstacle_blueprints()
        conf = self.config

        if self.player is None:
            raise PlayerNotAliveError(self.scenario_id)

        # --- Parameter extraction from config ---
        spawning_distance = conf.spawning_distance
        walking_distance_X = conf.walking_distance_X
        walking_distance_Y = conf.walking_distance_Y
        uncertain_steps = conf.uncertain_steps
        crossing_distance = conf.crossing_distance
        self.ped_speed = conf.ped_speed

        self.db = [-1.0, 15.0] if conf.char == "yielding" else [-1.0, 20.0]
        if self.dummy_car:
            self.db = [-1.0, 30.0] if conf.char == "yielding" else [-1.0, 20.0]
        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1

        # --- Spawn Walker ---
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0],
            carla.Transform(spawn_loc, obstacles[0][1].rotation),
        )
        if not self.walker:
            logger.error(f"Failed to spawn walker for {self.get_scenario_id()}")
            return
        self.my_world.walker = self.walker
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        # --- Path Planning ---
        # Path 1: Walk to the curb
        offsets_1 = [(walking_distance_X, -walking_distance_Y)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(255, 100, 0) if self.my_world.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # Path 2: Walk to the middle of the road (decision point)
        offsets_2 = [(walking_distance_X + crossing_distance, -walking_distance_Y)]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(255, 0, 0) if self.my_world.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed * mult,
        )

        # Path 3: Continue crossing road after decision
        offsets_3 = [(12, -walking_distance_Y), (12, 0)]
        path_3 = self._compute_plans(
            offsets_3,
            base_loc,
            color=carla.Color(0, 0, 255) if self.my_world.debug else None,
        )
        self.path_controller_3 = PathController(self.world, self.walker, path_3, self.ped_speed)

        # --- Controller Initialization ---
        delta = crossing_distance / (uncertain_steps + 1)
        uncertain_points = [
            self.get_point(((walking_distance_X + (i + 1) * delta), -walking_distance_Y))
            for i in range(uncertain_steps)
        ]
        self.uncertain = UncertainSteps(self.walker, uncertain_points, conf.char)

        turn_p = self.get_point((0, walking_distance_Y))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)
        self.look_behind_right = LookBehindRight(self.walker, path_1[0], conf.char)
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)

        self.desc_p = self.path_2[0]
        self.lean_forward = LeanForward(self.walker, self.desc_p)
        self.relaxer = Relaxer(self.walker, self.player, path_1[0])

        # Speed schedule for uncertain steps
        if conf.char == "forcing":
            self.path_controller_2.speed_schedule = [
                (uncertain_points[p], conf.ped_speed * 1.5 if p % 2 == 0 else conf.ped_speed * 1.0)
                for p in range(len(uncertain_points))
            ]
        else:
            self.path_controller_2.speed_schedule = [
                (uncertain_points[p], conf.ped_speed * 1.0 if p % 2 == 0 else conf.ped_speed * 0.5)
                for p in range(len(uncertain_points))
            ]

        # --- Cognitive State Initialization ---
        initial_icr, initial_son = self.get_initial_walker_state()
        self.walker.icr = initial_icr
        self.walker.son = initial_son
        self.walker.initial_son = initial_son
        self.iss_crossed = InternalStateSetter(self.walker, path_3[0], ICR.VERY_LOW, SON.AVERTING)

        # --- Debug Drawing ---
        if self.my_world.debug:
            self._draw_grid()
            self._draw_db()

        # --- Dummy Car Controller ---
        if self.dummy_car:
            if self.desc_p is None:
                logger.error("desc_p cannot be None at this point")
            else:
                player_loc = self.player.get_location()
                breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
                car_behave = conf.char == "forcing"
                car_to_desc = player_loc.y - self.desc_p.y - self.db[1]
                ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
                ped_time = ped_to_desc / self.ped_speed if self.ped_speed > 0 else float("inf")
                speed = car_to_desc / ped_time if ped_time != float("inf") else 8.0

                self.my_world.car_controller = CarController(
                    self.player,
                    breaking_point,
                    speed=speed - 1 if car_behave else speed - 0.5,
                    yielding=car_behave,
                )

    def tick(self) -> None:
        """Execute one simulation step for Scenario 05."""
        # Update references
        self.player = self.my_world.player
        self.walker = self.my_world.walker

        # Check if all required attributes are initialized
        if (
            self.walker is None
            or self.player is None
            or self.player is None
            or self.desc_p is None
            or self.path_controller_1 is None
            or self.path_controller_2 is None
            or self.path_controller_3 is None
            or self.uncertain is None
            or self.look_behind_right is None
            or self.turn_head is None
            or self.lean_forward is None
            or self.iss_crossed is None
            or self.desc_p is None
            or self.relaxer is None
        ):
            raise ScenarioSetupError(self.scenario_id)

        status = self.path_controller_1.step()
        self.uncertain.step()
        if self.dummy_car:
            self.my_world.car_controller.step()

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

        # Update world references at the end of the tick
        self.my_world.player = self.player
        self.my_world.walker = self.walker


class Scenario06Int(BaseScenario):
    """Implementation for Interactive Scenario 06.

    This scenario models a slower pedestrian (e.g., elderly) who intends to cross
    the street directly. When a conflict with the ego vehicle arises, the pedestrian
    decides to either continue crossing (if forcing) or to actively avoid the
    vehicle by walking around its front (if yielding). This requires the path
    prediction model to anticipate a significant deviation from the initial straight path.
    """

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        """Initialize scenario-specific attributes."""
        super().__init__(world, config)
        self.scenario_id = "06_int"

        # Initialize controllers and paths to None
        self.path_controller_1: PathController | None = None
        self.path_controller_2: PathController | None = None
        self.path_controller_3: PathController | None = None
        self.turn_head: TurnHeadRightBehind | None = None
        self.look_behind_right: LookBehindRight | None = None
        self.raise_arm: RaiseArm | None = None
        self.look_behind_left: LookBehindLeft | None = None
        self.reset: ResetPose | None = None
        self.relaxer: Relaxer | None = None
        self.iss_crossed: InternalStateSetter | None = None
        self.iss_crossed_2: InternalStateSetter | None = None

        # Scenario state and configuration attributes
        self.desc_p: carla.Location | None = None
        self.path_2: list[carla.Location] = []

    def get_spawn_details(
        self,
    ) -> tuple[str, list, tuple[float, float, float], tuple[float, float, float]]:
        """Return the spawn details for the ego vehicle and obstacles."""
        start = (92.5, 300, -90)
        end = (92.5, 200, -90)
        obstacles = []
        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        # Placeholder transform, as the actual spawn location is calculated in setup()
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 85
        walker_spawn_point.location.y = 300
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [walker_bp, walker_spawn_point]
        obstacles.append(walker)
        return self.scenario_id, obstacles, end, start

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        """Return the initial cognitive state of the walker."""
        return ICR.GOING_TO, SON.FORCING if self.config.char == "forcing" else SON.YIELDING

    def setup(self) -> None:
        """Set up the scenario, spawning actors and initializing controllers."""
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")
        obstacles = self.get_obstacle_blueprints()
        conf = self.config

        if self.player is None:
            raise PlayerNotAliveError(self.scenario_id)

        # --- Parameter extraction from config ---
        spawning_distance = conf.spawning_distance
        crossing_distance = conf.crossing_distance
        car_avoid_X = conf.car_avoid_X
        car_avoid_Y = conf.car_avoid_Y
        self.ped_speed = conf.ped_speed

        # Set decision box based on character
        self.db = [-1.0, 5.0] if conf.char == "yielding" else [-1.0, 10.0]
        if self.dummy_car:
            self.db = [-1.0, 5.0 + car_avoid_Y] if conf.char == "yielding" else [-1.0, 10.0]

        mult = 1.0 if conf.char == "yielding" else 1.1 * 1.1 * 1.1

        # --- Spawn Walker ---
        base_loc = obstacles[0][1].location + carla.Location(0, -spawning_distance, 0)
        spawn_loc = base_loc
        self.walker = self.world.try_spawn_actor(
            obstacles[0][0], carla.Transform(spawn_loc, obstacles[0][1].rotation)
        )
        if not self.walker:
            logger.error(f"Failed to spawn walker for {self.get_scenario_id()}")
            return
        self.my_world.walker = self.walker
        self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), self.ped_speed))
        self.world.tick()

        # --- Path Planning ---
        # Path 1: Walk to the middle of the road (decision point)
        offsets_1 = [(crossing_distance, 0.0)]
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(255, 0, 0) if self.my_world.debug else None,
        )
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)

        # Path 2: Continue straight across the road
        offsets_2 = [(12.0, 0.0), (12.0, 20.0)]
        self.path_2 = self._compute_plans(
            offsets_2,
            base_loc,
            color=carla.Color(0, 255, 0) if self.my_world.debug else None,
        )
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            self.path_2,
            self.ped_speed * mult,
        )

        # Path 3: Avoid the car by walking around it
        offsets_3 = [
            (crossing_distance + car_avoid_X, -car_avoid_Y),
            (12, -car_avoid_Y),
            (12, -car_avoid_Y + 20),
        ]
        self.path_controller_3 = PathController(
            self.world,
            self.walker,
            self._compute_plans(offsets_3, base_loc),
            self.ped_speed,
        )

        # --- Controller Initialization ---
        turn_p = self.get_point((0, 0))
        self.turn_head = TurnHeadRightBehind(self.walker, turn_p)
        self.look_behind_right = LookBehindRight(self.walker, path_1[0], conf.char)
        self.raise_arm = RaiseArm(
            self.walker,
            path_1[0],
            "forcing",
            self._get_p_from_vector(path_1[0], self.path_2[0], 0.5),
        )
        self.look_behind_left = LookBehindLeft(self.walker, mult=2)
        self.reset = ResetPose(self.walker)

        # Bug from original code: `vec` is a zero vector, so `desc_p` is `path_1[0]`
        vec = path_1[0] - path_1[0]
        self.desc_p = path_1[0] + 0.95 * vec

        if conf.char == "forcing":
            self.path_controller_2.speed_schedule = [
                (path_1[0] + per * path_1[0] - carla.Location(1, 0, 0), 10.9)
                for per in [0.0, 0.2, 0.4]
            ]
        elif conf.char == "yielding":
            self.path_controller_1.speed_schedule = [
                (path_1[0] - per * path_1[0] - carla.Location(1, 0, 0), 0.8)
                for per in [0.0, 0.2, 0.4]
            ]

        self.relaxer = Relaxer(self.walker, self.player, path_1[0] + 0.2 * vec)

        # --- Cognitive State Initialization ---
        initial_icr, initial_son = self.get_initial_walker_state()
        self.walker.icr = initial_icr
        self.walker.son = initial_son
        self.walker.initial_son = initial_son
        self.iss_crossed = InternalStateSetter(
            self.walker,
            offsets_3[1],
            ICR.VERY_LOW,
            SON.AVERTING,
        )
        self.iss_crossed_2 = InternalStateSetter(
            self.walker,
            self.path_2[0],
            ICR.VERY_LOW,
            SON.AVERTING,
        )

        # --- Debug Drawing ---
        if self.my_world.debug:
            self._draw_grid()
            self._draw_db()

        # --- Dummy Car Controller ---
        if self.dummy_car:
            if self.desc_p is None:
                logger.error("desc_p cannot be None at this point")
            else:
                player_loc = self.player.get_location()
                breaking_point = carla.Location(player_loc.x, self.desc_p.y + self.db[0], 0.5)
                car_behave = conf.char == "forcing"
                car_to_desc = player_loc.y - self.desc_p.y - self.db[1]
                ped_to_desc = l2_distance(self.walker.get_location(), self.desc_p)
                ped_time = ped_to_desc / self.ped_speed if self.ped_speed > 0 else float("inf")
                speed = car_to_desc / ped_time if ped_time != float("inf") else 8.0

                self.my_world.car_controller = CarController(
                    self.player,
                    breaking_point,
                    speed=speed - 1 if car_behave else speed - 0.5,
                    yielding=car_behave,
                )

    def tick(self) -> None:
        """Execute one simulation step for Scenario 06."""
        # Update references
        self.player = self.my_world.player
        self.walker = self.my_world.walker

        # Check if all required attributes are initialized
        if (
            self.walker is None
            or self.player is None
            or self.desc_p is None
            or self.player is None
            or self.path_controller_1 is None
            or self.path_controller_2 is None
            or self.path_controller_3 is None
            or self.turn_head is None
            or self.look_behind_right is None
            or self.raise_arm is None
            or self.iss_crossed is None
            or self.iss_crossed_2 is None
            or self.desc_p is None
            or self.reset is None
        ):
            raise ScenarioSetupError(self.scenario_id)

        status = self.path_controller_1.step()

        if self.dummy_car:
            self.my_world.car_controller.step()

        # self.look_behind_right.step()
        # self.turn_head.step()

        if status == "Done":
            if self.choice == "Avoid":
                self.reset.step()
                self.path_controller_3.step()
            elif self.choice == "Continue":
                status = self.raise_arm.step()
                logger.info(f"Called raise_arm, status is {status}")
                self.reset.step()
                self.path_controller_2.step()
            # if self.walker.initial_son == SON.FORCING:
            #     self.choice = "Continue"
            #     self.walker.icr = ICR.GOING_TO
            # elif self.walker.initial_son == SON.YIELDING:
            #     self.choice = "Avoid"
            #     self.walker.icr = ICR.PLANNING_TO

            elif l2_distance(self.walker.get_location(), self.desc_p) < 0.2:
                distance = y_distance(self.walker.get_location(), self.player.get_location()) - 2
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

        # Update world references at the end of the tick
        self.my_world.player = self.player
        self.my_world.walker = self.walker


class Scenario07Int(BaseScenario):
    """Interactive Scenario 07: Pedestrian interacts with car, then another pedestrian."""

    def __init__(self, world: "World", config: ControllerConfig) -> None:
        super().__init__(world, config)
        self.scenario_id = "07_int"

        # Specific attributes for this scenario
        self.curb_point: carla.Location | None = None
        self.sprint_multiplier: float = 1.5  # Default, will be set from config
        self.wait_duration: float = 2.0  # Default, will be set from config
        self.char: str = "yielding"  # Default, will be set from config

        self.look_across_street_left_controller: LookAcrossStreetLeft | None = None
        self.wave_hand_controller: SimplifiedWave | None = None
        self.lean_forward_controller: LeanForwardAndLook | None = None
        self.reset_pose_controller: ResetPose | None = None
        self.iss_at_curb: InternalStateSetter | None = None
        self.iss_sprinting: InternalStateSetter | None = None
        self.iss_meeting: InternalStateSetter | None = None

        self.walker2: carla.Actor | None = None
        self.main_ped_final_destination: carla.Location | None = None

        # State flags for multi-phase behavior
        self.at_curb: bool = False
        self.look_started: bool = False
        self.look_finished: bool = False
        self.wave_started: bool = False
        self.wave_total_duration: float = 2.0  # seconds, can be configured
        self.wave_phase_start_time: float | None = None  # Initialize
        self.wave_finished: bool = False
        self.lean_forward_started: bool = False
        self.lean_forward_finished: bool = False
        self.decision_phase_started: bool = False
        self.wait_start_time: float | None = None
        self.decided_action: str | None = None  # "Wait", "Sprint", "FinishedScenario"
        self.sprint_finished: bool = False
        self.meeting_started: bool = False

    def get_spawn_details(
        self,
    ) -> tuple[str, list, tuple[float, float, float], tuple[float, float, float]]:
        # These are placeholders; actual values come from config in setup.
        # For IConfig07, these need to match the general area of scenario05_int
        # Start and End for the ego vehicle
        start_coords = (92.5, 300.0, -90.0)  # Similar to IConfig05
        end_coords = (92.5, 200.0, -90.0)  # Similar to IConfig05
        obstacles = []

        # Main Pedestrian (walker)
        walker_bp_0001 = self.world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
        if walker_bp_0001.has_attribute("is_invincible"):
            walker_bp_0001.set_attribute("is_invincible", "false")

        # Spawn point for walker1 is determined dynamically in setup from base_loc
        # For now, provide a dummy transform for the blueprint list
        walker1_spawn_transform = carla.Transform(
            location=carla.Location(x=85.0, y=280.0, z=1.0),  # Placeholder
            rotation=carla.Rotation(yaw=90.0),  # Placeholder
        )
        obstacles.append([walker_bp_0001, walker1_spawn_transform])

        # Second Pedestrian (walker2) - its spawn is also dynamic in setup
        walker_bp_0002 = self.world.get_blueprint_library().filter("walker.pedestrian.0002")[0]
        if walker_bp_0002.has_attribute("is_invincible"):
            walker_bp_0002.set_attribute("is_invincible", "true")  # Often made invincible

        walker2_spawn_transform = carla.Transform(
            location=carla.Location(x=95.0, y=270.0, z=1.0),  # Placeholder
            rotation=carla.Rotation(yaw=270.0),  # Placeholder
        )
        obstacles.append([walker_bp_0002, walker2_spawn_transform])

        return self.scenario_id, obstacles, end_coords, start_coords

    def get_initial_walker_state(self) -> tuple[ICR, SON]:
        # This scenario's main walker starts interested
        return ICR.INTERESTED, SON.YIELDING if self.config.char == "yielding" else SON.FORCING

    def setup(self) -> None:
        logger.info(f"Setting up scenario: {self.get_scenario_id()}")

        # obstacle_blueprints contains [[bp1, transform1], [bp2, transform2]]
        # We need the blueprint for walker1 (index 0) and its intended base spawn transform
        obstacle_blueprints = self.get_obstacle_blueprints()
        conf = self.config

        if self.my_world.player is None:
            logger.error("Player actor is not set in the world. Cannot set up scenario.")
            return

        # --- Parameters from Config ---
        spawning_distance = conf.spawning_distance
        walking_distance_X = conf.walking_distance_X
        walking_distance_Y = conf.walking_distance_Y
        crossing_distance = conf.crossing_distance
        walk_after_crossing_X = conf.walk_after_crossing_X
        walk_after_crossing_Y = conf.walk_after_crossing_Y

        self.sprint_multiplier = conf.sprint_speed_multiplier
        self.wait_duration = conf.wait_duration
        self.char = conf.char  # Pedestrian's character
        self.ped_speed = conf.ped_speed

        self.db = [-1.0, 15.0] if self.char == "yielding" else [-1.0, 20.0]
        mult = 1.0  # Speed multiplier for initial crossing part, before sprint decision

        if self.my_world.debug:
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

        # --- Spawn Main Walker (walker1) ---
        # Base location calculation for walker1, similar to 05_int
        # obstacles[0][1] is the Transform for walker1 from get_spawn_details
        # (which is a placeholder)
        # We need a reliable way to get the *intended* starting area for the scenario.
        # Let's assume the scenario05_int reference point: x=85, y=280
        # And that spawning_distance is relative to the car's y.
        # The base_loc for path calculations should be consistent.
        # The actual spawn point of the walker is `base_loc` (adjusted for initial offset).
        scenario_reference_spawn_point = carla.Location(
            x=85.0,
            y=280.0,
            z=1.0,
        )  # Typical for 05_int like scenarios
        base_loc = scenario_reference_spawn_point + carla.Location(0, -spawning_distance, 0)
        spawn_loc_walker1 = base_loc  # Walker starts at this adjusted base_loc

        # Rotation for walker1, typically facing East (yaw=90) or based on config
        spawn_rotation_walker1 = carla.Rotation(yaw=90.0)  # Default to East
        if hasattr(conf, "initial_ped_yaw"):  # If specified in config
            spawn_rotation_walker1 = carla.Rotation(yaw=conf.initial_ped_yaw)

        self.walker = self.world.try_spawn_actor(
            obstacle_blueprints[0][0],  # Blueprint for walker1
            carla.Transform(spawn_loc_walker1, spawn_rotation_walker1),
        )
        if not self.walker:
            logger.error(f"Failed to spawn main walker (walker1) for {self.get_scenario_id()}")
            return
        self.my_world.walker = self.walker
        self.walker.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=self.ped_speed),
        )
        self.world.tick()  # Settle walker

        # --- Path 1: Walk Towards Curb ---
        # Offsets are relative to base_loc (which is spawn_loc_walker1 for this path)
        offsets_1 = [(walking_distance_X, -walking_distance_Y)]  # Walk towards curb
        path_1 = self._compute_plans(
            offsets_1,
            base_loc,
            color=carla.Color(255, 100, 0) if self.my_world.debug else None,
        )
        if not path_1:
            logger.error("Scenario 07_int: Path 1 calculation failed.")
            return
        self.path_controller_1 = PathController(self.world, self.walker, path_1, self.ped_speed)
        self.curb_point = path_1[-1]

        # --- Path 2: Cross Street (target point before continuing) ---
        # Target point after crossing_distance from curb_point
        target_across_loc = self.curb_point + carla.Location(float(crossing_distance), 0, 0)

        # Path 2 is just a single segment to this target_across_loc
        path_2_single_target = [target_across_loc]
        self.path_controller_2 = PathController(
            self.world,
            self.walker,
            path_2_single_target,
            self.ped_speed * mult,
        )  # Initial speed

        # --- Path 3: Continue after crossing/sprint to meet walker2 ---
        # Path 3 starts from target_across_loc
        # It goes further in +X by walk_after_crossing_X, then in -Y (towards walker2) by
        # walk_after_crossing_Y
        # Offsets for path_3 should be relative to target_across_loc if _compute_plans is used
        # with it as base or calculate absolute points from the original base_loc
        abs_path3_pt1 = target_across_loc + carla.Location(walk_after_crossing_X, 0, 0)
        abs_path3_pt2 = abs_path3_pt1 + carla.Location(
            0,
            -walk_after_crossing_Y,
            0,
        )  # -Y for "down" on map
        path_3 = [abs_path3_pt1, abs_path3_pt2]
        self.path_controller_3 = PathController(
            self.world,
            self.walker,
            path_3,
            self.ped_speed,
        )  # Normal speed after sprint

        self.main_ped_final_destination = path_3[-1]  # Final point for walker1

        # --- Spawn Second Pedestrian (walker2) ---
        ped2_blueprint = obstacle_blueprints[1][0]  # Blueprint for walker2

        # Spawn near where walker1 will end
        spawn_loc_ped2 = self.main_ped_final_destination + carla.Location(0, 0, 0)
        spawn_rotation_ped2 = carla.Rotation(yaw=270.0)  # Facing West, towards walker1's approach

        self.walker2 = self.world.try_spawn_actor(
            ped2_blueprint,
            carla.Transform(spawn_loc_ped2, spawn_rotation_ped2),
        )
        if self.walker2:
            self.walker2.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0),
            )
            logger.info(f"Spawned second pedestrian (walker2) at {spawn_loc_ped2}")
        else:
            logger.error(f"Failed to spawn second pedestrian at {spawn_loc_ped2}")

        # --- Initialize Controllers & State Flags ---
        self.look_across_street_left_controller = LookAcrossStreetLeft(
            self.walker,
            self.curb_point,
            duration=1.5,
        )
        self.wave_hand_controller = SimplifiedWave(self.walker, debug=self.my_world.debug)
        self.lean_forward_controller = LeanForwardAndLook(
            self.walker,
            start_pos=self.curb_point,  # Triggered at curb
            look_to="custom",
            head_turn_neck_pitch=0,
            head_turn_head_pitch=2,
            head_turn_head_roll=100,
        )
        self.reset_pose_controller = ResetPose(self.walker)

        # Initialize state flags
        self.at_curb = False
        self.look_started = False
        self.look_finished = False
        self.wave_started = False
        self.wave_total_duration = 2.0
        self.wave_finished = False
        self.lean_forward_started = False
        self.lean_forward_finished = False
        self.decision_phase_started = False
        self.wait_start_time = None
        self.decided_action = None
        self.sprint_finished = False
        self.meeting_started = False
        self.wave_phase_start_time = None

        # Debug Drawing
        if self.my_world.debug:
            self._draw_point(
                spawn_loc_walker1,
                color=carla.Color(0, 255, 255),  # Cyan for walker1 spawn
            )
            self._draw_point(
                self.curb_point,
                color=carla.Color(255, 255, 0),  # Yellow for curb point
            )
            self._draw_point(
                target_across_loc,
                color=carla.Color(255, 0, 255),  # Magenta for across street
            )
            if path_3:
                self._draw_point(
                    path_3[-1],
                    color=carla.Color(0, 255, 0),  # Green for walker1 end
                )
            if self.walker2:
                self._draw_point(
                    spawn_loc_ped2,
                    color=carla.Color(255, 165, 0),  # Orange for walker2 spawn
                )
            self.desc_p = self.curb_point  # Decision point for DB drawing
            if self.db:
                self._draw_db(self.db, color=carla.Color(255, 0, 255))

        # Initial Cognitive State for main walker
        initial_icr, initial_son = self.get_initial_walker_state()
        self.walker.icr = initial_icr
        self.walker.son = initial_son
        self.walker.initial_son = initial_son  # Store for reference

        # InternalStateSetters
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
        )  # Sprinting is forcing
        if self.walker2:
            self.iss_meeting = InternalStateSetter(
                self.walker,
                self.main_ped_final_destination,
                ICR.VERY_LOW,
                SON.AVERTING,
            )
        else:
            self.iss_meeting = None

        # --- Setup Dummy Car Controller ---
        if self.dummy_car:
            player_loc = self.my_world.player.get_location()
            # Car yields if pedestrian is "forcing", car forces if pedestrian is "yielding"
            car_should_yield = self.char == "forcing"
            estimated_car_speed_mps = 6.0  # Default/average speed for dummy car

            if self.curb_point is None:
                logger.error("Curb point is None, cannot set up car controller.")
                return

            # Brake 5m before ped's decision box start
            braking_y_coord = self.curb_point.y + self.db[0] - 5.0
            braking_point_car = carla.Location(player_loc.x, braking_y_coord, 0.5)

            self.car_controller = CarController(
                self.my_world.player,
                braking_point=braking_point_car if car_should_yield else None,
                speed=estimated_car_speed_mps,
                yielding=car_should_yield,
            )
            self.my_world.car_controller = self.car_controller
            logger.debug(
                f"Scenario 07_int: Car controller. Yielding: {car_should_yield}. "
                f"Speed: {estimated_car_speed_mps:.1f} m/s. "
                f"Braking Y if yielding: {braking_y_coord if car_should_yield else 'N/A'}",
            )

    def tick(self) -> None:
        # Ensure the main walker and its primary path controller are initialized.
        if (
            not self.walker
            or not self.my_world.player
            or not self.path_controller_1
            or not self.look_across_street_left_controller
            or not self.wave_hand_controller
            or not self.lean_forward_controller
        ):
            return

        current_time = time.time()  # Get current time for timed behaviors

        # --- Phase 1: Walk to curb ---
        # Pedestrian walks along path_1 towards the curb_point.
        if not self.at_curb:
            status1 = self.path_controller_1.step()
            if status1 == "Done":
                self.at_curb = True  # Mark arrival at curb
                # Stop the walker momentarily at the curb.
                self.walker.apply_control(
                    carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0),
                )
                logger.info("Scenario 07: Reached curb point.")
                # Update cognitive state: now planning to interact/cross.
                if self.iss_at_curb:
                    self.iss_at_curb.step()

        # --- Phase 2: Look across street (if at curb) ---
        # After reaching the curb, the pedestrian performs a "look across street" animation.
        elif self.at_curb and not self.look_finished:
            if not self.look_started:
                # Ensure the look animation starts from the walker's current position at the curb.
                self.look_across_street_left_controller.start_pos = self.walker.get_location()
                self.look_across_street_left_controller.step()  # Initiate the look
                self.look_started = True
                logger.info("Scenario 07: Starting 'look across street' pose blend.")

            # Continue stepping the look controller until it's done.
            look_status = self.look_across_street_left_controller.step()
            if look_status == "Done":
                self.look_finished = True  # Mark look as finished. The pose might be held.
                logger.info("Scenario 07: 'Look across street' pose finished and held.")

        # --- Phase 3: Wave Hand (after looking) ---
        # After looking, the pedestrian waves their hand.
        elif self.look_finished and not self.wave_finished:
            if not self.wave_started:
                # Reset any prior pose (like the "look") before starting to wave.
                if self.reset_pose_controller:
                    self.reset_pose_controller.step()  # reset current pose
                if self.reset_pose_controller:
                    self.reset_pose_controller.done = False  # allow reset_pose to be used again

                self.wave_hand_controller.start_waving()  # Initiate waving animation.
                self.wave_started = True
                self.wave_phase_start_time = current_time  # Record start time for waving duration.
                logger.info("Scenario 07: Starting simplified waving animation.")

            # Continue waving for the defined `wave_total_duration`.
            if self.wave_phase_start_time and (
                current_time - self.wave_phase_start_time < self.wave_total_duration
            ):
                self.wave_hand_controller.step()  # Step through the waving animation.
                # A small sleep can make the wave more visually discernible if steps are too fast.

                # TODO: time.sleep blocks CARLA's synchronous mode tick. Avoid if possible.
                time.sleep(0.2)
            else:  # Waving duration ended.
                self.wave_hand_controller.stop_waving()  # Stop the waving animation.
                self.wave_finished = True
                logger.info("Scenario 07: Simplified waving duration ended.")

        # --- Phase 4: Lean Forward (after waving) ---
        # After waving, the pedestrian leans forward, signaling intent to move.
        elif self.wave_finished and not self.lean_forward_finished:
            if not self.lean_forward_started:
                # Reset the waving pose before leaning forward.
                if self.reset_pose_controller:
                    self.reset_pose_controller.step()
                if self.reset_pose_controller:
                    self.reset_pose_controller.done = False

                # The LeanForward controller starts from walker's current position (curb_point).
                self.lean_forward_controller.start_pos = self.walker.get_location()
                self.lean_forward_controller.step()  # Initiate leaning.
                self.lean_forward_started = True
                logger.info("Scenario 07: Starting 'lean forward' pose.")

            # Continue stepping the lean controller.
            lean_status = self.lean_forward_controller.step()
            if lean_status == "Done":
                self.lean_forward_finished = True  # Mark leaning as finished.
                logger.info("Scenario 07: 'Lean forward' pose applied.")

        # --- Phase 5: Make Decision & Execute (after leaning) ---
        # After all preparatory actions, the pedestrian decides whether to "Wait" or "Sprint".
        elif self.lean_forward_finished and self.decided_action is None:
            # Reset the leaning pose before making the sprint/wait decision.
            if self.reset_pose_controller:
                self.reset_pose_controller.step()
            if self.reset_pose_controller:
                self.reset_pose_controller.done = False

            self.decision_phase_started = True
            self.wait_start_time = current_time  # Record time for potential waiting period.
            logger.info("Scenario 07: Leaning finished. Moving to decision (Wait/Sprint).")

            # Decision logic based on car's position relative to pedestrian's decision box (db).
            distance_to_car = y_distance(self.curb_point, self.my_world.player.get_location()) - 2
            car_in_db = self.decision_trigger(distance_to_car, self.db)

            if self.char == "yielding":
                # Yielding pedestrian: if car is close (in_db), decide to Wait. Otherwise, Sprint.
                if car_in_db:
                    self.decided_action = "Wait"
                    self.walker.icr = ICR.PLANNING_TO  # Still planning, but will yield.
                    self.walker.son = SON.YIELDING
                    logger.info("Scenario 07: Decided to Wait (Yielding Pedestrian, Car Close).")
                else:
                    self.decided_action = "Sprint"
                    logger.info("Scenario 07: Decided to Sprint (Yielding Pedestrian, Car Far).")
            elif self.char == "forcing":
                # Forcing pedestrian: always decides to Sprint.
                self.decided_action = "Sprint"
                logger.info("Scenario 07: Decided to Sprint (Forcing Pedestrian).")

            # If "Sprint" is chosen, set sprint speed and start PathController_2.
            if self.decided_action == "Sprint":
                sprint_speed = self.ped_speed * self.sprint_multiplier

                # Update speed for PathController_2
                self.path_controller_2.cur_speed = sprint_speed
                logger.info(f"Scenario 07: Setting sprint speed: {sprint_speed:.2f} m/s")
                if self.iss_sprinting:
                    self.iss_sprinting.step()  # Update cognitive state for sprinting.
                self.path_controller_2.step()  # Start moving along path_2 (sprint path).
            elif self.decided_action == "Wait":
                # If "Wait" is chosen, stop the walker.
                self.walker.apply_control(
                    carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0),
                )

        # --- Phase 6: Continue Executing Action (Wait or Sprint) ---
        elif self.decided_action == "Wait":
            # If waiting, remain stopped. Check if waiting duration is over or car has passed.
            self.walker.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0),
            )
            distance_to_car = y_distance(self.curb_point, self.my_world.player.get_location()) - 2
            car_in_db = self.decision_trigger(distance_to_car, self.db)

            # Handle None case for wait_start_time
            time_elapsed_decision = current_time - (self.wait_start_time or current_time)

            if not car_in_db or time_elapsed_decision >= self.wait_duration:
                # If car is no longer in decision box OR wait duration is over, switch to Sprint.
                self.decided_action = "Sprint"
                sprint_speed = self.ped_speed * self.sprint_multiplier
                self.path_controller_2.cur_speed = sprint_speed
                logger.info("Scenario 07: Wait finished or car passed, switching to sprint.")
                if self.iss_sprinting:
                    self.iss_sprinting.step()
                self.path_controller_2.step()

        elif self.decided_action == "Sprint" and not self.sprint_finished:
            # If sprinting, continue along path_2.
            status2 = self.path_controller_2.step()
            if status2 == "Done":
                self.sprint_finished = True  # Mark sprint as finished.
                logger.info("Scenario 07: Finished crossing (sprint path segment).")
                if self.path_controller_3:
                    # If there's a path_3 (to meet walker2), start it.
                    self.meeting_started = True
                    self.path_controller_3.cur_speed = self.ped_speed  # Resume normal speed.
                    self.path_controller_3.step()
                else:
                    # If no path_3, the scenario effectively ends for the main walker here.
                    self.decided_action = "FinishedScenario"
                    self.walker.apply_control(
                        carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0),
                    )
                # Update cognitive state for meeting phase or end of crossing.
                if hasattr(self, "iss_meeting") and self.iss_meeting:
                    self.iss_meeting.step()

        # --- Phase 7: Walk to meet walker2 (if applicable) ---
        elif self.meeting_started and self.decided_action != "FinishedScenario":
            if self.path_controller_3:
                status3 = self.path_controller_3.step()
                if status3 == "Done":
                    self.decided_action = "FinishedScenario"  # Mark scenario as finished.
                    logger.info("Scenario 07: Reached meeting point with walker2.")
                    self.walker.apply_control(
                        carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0),
                    )
            else:  # Should not happen if meeting_started is true, but as a fallback.
                self.decided_action = "FinishedScenario"

        # --- Dummy Car Controller Update ---
        # Advance the dummy car's behavior if it's enabled and controller exists.
        if self.dummy_car and self.car_controller:
            self.car_controller.step()


# Create a mapping for easy lookup
SCENARIO_MAP: dict[str, type[BaseScenario]] = {
    "01_int": Scenario01Int,
    "02_int": Scenario02Int,
    "03_int": Scenario03Int,
    "04_int": Scenario04Int,
    "05_int": Scenario05Int,
    "06_int": Scenario06Int,
    "07_int": Scenario07Int,
    # "01_non_int": Scenario01NonInt,
    # "02_non_int": Scenario02NonInt,
    # "03_non_int": Scenario03NonInt,
    # "04_non_int": Scenario04NonInt,
    # "05_non_int": Scenario05NonInt,
    # "06_non_int": Scenario06NonInt,
}
