import math

import carla


class CarController:
    def __init__(
        self,
        player_car,
        braking_point,
        speed,
        yielding=False,
    ):
        self.player_car = player_car
        self.braking_point = braking_point
        self.choice = yielding
        self.player_car.enable_constant_velocity(carla.Vector3D(speed, 0, 0))
        self.speed = speed

    def step(self):
        actor_location = self.player_car.get_location()

        if self.braking_point and self.choice:
            location = self.braking_point
            direction = location - actor_location
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)

            deceleration = math.tanh((direction_norm**3) / 30_000)
            self.player_car.enable_constant_velocity(
                carla.Vector3D(self.speed * deceleration, 0, 0),
            )

        return "Running"
