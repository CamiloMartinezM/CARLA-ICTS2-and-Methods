"""Custom exceptions for actor lifecycle management in CARLA scenarios."""


class ActorNotAliveError(Exception):
    """Base exception for when an expected actor is not alive or valid."""

    def __init__(
        self,
        actor_name: str,
        scenario_id: str | None = None,
        message: str | None = None,
    ) -> None:
        """Set up an ActorNotAliveError.

        Args:
            actor_name (str): Name of the actor that is not alive or valid.
            scenario_id (str | None): ID of the scenario in which the actor is expected.
            message (str | None): Custom error message. If None, a default message is used.
        """
        self.actor_name = actor_name
        self.scenario_id = scenario_id
        if message is None:
            message = f"Actor '{actor_name}' is not alive or valid"
            if scenario_id:
                message += f" in scenario '{scenario_id}'."
            else:
                message += "."
        super().__init__(message)


class WalkerNotAliveError(ActorNotAliveError):
    """Custom exception raised when the primary walker actor is not alive or valid."""

    def __init__(self, scenario_id: str | None = None, message: str | None = None) -> None:
        super().__init__("Walker", scenario_id, message)


class PlayerNotAliveError(ActorNotAliveError):
    """Custom exception raised when the player (ego vehicle) actor is not alive or valid."""

    def __init__(self, scenario_id: str | None = None, message: str | None = None) -> None:
        super().__init__("Player", scenario_id, message)
