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


class ScenarioSetupError(Exception):
    """Raised when a scenario's `tick()` is called before it's properly set up.

    A scenario is set up by calling its `setup()` method.
    """

    def __init__(self, scenario_id: str, missing_attrs: list[str] | None = None) -> None:
        """Initialize the ScenarioSetupError."""
        message = f"Scenario '{scenario_id}' tick failed. "
        if missing_attrs:
            message += f"Missing attributes: {', '.join(missing_attrs)}"
        super().__init__(message)
