"""Constructs the run command for the CARLA server."""


def run_server_command(config: dict) -> str:
    """Return the command to run CARLA.

    The `config` dictionary comes from the YAML configuration file in the project root.
    """
    carla = config["carla"]
    port = f"-carla-port={carla['port']}"
    carla_p = config["carla"]["path"]

    cmd = f"cd {carla_p} && "

    if carla["set_variables"]:
        cmd += " ".join(carla["set_variables"])

    cmd += " ./CarlaUE4.sh"

    if carla["quality"]:
        cmd += f" -quality-level={carla['quality']}"

    if carla["carla_server"]:
        cmd += " -carla-server"

    if carla["benchmark"]:
        cmd += " -benchmark"

    if carla["prevernvidia"]:
        cmd += " -prevernvidia"

    if carla["fps"]:
        cmd += f" -fps={carla['fps']}"

    cmd += f" {port}"

    return cmd
