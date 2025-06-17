import logging
import os

from pyhocon import ConfigFactory, ConfigTree

logger = logging.getLogger(__name__)


def parse_config(file: str) -> ConfigTree:
    return ConfigFactory.parse_file(file)  # type:ignore


def get_config(config_name: str, create_dir: bool = True, config_file: str = ""):
    config: dict = parse_config(config_file)[config_name]  # type:ignore

    if create_dir:
        os.makedirs(config["log_root"], exist_ok=True)

    return config
