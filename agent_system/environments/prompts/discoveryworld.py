"""Compatibility exports for the Chemistry prompt's historical import path."""

from agent_system.environments.env_package.discovery.chemistry.prompts import (
    DISCOVERYWORLD_TEMPLATE,
    DISCOVERYWORLD_TEMPLATE_NO_HIS,
    format_current_chemicals,
    format_empty_chemical_belief,
    format_key_status,
)

__all__ = [
    "DISCOVERYWORLD_TEMPLATE",
    "DISCOVERYWORLD_TEMPLATE_NO_HIS",
    "format_current_chemicals",
    "format_empty_chemical_belief",
    "format_key_status",
]
