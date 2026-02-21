"""
AILYDIAN Backend Client Library
Official Python client for accessing AILYDIAN-AGENT-ORCHESTRATOR backend services
"""

from .client import AILYDIANBackend
from .exceptions import (
    AILYDIANError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ProjectNotFoundError
)

__version__ = "1.0.0"
__all__ = [
    'AILYDIANBackend',
    'AILYDIANError',
    'AuthenticationError',
    'RateLimitError',
    'QuotaExceededError',
    'ProjectNotFoundError'
]
