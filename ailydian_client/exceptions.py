"""
AILYDIAN Backend Client Exceptions
"""


class AILYDIANError(Exception):
    """Base exception for AILYDIAN client"""
    pass


class AuthenticationError(AILYDIANError):
    """Authentication failed"""
    pass


class RateLimitError(AILYDIANError):
    """Rate limit exceeded"""
    def __init__(self, message, current_usage=None, limit=None):
        super().__init__(message)
        self.current_usage = current_usage
        self.limit = limit


class QuotaExceededError(AILYDIANError):
    """Resource quota exceeded"""
    def __init__(self, message, resource_type=None, current_usage=None, quota=None):
        super().__init__(message)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.quota = quota


class ProjectNotFoundError(AILYDIANError):
    """Project not registered"""
    pass


class DashboardAccessError(AILYDIANError):
    """Attempted to access forbidden dashboard resources"""
    pass
