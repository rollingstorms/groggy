from typing import List, Optional

from .types import EdgeId, NodeId


class GroggyError(Exception):
    """Base exception for all Groggy errors"""

    pass


class NodeNotFoundError(GroggyError):
    """Raised when a node is not found"""

    def __init__(self, node_id: NodeId, operation: str, suggestion: str = ""):
        self.node_id = node_id
        self.operation = operation
        self.suggestion = suggestion
        super().__init__(f"Node {node_id} not found during {operation}. {suggestion}")


class EdgeNotFoundError(GroggyError):
    """Raised when an edge is not found"""

    def __init__(self, edge_id: EdgeId, operation: str, suggestion: str = ""):
        self.edge_id = edge_id
        self.operation = operation
        self.suggestion = suggestion
        super().__init__(f"Edge {edge_id} not found during {operation}. {suggestion}")


class InvalidInputError(GroggyError):
    """Raised for invalid input parameters"""

    pass


class NotImplementedError(GroggyError):
    """Raised for features not yet implemented"""

    def __init__(self, feature: str, tracking_issue: Optional[str] = None):
        self.feature = feature
        self.tracking_issue = tracking_issue
        message = f"Feature '{feature}' is not yet implemented"
        if tracking_issue:
            message += f". See: {tracking_issue}"
        super().__init__(message)


class ValidationError(GroggyError):
    """Raised when pipeline validation fails"""

    def __init__(self, errors: List[str], warnings: List[str] = None):
        self.errors = errors
        self.warnings = warnings or []

        # Format error message
        message = f"Pipeline validation failed with {len(errors)} error(s)"
        if self.warnings:
            message += f" and {len(self.warnings)} warning(s)"
        message += ":\n\n"

        for i, err in enumerate(errors, 1):
            message += f"  {i}. {err}\n"

        if self.warnings:
            message += "\nWarnings:\n"
            for i, warn in enumerate(self.warnings, 1):
                message += f"  {i}. {warn}\n"

        super().__init__(message)
