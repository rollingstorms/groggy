from typing import Optional
from .types import NodeId, EdgeId

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
