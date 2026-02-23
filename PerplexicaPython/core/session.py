"""
Session management for real-time streaming and state management.
Handles event emission and block storage during agent execution.
"""

from typing import Any, Callable, Literal
from core.types import Block, TextBlock, SourceBlock, ResearchBlock, WidgetBlock
from core.utils import generate_id
import json


class SessionManager:
    """
    Manages session state and event streaming.
    Stores blocks and emits events for real-time updates.
    """
    
    def __init__(self, session_id: str | None = None):
        """
        Initialize session manager.
        
        Args:
            session_id: Optional session ID (generated if not provided)
        """
        self.id = session_id or generate_id()
        self._blocks: dict[str, Block] = {}
        self._listeners: list[Callable] = []
    
    def subscribe(self, callback: Callable[[str, Any], None]) -> Callable[[], None]:
        """
        Subscribe to session events.
        
        Args:
            callback: Function to call on events (event_name, data)
            
        Returns:
            Unsubscribe function
        """
        self._listeners.append(callback)
        
        def unsubscribe():
            if callback in self._listeners:
                self._listeners.remove(callback)
        
        return unsubscribe
    
    def emit(self, event: Literal["data", "end", "error"], data: Any = None):
        """
        Emit an event to all listeners.
        
        Args:
            event: Event name
            data: Event data
        """
        for listener in self._listeners:
            try:
                listener(event, data)
            except Exception as e:
                print(f"Error in session listener: {e}")
    
    def emit_block(self, block: Block):
        """
        Emit a new block and store it.
        
        Args:
            block: Block to emit
        """
        # Convert Pydantic model to dict for storage
        block_dict = block.model_dump() if hasattr(block, 'model_dump') else block
        self._blocks[block_dict['id']] = block_dict
        
        self.emit("data", {
            "type": "block",
            "block": block_dict
        })
    
    def update_block(self, block_id: str, patches: list[dict[str, Any]]):
        """
        Update an existing block using JSON patches.
        
        Args:
            block_id: ID of block to update
            patches: List of JSON patch operations
        """
        if block_id not in self._blocks:
            return
        
        # Apply patches (simplified - in production use jsonpatch library)
        for patch in patches:
            if patch["op"] == "replace":
                path = patch["path"]
                value = patch["value"]
                
                # Simple path parsing (e.g., "/data" -> ["data"])
                keys = [k for k in path.split("/") if k]
                
                # Navigate to the target
                target = self._blocks[block_id]
                for key in keys[:-1]:
                    target = target[key]
                
                # Set the value
                target[keys[-1]] = value
        
        self.emit("data", {
            "type": "updateBlock",
            "blockId": block_id,
            "patch": patches
        })
    
    def get_block(self, block_id: str) -> dict[str, Any] | None:
        """
        Get a block by ID.
        
        Args:
            block_id: Block ID
            
        Returns:
            Block data or None
        """
        return self._blocks.get(block_id)
    
    def get_all_blocks(self) -> list[dict[str, Any]]:
        """Get all blocks in order of creation."""
        return list(self._blocks.values())
    
    def remove_all_listeners(self):
        """Remove all event listeners."""
        self._listeners.clear()
    
    def clear_blocks(self):
        """Clear all stored blocks."""
        self._blocks.clear()
    
    @classmethod
    def create_session(cls, session_id: str | None = None) -> "SessionManager":
        """
        Factory method to create a new session.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            New SessionManager instance
        """
        return cls(session_id=session_id)
