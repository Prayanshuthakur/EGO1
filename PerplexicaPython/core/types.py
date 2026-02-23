"""
Core type definitions for the application.
Uses Pydantic for runtime type validation and serialization.
"""

from typing import Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# Message Types
# ============================================================================

class MessageRole(str, Enum):
    """Message role in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool responses


class ChatTurnMessage(BaseModel):
    """Simplified message format for chat history."""
    role: Literal["user", "assistant"]
    content: str


# ============================================================================
# Block Types (Response Components)
# ============================================================================

class BlockType(str, Enum):
    """Type of response block."""
    TEXT = "text"
    SOURCE = "source"
    RESEARCH = "research"
    WIDGET = "widget"


class TextBlock(BaseModel):
    """Text content block (streaming answer)."""
    id: str
    type: Literal[BlockType.TEXT] = BlockType.TEXT
    data: str  # The actual text content


class SearchResultMetadata(BaseModel):
    """Metadata for a search result."""
    title: str
    url: str
    snippet: str | None = None
    author: str | None = None
    published_date: str | None = None


class SearchResult(BaseModel):
    """A single search result with content."""
    metadata: SearchResultMetadata
    content: str  # Full extracted content


class SourceBlock(BaseModel):
    """Source citations block."""
    id: str
    type: Literal[BlockType.SOURCE] = BlockType.SOURCE
    data: list[SearchResult]


class ReasoningResearchBlock(BaseModel):
    """Research reasoning step."""
    id: str
    type: Literal["reasoning"] = "reasoning"
    reasoning: str


class ToolCallResearchBlock(BaseModel):
    """Research tool call step."""
    id: str
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_args: dict[str, Any]
    result: Any


class ResearchBlock(BaseModel):
    """Research process block with sub-steps."""
    id: str
    type: Literal[BlockType.RESEARCH] = BlockType.RESEARCH
    data: dict[str, Any] = Field(default_factory=lambda: {"subSteps": []})


class WidgetType(str, Enum):
    """Type of widget."""
    WEATHER = "weather"
    STOCK = "stock"
    CALCULATION = "calculation"


class WidgetBlock(BaseModel):
    """Widget output block."""
    id: str
    type: Literal[BlockType.WIDGET] = BlockType.WIDGET
    data: dict[str, Any]  # Contains widgetType and params


# Union type for all blocks
Block = TextBlock | SourceBlock | ResearchBlock | WidgetBlock


# ============================================================================
# Classification Types
# ============================================================================

class QueryClassification(BaseModel):
    """Classification of a user query."""
    skip_search: bool = Field(
        default=False,
        description="Whether to skip web search"
    )
    personal_search: bool = Field(
        default=False,
        description="Whether to search personal documents"
    )
    academic_search: bool = Field(
        default=False,
        description="Whether to perform academic search"
    )
    discussion_search: bool = Field(
        default=False,
        description="Whether to search discussions/forums"
    )
    show_weather_widget: bool = Field(
        default=False,
        description="Whether to show weather widget"
    )
    show_stock_widget: bool = Field(
        default=False,
        description="Whether to show stock widget"
    )
    show_calculation_widget: bool = Field(
        default=False,
        description="Whether to show calculation widget"
    )


class ClassificationResult(BaseModel):
    """Complete classification result."""
    classification: QueryClassification
    standalone_followup: str = Field(
        description="Context-independent reformulation of the query"
    )


# ============================================================================
# Search Types
# ============================================================================

class SearchSource(str, Enum):
    """Available search sources."""
    WEB = "web"
    ACADEMIC = "academic"
    DISCUSSION = "discussion"
    IMAGES = "images"
    VIDEOS = "videos"


class OptimizationMode(str, Enum):
    """Research optimization mode."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


# ============================================================================
# Agent Types
# ============================================================================

class ToolCall(BaseModel):
    """LLM tool/function call."""
    id: str
    name: str
    arguments: dict[str, Any]


class ActionOutput(BaseModel):
    """Output from an agent action."""
    type: str
    results: list[SearchResult] = Field(default_factory=list)
    data: Any | None = None


class ResearcherOutput(BaseModel):
    """Output from the researcher agent."""
    findings: list[ActionOutput]
    search_findings: list[SearchResult]


# ============================================================================
# Widget Types
# ============================================================================

class WeatherData(BaseModel):
    """Weather widget data."""
    location: str
    temperature: float
    feels_like: float
    humidity: int
    wind_speed: float
    description: str
    icon: str | None = None


class StockData(BaseModel):
    """Stock widget data."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int | None = None
    market_cap: float | None = None


class CalculationResult(BaseModel):
    """Calculation widget result."""
    expression: str
    result: str
    steps: list[str] | None = None


# ============================================================================
# Session Types
# ============================================================================

class SessionEvent(BaseModel):
    """Event emitted by a session."""
    event: Literal["data", "end", "error"]
    data: Any | None = None


# ============================================================================
# Database Types
# ============================================================================

class ChatMetadata(BaseModel):
    """Chat session metadata."""
    id: str
    title: str
    created_at: datetime
    sources: list[str] = Field(default_factory=list)
    files: list[dict[str, str]] = Field(default_factory=list)


class MessageStatus(str, Enum):
    """Message processing status."""
    ANSWERING = "answering"
    COMPLETED = "completed"
    ERROR = "error"


class MessageMetadata(BaseModel):
    """Message metadata."""
    id: int
    message_id: str
    chat_id: str
    backend_id: str
    query: str
    created_at: datetime
    response_blocks: list[dict[str, Any]] = Field(default_factory=list)
    status: MessageStatus = MessageStatus.ANSWERING
