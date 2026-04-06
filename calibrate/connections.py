# calibrate.connections module
"""
Connection types for injecting external agents into calibrate evaluations.

Usage:
    from calibrate.connections import TextAgentConnection

    # HTTP endpoint
    agent = TextAgentConnection(
        url="https://your-agent.com/chat",
        headers={"Authorization": "Bearer sk-..."},
    )

    # Python callable
    async def my_agent(messages: list[dict]) -> dict:
        ...
        return {"response": "Hello!", "tool_calls": []}

    # Use with LLM tests
    result = asyncio.run(tests.run(agent=agent, test_cases=[...]))
    result = asyncio.run(tests.run(agent=my_agent, test_cases=[...]))

    # Use with LLM simulations
    result = asyncio.run(simulations.run(agent=agent, personas=[...], ...))
    result = asyncio.run(simulations.run(agent=my_agent, personas=[...], ...))
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextAgentConnection:
    """
    Connect to an external text agent via HTTP POST.

    Calibrate sends a fixed request format and expects a fixed response format.
    The agent endpoint must conform to this contract.

    Request (POST to ``url``):

        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How can you help?"}
            ]
        }

    Response (agent must return):

        {
            "response": "The agent's reply text, or null if only tool calls were made",
            "tool_calls": [
                {"tool": "function_name", "arguments": {"key": "value"}}
            ]
        }

    Both fields are required. Use ``null`` / ``[]`` when not applicable.

    Alternatively, pass an async callable directly as the ``agent`` argument
    to ``tests.run()`` / ``simulations.run()`` for full flexibility — the callable
    receives the messages list and must return the same dict format above.

    Example (HTTP):
        >>> from calibrate.connections import TextAgentConnection
        >>> agent = TextAgentConnection(
        ...     url="https://your-agent.com/chat",
        ...     headers={"Authorization": "Bearer sk-..."},
        ... )

    Example (callable):
        >>> async def my_agent(messages: list[dict]) -> dict:
        ...     resp = await call_my_api(messages)
        ...     return {
        ...         "response": resp["text"],
        ...         "tool_calls": [
        ...             {"tool": c["name"], "arguments": c["args"]}
        ...             for c in resp.get("calls", [])
        ...         ],
        ...     }
    """

    url: str
    """HTTP(S) endpoint to POST the messages array to."""

    headers: Optional[dict] = field(default=None)
    """Optional HTTP headers (e.g. ``{"Authorization": "Bearer sk-..."}``). Default: none."""


__all__ = ["TextAgentConnection"]
