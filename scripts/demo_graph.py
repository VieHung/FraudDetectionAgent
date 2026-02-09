"""Demo runner for the ReAct graph.

Run this file to see the message flow produced by the graph in src/react_agent/graph.py.
"""

from __future__ import annotations

import asyncio
import os
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from react_agent import graph
from react_agent.context import Context


def _format_messages(messages: Iterable[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"USER: {msg.content}")
        elif isinstance(msg, ToolMessage):
            lines.append(f"TOOL[{msg.name}]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_names = ", ".join(tc.get("name", "?") for tc in msg.tool_calls)
                lines.append(f"ASSISTANT(tool_calls: {tool_names}): {msg.content}")
            else:
                lines.append(f"ASSISTANT: {msg.content}")
        else:
            lines.append(f"MESSAGE[{type(msg).__name__}]: {msg.content}")
    return "\n".join(lines)


async def main() -> None:
    """
    Docstring for main
    """
    prompt = os.getenv("DEMO_PROMPT", "Who is the founder of LangChain?")

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        context=Context(),
    )

    print(_format_messages(result["messages"]))


if __name__ == "__main__":
    asyncio.run(main())
