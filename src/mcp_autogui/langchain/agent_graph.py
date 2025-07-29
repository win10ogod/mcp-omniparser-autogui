#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import json
from langchain_core.tools import tool
from langchain_core.messages.tool import ToolMessage, ToolCall
from langchain_core.messages.ai import AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

class AgentState(MessagesState):
    pass

def create_agent_graph(llm, tools, debug=False):
    model_with_tools = llm.bind_tools(tools)

    details_on_screen_tool = None
    new_tools = []
    for tool in tools:
        if tool.name == 'omniparser_details_on_screen':
            details_on_screen_tool = tool
        else:
            new_tools.append(tool)
    tools = new_tools

    async def prefix_history_model(state: AgentState):
        ret = []
        for message in state["messages"]:
            if type(message) is ToolMessage and message.name == 'omniparser_details_on_screen':
                new_content = []
                try:
                    for content in json.loads(message.content):
                        if content['type'] == 'text': 
                            new_content.append(content)
                except json.JSONDecodeError:
                    pass
                message.content = json.dumps(new_content)
            ret.append(message)

        state["messages"].clear()
        return {"messages": ret}

    async def add_call_message_model(state: AgentState):
        new_ai_message = AIMessage(content='')
        new_ai_message.tool_calls = [ToolCall(name='omniparser_details_on_screen', args={}, id=str(uuid.uuid4()))]
        return {"messages": [new_ai_message]}

    async def call_model(state: AgentState):
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    async def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if not hasattr(last_message, 'tool_calls') or len(last_message.tool_calls) <= 0:
            return "respond"
        else:
            return "continue"

    workflow = StateGraph(AgentState)

    workflow.add_node("prefix_history", prefix_history_model)
    workflow.add_node("add_call_message", add_call_message_model)
    workflow.add_node("get_screen", ToolNode([details_on_screen_tool]))
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("add_call_message")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "respond": END,
        },
    )

    workflow.add_edge("tools", "prefix_history")
    workflow.add_edge("prefix_history", "add_call_message")
    workflow.add_edge("add_call_message", "get_screen")
    workflow.add_edge("get_screen", "agent")
    graph = workflow.compile(debug=debug)

    return graph