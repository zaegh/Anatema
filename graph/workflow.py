from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agents.decision_maker import create_decision_maker
from agents.supervisor import get_supervisor_model

from .helpers import create_smart_tool_node, get_supervisor_system_prompt
from .state import State


def compile_workflow(
    worker_tools: list,
    checkpointer,
):
    base_supervisor_model = get_supervisor_model()
    supervisor_with_tools = base_supervisor_model.bind_tools(worker_tools)

    system_prompt = get_supervisor_system_prompt()

    async def supervisor_node(state: State):
        recent_messages = state["messages"][-10:]

        messages_payload = [SystemMessage(content=system_prompt)] + list(
            recent_messages
        )

        response = await supervisor_with_tools.ainvoke(messages_payload)
        return {"messages": [response]}

    tool_node = create_smart_tool_node(ToolNode(worker_tools), create_decision_maker())

    # Funzione di routing basata sulla presenza di tool_calls
    def should_continue(state: State):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "FINISH"

    workflow = StateGraph(State)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor", should_continue, {"tools": "tools", "FINISH": END}
    )
    workflow.add_edge("tools", "supervisor")

    return workflow.compile(checkpointer=checkpointer)
