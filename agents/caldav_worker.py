from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


# Online search worker agent configuration
def create_caldav_worker(tools):
    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        base_url=settings.OLLAMA_BASE_URL,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        num_predict=500,
    )
    fallback_model = ChatMistralAI(
        model="ministral-3b-latest",
        api_key=settings.MISTRAL_API_KEY,
        temperature=0,
        max_tokens=500,
    )

    if settings.USE_CALDAV_SUBAGENT:
        caldav_worker_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=tools,
            system_prompt=(
                "You are an assistant for organizing and managing schedules. "
                "Use the tools at your disposal to search, create, modify, or remove events and tasks in the calendar or contacts."
            ),
        )

        @tool(parse_docstring=True)
        async def caldav_worker(query: str) -> str:
            """Use this tool for any information search, addition, deletion, or modification regarding calendar and/or contacts, schedules, and appointments.

            Args:
                query: The search query or the information to search, add, modify, or remove.
            """
            response = await caldav_worker_agent.ainvoke(
                {"messages": [HumanMessage(content=query)]}
            )
            return response["messages"][-1].content

        return [caldav_worker]
    else:
        return tools
