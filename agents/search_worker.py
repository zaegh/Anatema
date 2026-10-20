from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


# Online search worker agent configuration
def create_online_search_worker(tools):
    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        base_url=settings.OLLAMA_BASE_URL,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        num_predict=1000,
    )
    fallback_model = ChatMistralAI(
        model="ministral-3b-latest",
        api_key=settings.MISTRAL_API_KEY,
        temperature=0,
        max_tokens=1000,
    )

    search_worker_agent = None
    if settings.USE_SEARCH_SUBAGENT:
        search_worker_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=tools,
            system_prompt=(
                "You are an online research assistant. "
                "YOU MUST ALWAYS USE THE `online_search` TOOL FOR ANY INFORMATION REQUEST FROM THE WEB. "
                "Never say that you do not have tools: the online_search tool is at your full disposal."
            ),
        )

    @tool(parse_docstring=True)
    async def online_search_worker(query: str) -> str:
        """Use this tool for any information search on the web, websites, company data, or online VAT numbers/tax IDs.

        Args:
            query: The search query or the information to search on the internet.
        """
        if search_worker_agent:
            response = await search_worker_agent.ainvoke(
                {"messages": [HumanMessage(content=query)]}
            )
            return response["messages"][-1].content
        tools_map = {t.name: t for t in tools}
        return await tools_map.get("web_search").ainvoke({"query": query})

    return online_search_worker
