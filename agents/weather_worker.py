from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


# Online search worker agent configuration
def create_weather_worker(tools):
    tools_map = {t.name: t for t in tools}

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

    weather_worker_agent = None
    if settings.USE_WEATHER_SUBAGENT:
        weather_worker_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=tools,
            system_prompt=(
                "You are a weather research assistant. "
                "YOU MUST ALWAYS USE THE `weather_tool` TOOL FOR ANY REQUEST FOR INFORMATION. "
                "Never say that you do not have tools: the weather_tool is at your full disposal, use it."
            ),
        )

    @tool(parse_docstring=True)
    async def weather_worker(city: str, days: int = 1):
        """Use this tool for any weather information search.

        Args:
            city: The name of the city to search the weather for.
            days: Forecasts day to search for
        """
        if weather_worker_agent:
            response = await weather_worker_agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Che tempo farà a {city} per i prossimi {days} giorni?"
                        )
                    ]
                }
            )
            return response["messages"][-1].content
        return await tools_map.get("get_city_weather").ainvoke(
            {"city": city, "days": days}
        )

    return weather_worker
