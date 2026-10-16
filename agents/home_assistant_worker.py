from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


# Online search worker agent configuration
def create_home_assistant_worker(tools):
    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        base_url=settings.OLLAMA_BASE_URL,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        num_predict=1000,
    )
    fallback_model = ChatMistralAI(
        model="ministral-8b-latest",
        api_key=settings.MISTRAL_API_KEY,
        temperature=0,
        max_tokens=1000,
    )

    if settings.USE_HAOS_SUBAGENT:
        haos_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=tools,
            system_prompt=(
                "You are a specialized Home Assistant Execution Worker.\n"
                "Your sole responsibility is to translate user requests into precise Home Assistant actions, query entity states, and trigger automations or scripts.\n\n"
                "Operational Guidelines:\n"
                "1. Precision & Context: Always identify target entities, areas, domains, and specific state parameters (e.g., brightness %, temperature in °C, color, duration) required to fulfill the request.\n"
                "2. State Verification: When checking status or reading sensors, report exact values along with their unit of measurement.\n"
                "3. Execution & Safety: Perform actions directly. If an action is destructive, irreversible, or high-risk (e.g., unlocking smart locks, opening garage doors), flag it clearly in your response.\n"
                "4. Handling Ambiguity: If crucial parameters are missing or an entity/room cannot be determined, report the exact missing information back so the supervisor can ask the user for clarification.\n"
                "5. Output Format: Provide concise, factual, and action-oriented execution summaries."
            ),
        )

        @tool(parse_docstring=True)
        async def home_assistant_worker(query: str) -> str:
            """Executes home automation commands or queries device statuses via Home Assistant.

            Use this tool whenever you need to interact with smart home devices, lights, climate controls,
            switches, sensors, or check the state of the home environment.

            Args:
                query: A clear and descriptive natural language instruction or query for Home Assistant.
                       Should specify target entities/rooms and the desired action or state check
                       (e.g., 'Turn on the living room lights at 50% brightness', 'What is the current temperature in the bedroom?').

            Returns:
                A string containing the result of the action, the updated device status, or error messages from Home Assistant.
            """
            if haos_agent:
                response = await haos_agent.ainvoke(
                    {"messages": [HumanMessage(content=query)]}
                )
                return response["messages"][-1].content
            tools_map = {t.name: t for t in tools}
            return await tools_map.get("haos").ainvoke({"query": query})

        return [home_assistant_worker]
    else:
        return tools
