from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


def create_knowledge_workers(tools: list):
    model = ChatOllama(
        model="qwen2.5:7b",
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

    writer_tools = [
        t
        for t in tools
        if any(k in t.name.lower() for k in ["search", "delete", "update", "add"])
    ]
    reader_tools = [t for t in tools if any(k in t.name.lower() for k in ["search"])]
    final_tools = []

    if settings.USE_KNOWLEDGE_WRITER_SUBAGENT:
        writer_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=writer_tools,
            system_prompt=(
                "You are a strict execution agent for Knowledge Base write operations.\n"
                "AVAILABLE TOOLS: `add_knowledge`, `delete_knowledge`, `update_knowledge`, `search_knowledge`\n\n"
                "CRITICAL RULES:\n"
                "- If asked to create a note, extract title and content from the instruction and execute `add_knowledge`.\n"
                "- If asked to update a note, identify the note using `search_knowledge`, extract title and content from the instruction and execute `update_knowledge`.\n"
                "- If asked to delete, cancel, destroy a note, identify the note using `search_knowledge`, extract title and content from the instruction and execute 'delete_knowledge'.\n"
                "- If category is needed but missing, use 'all'.\n"
            ),
        )

        @tool(parse_docstring=True)
        async def knowledge_writer(instruction: str):
            """Use this tool exclusively when the user asks to store, save, annotate, modify, or delete information in the Knowledge Base.

            Args:
                instruction: The detailed instruction specifying what to save, update, or remove.
            """
            response = await writer_agent.ainvoke(
                {"messages": [HumanMessage(content=instruction)]}
            )
            return response["messages"][-1].content

        final_tools.append(knowledge_writer)
    else:
        final_tools.extend(writer_tools)

    if settings.USE_KNOWLEDGE_READER_SUBAGENT:
        reader_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=reader_tools,
            system_prompt=(
                "You are a strict execution agent for Knowledge Base search operations.\n"
                "AVAILABLE TOOLS: `search_knowledge`.\n\n"
                "CRITICAL RULES:\n"
                "- If the user asks to list, show, or display ALL notes/information (e.g. 'elencami le note', 'mostra tutto'), "
                "YOU MUST ALWAYS call `search_knowledge(category='all', query='all')` FIRST before answering.\n"
                "- If asked for specific information, call `search_knowledge` with the appropriate category/query.\n"
                "- NEVER tell the user the list is limited or empty without calling `search_knowledge` first.\n"
                "- Format the search results into a clean, clear summary for the user."
            ),
        )

        @tool(parse_docstring=True)
        async def knowledge_reader(query: str):
            """Use this tool exclusively to search, read, find, or consult information already present in the Knowledge Base.

            Args:
                query: The search query or information request.
            """
            response = await reader_agent.ainvoke(
                {"messages": [HumanMessage(content=query)]}
            )
            return response["messages"][-1].content

        final_tools.append(knowledge_reader)
    else:
        for t in reader_tools:
            if t.name not in {t.name for t in final_tools}:
                final_tools.append(t)

    return final_tools
