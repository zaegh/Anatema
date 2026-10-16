import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from config import settings


class ToolApproval(BaseModel):
    approved: bool = Field(
        description="True if user authorize, confirms or wants to process. False if refuses, blockes or deny action."
    )
    reasoning: str = Field(
        description="A short motivation about why you chose that response."
    )


def get_protected_keywords():
    return [
        "knowledge_writer",
        "add_knowledge",
        "update_knowledge",
        "delete_knowledge",
        "email_sender",
    ]


def get_interrupt_payload(tool_name: str, args: dict) -> dict:
    """
    Genera un payload e un messaggio personalizzato in base al tool e ai suoi argomenti.
    """
    match tool_name:
        case "knowledge_writer":
            description = args.get("query", args.get("search", "ricerca generica"))
            message = f"🔍 Il modello vuole interrogare la Knowledge Base con la query: *'{description}'*."
        case "add_knowledge":
            message = f"🔍 Il modello vuole aggiungere alla Knowledge Base:\nTitolo: *'{args.get('title')}'*\nContenuto: *'{args.get('content')}*'"
        case "update_knowledge":
            message = f"🔍 Il modello vuole modifica un record della Knowledge Base in:\nTitolo: *'{args.get('title')}'*\nContenuto: *'{args.get('content')}*'"
        case "delete_knowledge":
            message = f"🔍 Il modello vuole cancellare il record della Knowledge Base in:\nContenuto: *'{args.get('content')}*'"
        case "email_sender":
            to = args.get("to", "destinatario sconosciuto")
            subject = args.get("subject", "senza oggetto")
            indented_body = "\n".join(
                [f"> {line}" for line in args.get("body").splitlines()]
            )
            message = (
                f"✉️ Il modello sta per inviare un'email a **{to}**.\n\n"
                f"**Oggetto**: *{subject}*\n\n"
                f"**Contenuto**:\n"
                f"{indented_body}"
            )
        case _:
            message = f"⚠️ Il modello richiede l'autorizzazione per eseguire il tool `{tool_name}` con argomenti: `{args}`."
    return {"tool_name": tool_name, "message": message}


def create_smart_tool_node(standard_tool_node, base_model):
    """
    Factory function che avvolge il ToolNode standard aggiungendo
    un controllo condizionale per l'interrupt.
    """
    classifier = base_model.with_structured_output(ToolApproval)

    async def smart_tool_node(state):
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get("name", "").lower()
                protected_keywords = get_protected_keywords()

                if any(keyword in tool_name for keyword in protected_keywords):
                    user_decision = interrupt(
                        get_interrupt_payload(tool_name, tool_call.get("args", {}))
                    )

                    prompt = f"""
                    You are managing a security system for an AI assistant.
                    The user was asked to authorize the use of the '{tool_name}' tool with arguments: {tool_call.get("args")}.
                    The user's response is: '{user_decision}'.
                    Determine if this response is an approval (True) or a denial (False).
                    """
                    decision_result = await classifier.ainvoke(prompt)
                    if not decision_result.approved:
                        return {
                            "messages": [
                                ToolMessage(
                                    content=decision_result.reasoning,
                                    tool_call_id=tool_call["id"],
                                )
                            ]
                        }
        return await standard_tool_node.ainvoke(state)

    return smart_tool_node


def get_supervisor_system_prompt():
    WORKER_REGISTRY = {
        "email_reader": "read emails.",
        "email_sender": "sends email",
        "online_search_worker": "web searches.",
        "caldav_worker": "calendar, events, contacts.",
        "weather_worker": "weather forecasts.",
        "knowledge_reader": "search or consult the knowledge base.",
        "knowledge_writer": "save, edit, or delete notes in the knowledge base.",
        "home_assistant_worker": "manage home automations, control smart devices, or query states.",
    }
    feature_flags = {
        "email_reader": settings.USE_EMAIL_READER_SUBAGENT,
        "email_sender": settings.USE_EMAIL_SENDER_SUBAGENT,
        "online_search_worker": settings.USE_SEARCH_SUBAGENT,
        "caldav_worker": settings.USE_CALDAV_SUBAGENT,
        "weather_worker": settings.USE_WEATHER_SUBAGENT,
        "knowledge_reader": settings.USE_KNOWLEDGE_READER_SUBAGENT,
        "knowledge_writer": settings.USE_KNOWLEDGE_WRITER_SUBAGENT,
        "home_assistant_worker": settings.USE_HAOS_SUBAGENT,
    }
    active_workers = [
        f"- {name}: {description}"
        for name, description in WORKER_REGISTRY.items()
        if feature_flags.get(name, False)
    ]

    workers_block = (
        "\n".join(active_workers)
        if active_workers
        else "- None available at the moment."
    )
    today_date = datetime.now(ZoneInfo("Europe/Rome")).strftime("%Y-%m-%d")

    return textwrap.dedent(f"""\
        You are Anatema, a concise and sharp multi-agent supervisor.
        Today is {today_date}.

        WORKERS:
        {workers_block}

        RULES:
        - ALWAYS call a tool when the user requests an action, search, or note-taking.
        - Respond directly ONLY for general greetings or simple conversation.
        - Be concise (max 1-2 sentences).
        """)
