from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


# Email worker agent configuration
def create_email_workers(tools):
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

    if any(
        [not settings.USE_EMAIL_READER_SUBAGENT, not settings.USE_EMAIL_SENDER_SUBAGENT]
    ):
        tools_map = {t.name: t for t in tools}

    reader_agent = None
    if settings.USE_EMAIL_READER_SUBAGENT:
        reader_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=[t for t in tools if t.name in ["get_emails"]],
            system_prompt=(
                "You are an assistant specialized in email. Use the tools at your disposal to search and read emails when requested."
            ),
        )

    @tool(parse_docstring=True)
    async def email_reader(task: str) -> str:
        """Use this tool SOLELY and EXCLUSIVELY to manage or read mails.

        Args:
            task: The detailed description of the email operation to be performed.
        """
        if reader_agent:
            response = await reader_agent.ainvoke(
                {"messages": [HumanMessage(content=task)]}
            )
            return response["messages"][-1].content
        return await tools_map.get("get_emails").ainvoke({"query": task})

    sender_agent = None
    if settings.USE_EMAIL_SENDER_SUBAGENT:
        sender_agent = create_agent(
            model=model.with_fallbacks([fallback_model]),
            tools=[t for t in tools if t.name in ["send_email"]],
            system_prompt=(
                "You are an assistant specialized in email. Use the tools at your disposal to write and send emails when requested."
            ),
        )

    @tool(parse_docstring=True)
    async def email_sender(
        to: str,
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """Use this tool SOLELY and EXCLUSIVELY for sending emails.

        Args:
            to: the address of the email recipient
            subject: subject of the email
            body: content of the email
            cc: Optional email address(es) for carbon copy (comma-separated if multiple).
            bcc: Optional email address(es) for blind carbon copy (comma-separated if multiple).
        """
        if sender_agent:
            if cc is None:
                cc = []
            if bcc is None:
                bcc = []
            cc_str = ", ".join(cc) if cc else "nessuno"
            bcc_str = ", ".join(bcc) if bcc else "nessuno"
            prompt = f"Invia un'email a {to} con oggetto '{subject}' e testo: {body}. CC: {cc_str}, BCC: {bcc_str}"
            response = await sender_agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )
            return response["messages"][-1].content
        return await tools_map.get("send_email").ainvoke(
            {
                "to": to,
                "subject": subject,
                "body": body,
                "cc": cc,
                "bcc": bcc,
            }
        )

    return [email_reader, email_sender]
