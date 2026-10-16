from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from agents.caldav_worker import create_caldav_worker
from agents.email_worker import create_email_workers
from agents.knowledge_worker import create_knowledge_workers
from agents.search_worker import create_online_search_worker
from agents.weather_worker import create_weather_worker
from config import settings
from core.database import SQLALCHEMY_DATABASE_URL, Base, engine
from graph.workflow import compile_workflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create app DB columns
    Base.metadata.create_all(bind=engine)
    async with AsyncConnectionPool(
        conninfo=SQLALCHEMY_DATABASE_URL,
        kwargs={"autocommit": True},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.db_pool = pool
        # Start MCP servers
        client = MultiServerMCPClient(
            {
                "imap": {
                    "url": "http://imap_mcp:8001/sse",
                    "transport": "sse",
                },
                "online_search": {
                    "url": "http://online_search_mcp:8002/sse",
                    "transport": "sse",
                },
                "caldav": {
                    "url": "http://caldav_mcp:3000/mcp",
                    "transport": "http",
                    "headers": {
                        "Authorization": f"Bearer {settings.CALDAV_BEARER_TOKEN}"
                    },
                },
                "weather": {
                    "url": "http://weather_mcp:8003/sse",
                    "transport": "sse",
                },
                "knowledge": {
                    "url": "http://knowledge_mcp:8004/sse",
                    "transport": "sse",
                    "headers": {"X-Assistant-Id": settings.POSTGRES_DB},
                },
            }
        )
        app.state.mcp_client = client
        worker_tools = [
            *create_caldav_worker(await client.get_tools(server_name="caldav")),
            *create_email_workers(await client.get_tools(server_name="imap")),
            create_online_search_worker(
                await client.get_tools(server_name="online_search")
            ),
            create_weather_worker(await client.get_tools(server_name="weather")),
            *create_knowledge_workers(await client.get_tools(server_name="knowledge")),
        ]
        app.state.agent_graph = compile_workflow(
            worker_tools=worker_tools,
            checkpointer=app.state.checkpointer,
        )
        yield
