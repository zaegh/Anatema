import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from users.security import verify_open_webui_secret_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "Anatema", "object": "model", "owned_by": "Yiska Benally"}],
    }


@router.post("/chat/completions")
async def chat_completions(
    request: Request, api_key: str = Depends(verify_open_webui_secret_key)
):
    body = await request.json()
    messages = body.get("messages", [])
    latest_message = messages[-1]["content"] if messages else ""

    config = {
        "configurable": {
            "thread_id": request.headers.get("x-openwebui-chat-id", "default_thread")
        }
    }

    graph = request.app.state.agent_graph

    current_state = await graph.aget_state(config)

    async def event_generator():
        try:
            if current_state.interrupts:
                stream_input = Command(resume=latest_message)
            else:
                stream_input = {"messages": [("user", latest_message)]}

            async for chunk, metadata_meta in graph.astream(
                stream_input, config=config, stream_mode="messages"
            ):
                if (
                    metadata_meta.get("langgraph_node") == "supervisor"
                    and chunk.content
                ):
                    data = {
                        "choices": [
                            {
                                "delta": {"content": chunk.content},
                                "finish_reason": None,
                            }
                        ]
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            post_state = await graph.aget_state(config)
            if post_state.interrupts:
                data = post_state.interrupts[0].value
                approval_message = (
                    f"\n\n{data.get('message')}\n\n"
                    f"*Rispondi **'sì'** per autorizzare o **'no'** per rifiutare.*"
                )

                data = {
                    "choices": [
                        {
                            "delta": {"content": approval_message},
                            "finish_reason": "stop",
                        }
                    ]
                }
                yield f"data: {json.dumps(data)}\n\n"
            else:
                data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(data)}\n\n"

        except Exception:
            logger.exception("Error during graph execution")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
