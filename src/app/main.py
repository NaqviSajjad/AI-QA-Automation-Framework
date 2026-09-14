"""The application under test: a customer-support AI chat app.

A real (if small) FastAPI service - not a stub - so the framework exercises the
whole path a user takes: browser -> HTTP API -> agent -> RAG/tools -> LLM ->
response. Everything the tests need is here: stable `data-testid` hooks, a
conversation id, a loading state, an error state with retry, and an escalation
indicator.

Run it directly with:  uvicorn src.app.main:app --port 8000
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.agents.graph import get_agent
from src.app.pages import CHAT_PAGE, LOGIN_PAGE
from src.evaluation.collector import record_event
from src.llm.base import LLMEmptyResponseError, LLMError, LLMTimeoutError
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Acme Cloud Customer Support AI", version="1.0.0")

# In-memory stores. This is a test target, not a production service.
SESSIONS: dict[str, str] = {}
CONVERSATIONS: dict[str, dict[str, Any]] = {}
FEEDBACK: list[dict[str, Any]] = []
ESCALATIONS: list[dict[str, Any]] = []


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #
class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    account_id: str | None = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: str = Field(pattern="^(up|down)$")
    comment: str | None = Field(default=None, max_length=1000)


class EscalationRequest(BaseModel):
    conversation_id: str
    reason: str = Field(min_length=3, max_length=500)


def require_session(
    session: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> str:
    token = session
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token or token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Authentication required")
    return SESSIONS[token]


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root(session: str | None = Cookie(default=None)) -> RedirectResponse:
    return RedirectResponse("/chat" if session in SESSIONS else "/login", status_code=302)


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_page() -> HTMLResponse:
    return HTMLResponse(LOGIN_PAGE)


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_page(session: str | None = Cookie(default=None)) -> Response:
    if session not in SESSIONS:
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(CHAT_PAGE)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    agent = get_agent()
    return {
        "status": "ok",
        "llm_provider": agent.llm.name,
        "agent_runtime": agent.runtime,
        "prompt_version": settings.prompts.active_version,
    }


@app.post("/api/auth/login")
def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    if payload.email != settings.auth.email or payload.password != settings.auth.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = secrets.token_urlsafe(24)
    SESSIONS[token] = payload.email
    response.set_cookie("session", token, httponly=True, samesite="lax")
    return {"token": token, "email": payload.email}


@app.post("/api/auth/logout")
def logout(response: Response, session: str | None = Cookie(default=None)) -> dict[str, str]:
    SESSIONS.pop(session or "", None)
    response.delete_cookie("session")
    return {"status": "logged_out"}


@app.post("/api/chat")
def chat(payload: ChatRequest, user: str = Depends(require_session)) -> JSONResponse:
    started = time.perf_counter()
    conversation = CONVERSATIONS.setdefault(
        payload.conversation_id or "",
        {"conversation_id": payload.conversation_id, "user": user, "messages": []},
    )
    history = [
        {"role": m["role"], "content": m["content"]} for m in conversation["messages"]
    ]

    try:
        state = get_agent().run(
            payload.message,
            history=history,
            conversation_id=payload.conversation_id or None,
            account_id=payload.account_id,
        )
    except LLMTimeoutError as exc:
        logger.warning("chat_timeout", extra={"error": str(exc)})
        return JSONResponse(
            status_code=504,
            content={"error": "ai_timeout", "message": "The assistant took too long to respond. Please retry."},
        )
    except LLMEmptyResponseError as exc:
        logger.warning("chat_empty", extra={"error": str(exc)})
        return JSONResponse(
            status_code=502,
            content={"error": "ai_empty_response", "message": "The assistant returned an empty response. Please retry."},
        )
    except LLMError as exc:
        logger.error("chat_failed", extra={"error": str(exc)})
        return JSONResponse(
            status_code=502,
            content={"error": "ai_unavailable", "message": "The assistant is temporarily unavailable. Please retry."},
        )

    body = state.to_public_dict()
    body["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)

    # Move the conversation into its real id the first time round.
    CONVERSATIONS.pop("", None)
    conversation["conversation_id"] = state.conversation_id
    conversation["messages"].extend(
        [
            {"role": "user", "content": payload.message},
            {"role": "assistant", "content": state.response or ""},
        ]
    )
    conversation["category"] = state.category
    conversation["requires_escalation"] = state.requires_escalation
    CONVERSATIONS[state.conversation_id] = conversation

    record_event(
        "latency", latency_ms=body["latency_ms"], conversation_id=state.conversation_id
    )
    record_event(
        "agent",
        conversation_id=state.conversation_id,
        tool_calls=len(state.tool_calls),
        unauthorized=len(state.unauthorized_tool_attempts),
        llm_calls=state.llm_calls,
    )
    if state.requires_escalation:
        ESCALATIONS.append(
            {"conversation_id": state.conversation_id, "reason": "agent_decision", "source": "agent"}
        )
    return JSONResponse(status_code=200, content=body)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user: str = Depends(require_session)) -> dict[str, Any]:
    conversation = CONVERSATIONS.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conversation_id,
        "user": conversation["user"],
        "category": conversation.get("category"),
        "requires_escalation": conversation.get("requires_escalation", False),
        "message_count": len(conversation["messages"]),
        "messages": conversation["messages"],
    }


@app.delete("/api/conversations/{conversation_id}")
def reset_conversation(conversation_id: str, user: str = Depends(require_session)) -> dict[str, str]:
    CONVERSATIONS.pop(conversation_id, None)
    return {"status": "reset", "conversation_id": conversation_id}


@app.post("/api/feedback", status_code=201)
def feedback(payload: FeedbackRequest, user: str = Depends(require_session)) -> dict[str, Any]:
    if payload.conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")
    entry = {**payload.model_dump(), "user": user}
    FEEDBACK.append(entry)
    return {"status": "recorded", **entry}


@app.post("/api/escalation", status_code=201)
def escalate(payload: EscalationRequest, user: str = Depends(require_session)) -> dict[str, Any]:
    if payload.conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")
    ticket = {
        "ticket_id": f"ESC-{len(ESCALATIONS) + 1001}",
        "conversation_id": payload.conversation_id,
        "reason": payload.reason,
        "status": "queued",
        "source": "customer",
    }
    ESCALATIONS.append(ticket)
    CONVERSATIONS[payload.conversation_id]["requires_escalation"] = True
    return ticket


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": request.url.path},
        headers={"X-Request-Id": secrets.token_hex(8)},
    )
