from __future__ import annotations

import secrets
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .config import Settings
from .llm import create_llm
from .logging import configure_logging, elapsed_ms, request_id_context
from .models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ManualMemoryRequest,
    QueryRequest,
    QueryResponse,
    RetractRequest,
    SupersedeRequest,
    TurnRequest,
    TurnResponse,
)
from .store import AmbiguousPersonError, SqliteWorldStore
from .world import SocialMemoryWorld


def build_world(settings: Settings) -> SocialMemoryWorld:
    return SocialMemoryWorld(
        store=SqliteWorldStore(settings.database_path),
        model=create_llm(settings),
        extraction_batch_size=settings.extraction_batch_size,
        extraction_batch_timeout_seconds=settings.extraction_batch_timeout_seconds,
    )


def create_app(
    settings: Settings,
    world: SocialMemoryWorld | None = None,
) -> FastAPI:
    configure_logging(settings.logging_level, settings.logging_format)
    world = world or build_world(settings)
    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        started = time.perf_counter()
        logger.info("startup_begin")
        await world.start()
        logger.info("startup_complete", extra={"duration_ms": elapsed_ms(started)})
        try:
            yield
        finally:
            started = time.perf_counter()
            await world.stop()
            logger.info("shutdown_complete", extra={"duration_ms": elapsed_ms(started)})

    app = FastAPI(
        title="GossipMemo",
        version="0.1.0",
        description="Provenance-aware social memory for agents",
        lifespan=lifespan,
    )
    app.state.world = world
    app.state.settings = settings

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "").strip()
        valid = (
            supplied
            and len(supplied) <= 128
            and all(32 <= ord(character) < 127 for character in supplied)
        )
        request_id = supplied if valid else str(uuid.uuid4())
        token = request_id_context.set(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            logger.exception(
                "http_request_failed",
                extra={"method": request.method, "path": request.url.path},
            )
            raise
        finally:
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": elapsed_ms(started),
                },
            )
            request_id_context.reset(token)

    @app.exception_handler(AmbiguousPersonError)
    async def ambiguous_person(
        _: Request, error: AmbiguousPersonError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    async def authorize(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not settings.api_key:
            return
        expected = f"Bearer {settings.api_key}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid bearer token",
            )

    protected = [Depends(authorize)]

    @app.get("/health", response_model=HealthResponse)
    @app.get("/healthz", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        return world.health()

    @app.post(
        "/v1/spaces/{space_id}/ingest",
        response_model=IngestResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=protected,
    )
    async def ingest(space_id: str, request: IngestRequest) -> IngestResponse:
        return await world.ingest(space_id, request)

    @app.post(
        "/v1/spaces/{space_id}/turns",
        response_model=TurnResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=protected,
    )
    async def turn(space_id: str, request: TurnRequest) -> TurnResponse:
        return await world.turn(space_id, request)

    @app.post(
        "/v1/spaces/{space_id}/query",
        response_model=QueryResponse,
        dependencies=protected,
    )
    async def query(space_id: str, request: QueryRequest) -> QueryResponse:
        return await world.query(space_id, request)

    @app.get("/v1/spaces/{space_id}/context", dependencies=protected)
    async def context_bundle(space_id: str) -> dict:
        return world.store.context_bundle(space_id).model_dump()

    @app.get(
        "/v1/spaces/{space_id}/people/{person_id}",
        dependencies=protected,
    )
    async def person_dossier(space_id: str, person_id: str) -> dict:
        context = world.store.read(
            space_id,
            QueryRequest(
                question="dossier",
                people=[person_id],
                include_relationships=True,
                expand_relationships=1,
                include_evidence=True,
                limit=100,
            ),
        )
        if not context.people:
            raise HTTPException(status_code=404, detail="person not found or ambiguous")
        return context.model_dump()

    @app.get(
        "/v1/spaces/{space_id}/relationships/{relationship_id}",
        dependencies=protected,
    )
    async def relationship_dossier(space_id: str, relationship_id: str) -> dict:
        context = world.store.relationship_context(space_id, relationship_id)
        if not context:
            raise HTTPException(status_code=404, detail="relationship not found")
        relationship, memories, _watermark = context
        return {
            "relationship": relationship.model_dump(),
            "memories": [memory.model_dump() for memory in memories],
        }

    @app.post(
        "/v1/spaces/{space_id}/memories",
        status_code=status.HTTP_201_CREATED,
        dependencies=protected,
    )
    async def add_memory(space_id: str, request: ManualMemoryRequest) -> dict:
        return {"id": world.add_memory(space_id, request), "status": "active"}

    @app.post(
        "/v1/spaces/{space_id}/memories/{memory_id}/retract",
        dependencies=protected,
    )
    async def retract_memory(
        space_id: str, memory_id: str, request: RetractRequest
    ) -> dict:
        if not world.retract_memory(space_id, memory_id, request.reason):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"id": memory_id, "status": "retracted"}

    @app.post(
        "/v1/spaces/{space_id}/memories/{memory_id}/supersede",
        status_code=status.HTTP_201_CREATED,
        dependencies=protected,
    )
    async def supersede_memory(
        space_id: str, memory_id: str, request: SupersedeRequest
    ) -> dict:
        replacement_id = world.supersede_memory(space_id, memory_id, request)
        if not replacement_id:
            raise HTTPException(status_code=404, detail="active memory not found")
        return {
            "id": replacement_id,
            "status": "active",
            "supersedes_memory_id": memory_id,
        }

    return app
