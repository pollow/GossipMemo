from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .config import Settings
from .llm import create_llm
from .models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ManualMemoryRequest,
    QueryRequest,
    QueryResponse,
    RetractRequest,
    SupersedeRequest,
)
from .store import AmbiguousPersonError, SqliteWorldStore
from .world import SocialMemoryWorld


def build_world(settings: Settings) -> SocialMemoryWorld:
    return SocialMemoryWorld(
        store=SqliteWorldStore(settings.database_path),
        model=create_llm(settings),
    )


def create_app(
    settings: Settings,
    world: SocialMemoryWorld | None = None,
) -> FastAPI:
    world = world or build_world(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await world.start()
        try:
            yield
        finally:
            await world.stop()

    app = FastAPI(
        title="GossipMemo",
        version="0.1.0",
        description="Provenance-aware social memory for agents",
        lifespan=lifespan,
    )
    app.state.world = world
    app.state.settings = settings

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
        "/v1/spaces/{space_id}/query",
        response_model=QueryResponse,
        dependencies=protected,
    )
    async def query(space_id: str, request: QueryRequest) -> QueryResponse:
        return await world.query(space_id, request)

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
        relationship, memories = context
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
