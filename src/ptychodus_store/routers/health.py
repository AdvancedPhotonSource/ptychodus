from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from ptychodus_store.routers.deps import SessionDep
from ptychodus_store.routers.schemas import HealthRead

router = APIRouter(tags=['health'])


@router.get('/health', response_model=HealthRead)
async def health(request: Request, session: SessionDep) -> HealthRead:
    db_state: str = 'ok'
    try:
        await session.execute(text('SELECT 1'))
    except Exception:  # noqa: BLE001
        db_state = 'down'

    watcher = getattr(request.app.state, 'watcher', None)
    if watcher is None:
        watcher_state = 'disabled'
    elif watcher.is_alive:
        watcher_state = 'alive'
    else:
        watcher_state = 'dead'

    status = 'ok' if db_state == 'ok' and watcher_state != 'dead' else 'degraded'
    return HealthRead(status=status, db=db_state, watcher=watcher_state)  # type: ignore[arg-type]
