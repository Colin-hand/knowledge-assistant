import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from knowledge_assistant.api.routes import router
from knowledge_assistant.log import get_logger, new_trace_id, setup_logging

logger = get_logger(__name__)


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="Internal Knowledge Assistant", version="0.1.0")

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        trace_id = new_trace_id()
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled_api_error")
            return JSONResponse(
                status_code=500,
                content={"detail": "internal error", "trace_id": trace_id},
                headers={"X-Trace-Id": trace_id},
            )
        response.headers["X-Trace-Id"] = trace_id
        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )
        return response

    app.include_router(router)
    return app


app = create_app()
