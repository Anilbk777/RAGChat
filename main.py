import logging
import os
from pathlib import Path

import httpx
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from custom_types import RAGError, QueryRequest
from services import (
    inngest_client,
    rag_ingest_pdf,
    rag_query_pdf_ai,
    validate_pdf_file,
)

load_dotenv()

logger = logging.getLogger("uvicorn")


app = FastAPI(title="DocMind RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.exception_handler(RAGError)
async def rag_error_handler(request, exc: RAGError):
    logger.error(f"RAGError [{exc.status_code}]: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    errors = [
        f"{' → '.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request: " + "; ".join(errors)},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected server error occurred. Please try again."},
    )


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise RAGError(f"Failed to read uploaded file: {e}", 400)

    validate_pdf_file(file, raw_bytes)

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.filename

    try:
        file_path.write_bytes(raw_bytes)
    except OSError as e:
        raise RAGError(f"Could not save file to disk: {e}", 500)

    logger.info(f"[API] PDF saved: {file_path} ({len(raw_bytes) / 1024:.1f} KB)")

    try:
        await inngest_client.send(
            inngest.Event(
                name="rag/ingest_pdf",
                data={
                    "pdf_path": str(file_path.resolve()),
                    "source_id": file.filename,
                },
            )
        )
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise RAGError(f"Failed to trigger ingestion pipeline: {e}", 502)

    return {"status": "triggered", "filename": file.filename}


@app.post("/api/query")
async def query_pdf(req: QueryRequest):
    try:
        events = await inngest_client.send(
            inngest.Event(
                name="rag/query_pdf_ai",
                data={"question": req.question, "top_k": req.top_k},
            )
        )
    except Exception as e:
        raise RAGError(f"Failed to send query event to Inngest: {e}", 502)

    if not events:
        raise RAGError("Inngest did not return an event ID.", 502)

    return {"event_id": events[0]}


@app.get("/api/run/{event_id}")
async def get_run_status(event_id: str):
    if not event_id.strip():
        raise RAGError("event_id cannot be empty.", 400)

    inngest_api = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{inngest_api}/events/{event_id}/runs")
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        raise RAGError("Timed out contacting Inngest API.", 504)
    except httpx.HTTPStatusError as e:
        raise RAGError(f"Inngest API returned error {e.response.status_code}.", 502)
    except Exception as e:
        raise RAGError(f"Could not reach Inngest API: {e}", 502)

    runs = data.get("data", [])
    if not runs:
        return {"status": "pending"}

    run = runs[0]
    status = run.get("status")
    output = run.get("output")

    if status in ("Failed", "Cancelled"):
        error_msg = "The background job failed."
        if isinstance(output, dict) and output.get("error"):
            error_msg = output["error"]
        return {"status": status, "error": error_msg}

    return {"status": status, "output": output}

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
