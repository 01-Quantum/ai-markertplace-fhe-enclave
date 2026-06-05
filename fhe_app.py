from typing import Literal
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field

from logging_config import setup_logging

logger = setup_logging()

from auth import API_PREFIX, supabase_auth_middleware
from key_storage import KEYS_DIR, generate_and_store
from supabase_db import SupabaseError, insert_fhe_key_record

app = FastAPI(
    title="FHE Vault",
    version="1.0.0",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)
app.middleware("http")(supabase_auth_middleware)
router = APIRouter(prefix=API_PREFIX)


class CreateFheKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Human-readable key name")
    key_type: Literal["CKKS"] = Field(
        default="CKKS",
        validation_alias=AliasChoices("key-type", "key_type"),
        serialization_alias="key-type",
        description="FHE scheme (currently CKKS only)",
    )
    mult_depth: int = Field(
        default=7,
        ge=1,
        le=32,
        validation_alias=AliasChoices("mult-depth", "mult_depth", "multiplicative_dep"),
        serialization_alias="mult-depth",
        description="Multiplicative depth for CKKS parameters",
    )

    model_config = {"populate_by_name": True}


class CreateFheKeyResponse(BaseModel):
    key_id: str
    scheme: str
    multiplicative_depth: int
    supabase_record: dict


def _scheme_label(key_type: str) -> str:
    if key_type.upper() == "CKKS":
        return "OpenFHE CKKS"
    return f"OpenFHE {key_type}"


def _user_id(request: Request) -> UUID:
    return UUID(request.state.supabase_user["id"])


def _access_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=403, detail="Missing or invalid Authorization header")
    return auth[7:].strip()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "Validation error %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        logger.error(
            "HTTP error %s %s -> %s: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
    else:
        logger.warning(
            "HTTP error %s %s -> %s: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(SupabaseError)
async def supabase_error_handler(request: Request, exc: SupabaseError):
    logger.error(
        "Supabase error %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled error %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@router.get("/health")
def health():
    return {"status": "ok", "keys_dir": str(KEYS_DIR)}


@router.post("/keys", response_model=CreateFheKeyResponse)
def create_fhe_key(body: CreateFheKeyRequest, request: Request):
    if body.key_type != "CKKS":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported key-type '{body.key_type}'. Only CKKS is supported.",
        )

    user_id = _user_id(request)
    access_token = _access_token(request)

    key_id = generate_and_store(mult_depth=body.mult_depth)
    scheme = _scheme_label(body.key_type)
    record = insert_fhe_key_record(
        key_name=body.name,
        scheme=scheme,
        multiplicative_depth=body.mult_depth,
        key_id=key_id,
        user_id=user_id,
        access_token=access_token,
    )

    return CreateFheKeyResponse(
        key_id=key_id,
        scheme=scheme,
        multiplicative_depth=body.mult_depth,
        supabase_record=record,
    )
app.include_router(router)
