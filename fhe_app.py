from typing import Literal
from uuid import UUID

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field

from logging_config import setup_logging

logger = setup_logging()

from auth import API_PREFIX, supabase_auth_middleware
from fhe_key_gen import NUM_SLOTS, RING_BASE, RING_DIM
from fhe_encrypt_service import (
    ENCRYPTED_DIR,
    ManifestContext,
    encrypt_csv,
    plan_encryption,
    preprocess_csv,
)
from fhe_key_load import KeyLoadError
from key_storage import KEYS_DIR, generate_and_store
from supabase_db import (
    SupabaseError,
    SupabaseNotFoundError,
    resolve_fhe_key,
    resolve_model,
    insert_fhe_key_record,
)

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
    num_slots: int = Field(
        ...,
        description="Maximum number of CKKS slots (NumSlots = 2^(ring_base-1))",
    )
    slots: int = Field(
        ...,
        description="Same as num_slots; stored in Supabase fhe_keys.slots",
    )
    ring_base: int = Field(..., description="Ring base parameter (ring_dim = 2^ring_base)")
    ring_dim: int = Field(..., description="Ring dimension used for key generation")
    supabase_record: dict


class FheEncryptResponse(BaseModel):
    model_id: str
    fhe_key_id: str
    fhe_key_storage_path: str
    file_name: str | None
    file_size: int
    slots: int
    params_count: int
    rows_per_ciphertext: int
    total_rows: int
    ciphertext_count: int
    removed_columns: list[str]
    columns: list[str]
    encrypt_id: str
    output_dir: str
    ciphertext_files: list[str]
    manifest_file: str
    status: str


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


@app.exception_handler(KeyLoadError)
async def key_load_error_handler(request: Request, exc: KeyLoadError):
    logger.warning(
        "Key load error %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(SupabaseNotFoundError)
async def supabase_not_found_handler(request: Request, exc: SupabaseNotFoundError):
    logger.warning(
        "Supabase not found %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(status_code=404, content={"detail": str(exc)})


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
    return {
        "status": "ok",
        "keys_dir": str(KEYS_DIR),
        "encrypted_dir": str(ENCRYPTED_DIR),
    }


@router.post(
    "/keys",
    response_model=CreateFheKeyResponse,
    summary="Generate Key Pair",
    description="Generate a CKKS FHE key pair and store it locally. Returns the key id and slot capacity.",
)
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
        slots=NUM_SLOTS,
        key_id=key_id,
        user_id=user_id,
        access_token=access_token,
    )

    logger.info(
        "Generated key pair key_id=%s ring_base=%s ring_dim=%s num_slots=%s",
        key_id,
        RING_BASE,
        RING_DIM,
        NUM_SLOTS,
    )

    return CreateFheKeyResponse(
        key_id=key_id,
        scheme=scheme,
        multiplicative_depth=body.mult_depth,
        num_slots=NUM_SLOTS,
        slots=NUM_SLOTS,
        ring_base=RING_BASE,
        ring_dim=RING_DIM,
        supabase_record=record,
    )


@router.post("/fhe-encrypt", response_model=FheEncryptResponse)
async def fhe_encrypt(
    request: Request,
    model_id: str = Form(...),
    fhe_key_id: str = Form(...),
    file: UploadFile = File(...),
):
    user_id = _user_id(request)
    access_token = _access_token(request)
    file_content = await file.read()

    logger.info(
        "fhe-encrypt request user_id=%s model_id=%s fhe_key_id=%s file=%s size=%s",
        user_id,
        model_id,
        fhe_key_id,
        file.filename,
        len(file_content),
    )

    fhe_key = resolve_fhe_key(fhe_key_id=fhe_key_id, access_token=access_token)
    model = resolve_model(model_id=model_id, access_token=access_token)

    logger.info(
        "fhe-encrypt resolved fhe_key supabase_id=%s storage_path=%s slots=%s key_name=%s "
        "model_name=%s model_type=%s",
        fhe_key.id,
        fhe_key.storage_path,
        fhe_key.slots,
        fhe_key.key_name,
        model.name,
        model.model_type,
    )

    try:
        csv_result = preprocess_csv(file_content)
        logger.info(
            "fhe-encrypt csv columns=%s removed=%s params_count=%s",
            csv_result.columns,
            csv_result.removed_columns,
            model.params_count,
        )
        encrypt_plan = plan_encryption(fhe_key.slots, model.params_count, csv_result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        encrypt_result = encrypt_csv(
            fhe_key_storage_path=fhe_key.storage_path,
            manifest_ctx=ManifestContext(
                supabase_fhe_key_id=fhe_key.id,
                fhe_key_storage_path=fhe_key.storage_path,
                model_id=model.id,
                model_name=model.name,
                model_type=model.model_type,
            ),
            csv=csv_result,
            plan=encrypt_plan,
        )
    except KeyLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "fhe-encrypt complete encrypt_id=%s slots=%s params_count=%s "
        "rows_per_ciphertext=%s total_rows=%s ciphertext_count=%s "
        "output_dir=%s files=%s",
        encrypt_result.encrypt_id,
        encrypt_plan.slots,
        encrypt_plan.params_count,
        encrypt_plan.rows_per_ciphertext,
        encrypt_plan.total_rows,
        encrypt_plan.ciphertext_count,
        encrypt_result.output_dir,
        encrypt_result.ciphertext_files,
    )

    return FheEncryptResponse(
        model_id=model_id,
        fhe_key_id=fhe_key_id,
        fhe_key_storage_path=fhe_key.storage_path,
        file_name=file.filename,
        file_size=len(file_content),
        slots=encrypt_plan.slots,
        params_count=encrypt_plan.params_count,
        rows_per_ciphertext=encrypt_plan.rows_per_ciphertext,
        total_rows=encrypt_plan.total_rows,
        ciphertext_count=encrypt_plan.ciphertext_count,
        removed_columns=encrypt_plan.removed_columns,
        columns=encrypt_plan.columns,
        encrypt_id=encrypt_result.encrypt_id,
        output_dir=str(encrypt_result.output_dir),
        ciphertext_files=encrypt_result.ciphertext_files,
        manifest_file=encrypt_result.manifest_file,
        status="encrypted",
    )


app.include_router(router)
