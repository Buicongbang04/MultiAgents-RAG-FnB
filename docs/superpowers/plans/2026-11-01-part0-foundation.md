# Part 0 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing chatbot reproducible and testable, move the catalog/knowledge source of truth to PostgreSQL (Neo4j becomes a projection), add a Claude LLM backend behind the existing LLM interface, and bring the whole stack up with one `docker compose up` + one bootstrap command.

**Architecture:** FastAPI app keeps its current chat pipeline. New `app/db` (SQLAlchemy 2 async + Alembic) owns catalog tables; `app/catalog` reads them; `app/knowledge/graph_writer.py` (refactored out of `scripts/ingest_mock_to_neo4j.py`) projects Postgres rows into the *same* Neo4j graph shape the retriever already uses. `app/llm` gains `generate_structured()` and an `AnthropicClient`.

**Tech Stack:** Python 3.11, uv + pyproject.toml + uv.lock, FastAPI, SQLAlchemy 2.x (async, asyncpg), Alembic, PostgreSQL 16, Neo4j 5, `anthropic` SDK, pytest + pytest-asyncio, Docker Compose, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-01-fnb-ordering-platform-design.md` (sections 2 H10, 5, 6.1, 6.3, 7 row "0", 10, 11).

## Global Constraints

- Python 3.11 only (`requires-python = ">=3.11,<3.12"`).
- Config rule stays: schema fields in `app/core/config.py` have **no defaults**; every value lives in `configs/app.yaml`; sections use `extra="forbid"`; override via env with `__` (spec §3.1 of README).
- Secrets only from env / `.env` (never committed): `NEO4J__PASSWORD`, `DATABASE__URL` password, `ANTHROPIC_API_KEY`.
- Money is integer VND (`int`), never float.
- All Vietnamese text stored NFC-normalized (`unicodedata.normalize("NFC", s)`).
- Neo4j node ids for menu variants keep the legacy ids (`menu_0001`…), FAQ ids (`faq_0001`…), chunk ids (`doc_0001`…).
- Every design decision gets an ADR in `docs/adr/NNNN-kebab-title.md` (Context / Decision / Why / Alternatives / Consequences) and is listed in the task's final report to the user (spec §11). ADR numbers for this plan are fixed below.
- Existing behaviour must not regress: the conversation probe (Task 9) must show the same intents and no new errors versus the baseline recorded in spec §2.
- Commit after each task on branch `feat/part0-foundation` (never on `main`).

## Review Focus

1. **Bootstrap run twice** → no duplicate rows in Postgres, no duplicate nodes in Neo4j, same counts (21 products, 55 variants, 30 FAQ, 20 chunks). Test in Task 4 and Task 5.
2. **Product status `sold_out` or `hidden`** → projected `MenuItem.available = false`, so `get_menu_names()` and retrieval exclude it. Test in Task 5.
3. **`llm.backend: anthropic` without credentials** → app fails at startup with a clear message naming `ANTHROPIC_API_KEY`, not on the first customer request. Test in Task 6.
4. **Postgres down** → `/health` returns `"postgres": false` and status `degraded` instead of raising. Test in Task 7.
5. **Unicode-composed vs decomposed Vietnamese** (e.g. "Trà vải" typed with combining marks) → seed stores NFC and lookups normalize, so names still match. Test in Task 4.

## File Structure

```
pyproject.toml, uv.lock                 # Task 1 — replaces requirements*.txt as source of truth
docs/adr/0000-template.md … 0009-*.md    # one ADR per decision (task that makes it)
tests/conftest.py, tests/unit/, tests/integration/
scripts/checks/                          # old ad-hoc tests/*.py moved here (not pytest)
app/db/base.py                           # DeclarativeBase, naming convention
app/db/session.py                        # engine/sessionmaker factory, get_db dependency, ping
app/db/models/catalog.py                 # Product, ProductVariant, OptionGroup, ProductOptionGroup, Faq, PolicyChunk, ShopSetting
alembic.ini, alembic/env.py, alembic/versions/0001_catalog.py
app/catalog/schemas.py                   # Pydantic DTOs returned by repository
app/catalog/repository.py                # read functions used by projection and Part 1
app/catalog/seed.py                      # CSV → Postgres (idempotent)
app/knowledge/graph_writer.py            # Postgres DTOs → Neo4j (refactor of scripts/ingest_mock_to_neo4j.py)
app/knowledge/indexes.py                 # constraints + vector indexes
scripts/bootstrap.py                     # migrate → seed → project → embed
app/llm/base.py (modify), app/llm/anthropic_client.py, app/llm/scripted.py, app/llm/factory.py (modify), app/llm/sglang.py (modify)
app/rag/embedding_client.py, app/rag/reranker.py, app/rag/retriever.py (modify) # off-loop + dtype
app/main.py, app/api/routes.py (modify)  # warmup, health
Dockerfile, docker-compose.yml, .env.example, .dockerignore
.github/workflows/ci.yml
```

---

### Task 1: Reproducible environment, pytest, ADR scaffolding

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `docs/adr/0000-template.md`, `docs/adr/0001-record-decisions-as-adrs.md`, `docs/adr/0003-uv-pyproject-lockfile.md`, `tests/conftest.py`, `tests/unit/test_config.py`
- Modify: `.gitignore` (add `.venv/`, `.venv-*/`), `requirements*.txt` (replace contents with a one-line pointer comment to `pyproject.toml`, keep files so old docs don't 404), `Instructions.md` (install section → `uv sync --extra ml --extra ui`), `.vscode/settings.json` (drop conda manager)
- Move: `tests/*.py` → `scripts/checks/` (`git mv`; they are scripts, not pytest tests)

**Interfaces:**
- Produces: `uv sync --extra ml --extra ui` creates `.venv`; `uv run pytest` runs `tests/`; marker `integration` registered; fixture `settings` in `tests/conftest.py` returning `get_settings()`.

- [ ] **Step 1:** Write `pyproject.toml`: project deps = current `requirements.txt` set + `sqlalchemy[asyncio]>=2.0,<2.1`, `asyncpg`, `alembic`, `anthropic` (latest 1.x). Extras: `ml` (torch, transformers==4.57.6, huggingface_hub==0.36.0, sentence-transformers, `datasets>=3,<4.4`, peft, accelerate, bitsandbytes>=0.44.0, trl), `ui` (chainlit==2.0.3), `serving` (sglang[all]). `[tool.pytest.ini_options]`: `testpaths=["tests"]`, `asyncio_mode="auto"`, `asyncio_default_fixture_loop_scope="function"`, `markers=["integration: needs Postgres/Neo4j from docker compose"]`, `addopts="-m 'not integration'"`. The `datasets`/`huggingface_hub` pins are the ones proven to work in spec §2 H10.
- [ ] **Step 2:** `uv lock && uv sync --extra ml --extra ui` — expected: resolves; `uv run python -c "import datasets, transformers, sentence_transformers"` exits 0.
- [ ] **Step 3: Write the failing test** `tests/unit/test_config.py::test_settings_load_and_forbid_unknown_keys` — asserts `get_settings().llm.backend in {"mock","sglang","vllm"}` and that constructing `LLMSection(**get_settings().llm.model_dump(), typo_key=1)` raises `pydantic.ValidationError` (Task 6 adds `"anthropic"` to the set).
- [ ] **Step 4:** `uv run pytest -q` → expected `1 passed`.
- [ ] **Step 5:** Write ADR 0000 template, ADR 0001 (why ADRs: user requirement spec §11), ADR 0003 (uv+lock: unpinned ML deps broke on 2026-11-01 — datasets 2.14 vs pyarrow 26, hub 2.x vs transformers 4.57; alternatives: pip-tools, conda-lock, poetry).
- [ ] **Step 6: Commit** `chore: pin deps with uv lockfile, add pytest + ADR scaffolding`

---

### Task 2: Docker Compose topology

**Files:**
- Create: `docker-compose.yml`, `Dockerfile`, `.dockerignore`, `.env.example`, `docs/adr/0007-compose-topology.md`
- Modify: `Instructions.md` (new quick start), `configs/app.yaml` (add `database:` section — values only; schema added in Task 3)

**Interfaces:**
- Produces services: `postgres` (postgres:16, db `fnb`, host port `${POSTGRES_PORT:-5433}`), `postgres-test` (profile `test`, db `fnb_test`, host port `${POSTGRES_TEST_PORT:-5434}`), `neo4j` (neo4j:5, 7474/7687), `neo4j-test` (profile `test`, host 7688 bolt), `backend` (built from `Dockerfile`, port 8001, depends_on healthy postgres+neo4j, GPU reservation optional via `profiles: [gpu]` override), `sglang` (profile `local-llm`, GPU, port 30000). All with healthchecks and named volumes.

- [ ] **Step 1:** Write compose + Dockerfile (python:3.11-slim, install uv, `uv sync --frozen --extra ml --no-dev`, `CMD uv run uvicorn app.main:app --host 0.0.0.0 --port 8001`). Host ports chosen to avoid the user's existing Postgres on 15432 and keep 5432 free.
- [ ] **Step 2:** `.env.example` lists `NEO4J__PASSWORD`, `POSTGRES_PASSWORD`, `DATABASE__URL`, `ANTHROPIC_API_KEY`, `LLM__BACKEND`, `EMBEDDING__DEVICE`, `RERANKER__DEVICE`.
- [ ] **Step 3: Verify** `docker compose up -d postgres neo4j && docker compose ps` → both `healthy`. `docker compose --profile test up -d postgres-test neo4j-test` → healthy.
- [ ] **Step 4:** ADR 0007 (why compose for infra + backend, test services isolated so integration tests never wipe the dev graph; why SGLang is an opt-in profile now that cloud is default).
- [ ] **Step 5: Commit** `chore: docker compose for postgres, neo4j, backend, optional sglang`

---

### Task 3: Postgres catalog schema + Alembic

**Files:**
- Create: `app/db/base.py`, `app/db/session.py`, `app/db/models/__init__.py`, `app/db/models/catalog.py`, `alembic.ini`, `alembic/env.py`, `alembic/versions/0001_catalog.py`, `tests/integration/conftest.py`, `tests/integration/test_catalog_schema.py`, `docs/adr/0002-postgres-source-of-truth.md`, `docs/adr/0004-sqlalchemy-async-alembic.md`
- Modify: `app/core/config.py` (add `DatabaseSection(url: str, echo: bool, pool_size: int)` and `database: DatabaseSection` on `Settings`), `configs/app.yaml`

**Interfaces:**
- Produces (`app/db/session.py`): `get_engine() -> AsyncEngine` (lru_cached), `get_sessionmaker() -> async_sessionmaker[AsyncSession]`, `async def get_db() -> AsyncIterator[AsyncSession]` (FastAPI dependency), `async def ping_database() -> bool`.
- Produces (`app/db/models/catalog.py`):
  - `ProductStatus(str, Enum)`: `active`, `sold_out`, `hidden`.
  - `Product`: `id: str` (slug PK, e.g. `"bac-xiu"`), `name_vi`, `name_en`, `category` (`coffee|tea|freeze|food`), `description`, `tags: list[str]` (ARRAY), `ingredients: list[str]`, `image_url: str|None`, `status: ProductStatus`, `sort_order: int`, `created_at`, `updated_at`.
  - `ProductVariant`: `id: str` (legacy `menu_0001`), `product_id` FK, `size: str` (`S|M|L`), `price_vnd: int` (CHECK > 0), unique (`product_id`,`size`).
  - `OptionGroup`: `code: str` PK (`sugar|ice|temperature`), `name_vi`, `choices: list[dict]` (JSONB, ordered `[{"code","label_vi"}]`), `default_choice: str`.
  - `ProductOptionGroup`: (`product_id`,`group_code`) PK.
  - `Faq`: `id`, `topic`, `question`, `answer`, `language`.
  - `PolicyChunk`: `id`, `chunk_index: int`, `language`, `text`, `entities: list[dict]` (JSONB), `source_file`.
  - `ShopSetting`: `key: str` PK, `value: dict` (JSONB), `updated_at`.
- Integration fixture `db_session` (tests/integration/conftest.py): connects to `fnb_test` via env `TEST_DATABASE_URL` (default `postgresql+asyncpg://fnb:fnb@localhost:5434/fnb_test`), runs `alembic upgrade head` once per session, wraps each test in a rolled-back transaction.

- [ ] **Step 1: Write the failing test** `test_catalog_schema.py::test_variant_price_must_be_positive` — inserting `ProductVariant(price_vnd=0)` raises `IntegrityError`; `::test_variant_size_unique_per_product` — two `M` variants for one product raise `IntegrityError`.
- [ ] **Step 2:** `uv run pytest -m integration tests/integration/test_catalog_schema.py` → FAIL (module not found).
- [ ] **Step 3:** Implement models, session, Alembic env (async, reads `get_settings().database.url`, overridable by `-x url=`), migration `0001_catalog`. Naming convention on `MetaData` so constraint names are stable.
- [ ] **Step 4:** Re-run → PASS.
- [ ] **Step 5:** ADR 0002 (Postgres = truth, Neo4j = projection; why: transactions, constraints, admin edits in Part 3, spec §6.1; alternative: keep Neo4j as truth — rejected, no constraints/transactions for orders). ADR 0004 (SQLAlchemy 2 async + Alembic vs SQLModel/raw asyncpg).
- [ ] **Step 6: Commit** `feat(db): catalog schema in postgres with alembic`

---

### Task 4: Seed catalog from mock data (with approved FAQ rewrites)

**Files:**
- Create: `app/catalog/seed.py`, `app/catalog/schemas.py`, `app/catalog/repository.py`, `tests/unit/test_seed_rules.py`, `tests/integration/test_seed.py`, `docs/adr/0008-option-groups-and-seed-rules.md`
- Modify: `data/mock/faq.csv` rows `faq_0010`, `faq_0011`, `faq_0028`; any `data/mock/docs.jsonl` chunk that says delivery is only via third-party apps or that pre-ordering is impossible

**Interfaces:**
- Produces (`app/catalog/seed.py`): `def group_menu_rows(rows: list[dict]) -> list[ProductSeed]` (pure), `def option_groups_for(category: str, tags: set[str]) -> list[str]` (pure), `async def seed_catalog(db: AsyncSession, data_dir: Path, *, reset: bool = False) -> SeedStats`. `SeedStats(products:int, variants:int, faqs:int, chunks:int)`.
- Produces (`app/catalog/schemas.py`): `VariantDTO(id, size, price_vnd)`, `ProductDTO(id, name_vi, name_en, category, description, tags, ingredients, image_url, status, variants: list[VariantDTO], option_groups: list[str])`, `FaqDTO`, `PolicyChunkDTO`, `OptionGroupDTO(code, name_vi, choices, default_choice)`.
- Produces (`app/catalog/repository.py`): `async def list_products(db, *, include_hidden: bool = False) -> list[ProductDTO]`, `async def get_product(db, product_id: str) -> ProductDTO | None`, `async def list_faqs(db) -> list[FaqDTO]`, `async def list_policy_chunks(db) -> list[PolicyChunkDTO]`, `async def list_option_groups(db) -> list[OptionGroupDTO]`, `async def get_shop_setting(db, key: str) -> dict | None`.

Values pinned by spec §3 / §10 (exact):
- Option groups: `sugar` "Đường" choices `khong|it|binh_thuong|nhieu` → labels "Không đường", "Ít đường", "Bình thường", "Nhiều đường", default `binh_thuong`. `ice` "Đá" choices `khong|it|binh_thuong` → "Không đá", "Ít đá", "Bình thường", default `binh_thuong`. `temperature` "Nóng/Đá" choices `da|nong` → "Đá", "Nóng", default `da`.
- Assignment rule (`option_groups_for`): category `food` → `[]`; tag `hot` → `["sugar"]`; tag `iced` → `["sugar","ice"]`; category `freeze` → `["sugar"]`; category `tea` → `["sugar","ice"]`; category `coffee` with neither `hot` nor `iced` (Cappuccino, Latte, Americano) → `["sugar","ice","temperature"]`.
- Product slug: NFC → strip diacritics (`đ→d`) → lowercase → non-alnum to `-` (e.g. "Bạc xỉu" → `bac-xiu`, "Trà đào cam sả" → `tra-dao-cam-sa`).
- Product description/tags/ingredients: from the `S` row's description with the size/price clause removed (regex `\s*size\s+[SML]\s*\([^)]*\)`), tags/ingredients = union across sizes.
- Shop settings seeded: `delivery` = `{"fee_vnd": 15000, "max_days_ahead": 2, "open": "07:00", "close": "22:00"}`; `queue` = `{"scheduled_lead_minutes": 20}`.
- FAQ rewrites (exact copy):
  - `faq_0010` answer: "Có ạ! Quán tự giao hàng, phí giao **15.000đ**/đơn. Anh/chị đặt ngay trong khung chat này, chọn **Giao hàng** và nhập địa chỉ là được ạ."
  - `faq_0011` answer: "Quán không bán qua GrabFood hay ShopeeFood ạ. Anh/chị đặt trực tiếp ngay tại đây, quán tự giao với phí **15.000đ**/đơn ạ."
  - `faq_0028` answer: "Quán không nhận giữ bàn trước ạ. Nhưng anh/chị có thể **đặt món trước** (mang đi hoặc giao hàng) và hẹn giờ nhận trong vòng 2 ngày, từ 7:00 đến 22:00 ạ."

- [ ] **Step 1: Write failing unit tests** `test_seed_rules.py`: `test_option_groups_rules` (one assert per rule row above, using real product tag sets: Bạc xỉu `{"iced",…}` → sugar,ice; Cà phê đen nóng → sugar; Latte `{}`/coffee → sugar,ice,temperature; Chocolate Freeze → sugar; Trà vải → sugar,ice; Sừng Bò → []); `test_group_menu_rows_counts` (reading real `data/mock/menu.csv` → 21 products, 55 variants, Bạc xỉu sizes `["S","M","L"]` prices `[35000,45000,55000]`); `test_slug` ("Bạc xỉu"→"bac-xiu", "Trà đào cam sả"→"tra-dao-cam-sa"); `test_nfc_normalization` (decomposed `unicodedata.normalize("NFD","Trà vải")` seeds as the NFC string); `test_no_third_party_delivery_claims` (no FAQ answer or docs.jsonl text contains "GrabFood" except faq_0011's "không bán qua GrabFood", none contains "không nhận đặt bàn trước").
- [ ] **Step 2:** `uv run pytest tests/unit/test_seed_rules.py` → FAIL.
- [ ] **Step 3:** Implement `seed.py`, `schemas.py`, `repository.py`; edit the CSV/JSONL rows.
- [ ] **Step 4:** Unit tests PASS.
- [ ] **Step 5: Write failing integration test** `test_seed.py::test_seed_is_idempotent` — `seed_catalog` twice → `SeedStats(21,55,30,20)` both times and table counts unchanged; `::test_repository_returns_variants_sorted` — `get_product(db,"bac-xiu").variants` sizes `S,M,L`.
- [ ] **Step 6:** Implement until PASS (`INSERT … ON CONFLICT DO UPDATE`; `reset=True` truncates catalog tables first).
- [ ] **Step 7:** ADR 0008 (option-group model and rules; why temperature only for untagged coffee; why free of charge — spec §3).
- [ ] **Step 8: Commit** `feat(catalog): seed postgres catalog from mock data, rewrite delivery/pre-order FAQ`

---

### Task 5: Project Postgres → Neo4j, automatic indexes, bootstrap command

**Files:**
- Create: `app/knowledge/__init__.py`, `app/knowledge/graph_writer.py`, `app/knowledge/indexes.py`, `scripts/bootstrap.py`, `tests/integration/test_projection.py`
- Modify: `scripts/ingest_mock_to_neo4j.py` (becomes a thin deprecated wrapper calling bootstrap), `scripts/embed_graph.py` (expose `async def embed_graph() -> dict`), `app/rag/menu.py` (add `invalidate_menu_caches()`; normalize names NFC in `get_menu_variants`), `app/rag/neo4j_client.py` (accept explicit uri/user/password for test instance), `README.md`/`Instructions.md` (remove manual Cypher step)

**Interfaces:**
- Consumes: Task 4 repository functions and DTOs.
- Produces: `async def project_catalog(db: AsyncSession, graph: Neo4jClient) -> ProjectionStats` (clears graph, writes MenuItem per variant with the **same properties as today** — `id, name_vi, name_en, price, size, category, ingredients, description, tags, available` — plus `product_id`; `available = (product.status == "active")`; then FAQ, Chunk, Entity, Category, domain entities and links exactly as the current ingest script). `def ensure_indexes(graph) -> None` (constraints + the three 1024-dim cosine vector indexes `menu_embedding`, `faq_embedding`, `chunk_embedding`, `IF NOT EXISTS`). `invalidate_menu_caches() -> None` clears `get_menu_vocabulary`/`get_menu_names` lru caches.
- `python -m scripts.bootstrap [--reset] [--skip-embed]`: alembic upgrade head → `seed_catalog` → `ensure_indexes` → `project_catalog` → `embed_graph` (unless skipped) → prints counts.

- [ ] **Step 1: Write failing integration tests** (against `neo4j-test`, env `TEST_NEO4J_URI` default `bolt://localhost:7688`): `test_projection_counts` (MenuItem 55, FAQ 30, Chunk 20, Category 4 after project), `test_projection_idempotent` (project twice → same counts), `test_sold_out_not_available` (set `bac-xiu` status `sold_out`, project → its 3 MenuItems have `available=false`; after `invalidate_menu_caches()` "Bạc xỉu" not in `get_menu_names()` when menu module points at the test graph), `test_vector_indexes_online` (`SHOW VECTOR INDEXES` has the 3 names).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement by moving Cypher from `scripts/ingest_mock_to_neo4j.py` into `graph_writer.py` functions that take DTOs instead of CSV rows. Do not change Cypher semantics (retriever depends on them).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Verify against dev stack** `uv run python -m scripts.bootstrap --reset` → prints products 21 / variants 55 / faqs 30 / chunks 20 / embedded menu 55, faq 30, chunks 20.
- [ ] **Step 6: Commit** `feat(knowledge): project postgres catalog into neo4j, bootstrap command`

---

### Task 6: LLM layer — structured output, Claude backend, scripted fake

**Files:**
- Create: `app/llm/anthropic_client.py`, `app/llm/scripted.py`, `app/llm/errors.py`, `tests/unit/test_llm_factory.py`, `tests/unit/test_anthropic_client.py`, `docs/adr/0005-llm-provider-interface-claude-default.md`
- Modify: `app/llm/base.py`, `app/llm/factory.py`, `app/llm/sglang.py` (shared `httpx.AsyncClient` created once, closed in `aclose`; implement `generate_structured` via OpenAI-compatible `response_format={"type":"json_schema",...}`), `app/llm/mock.py` (`generate_structured` raises `LLMUnavailableError`), `app/core/config.py` (`LLMSection.backend: Literal["mock","sglang","vllm","anthropic"]`; new `AnthropicSection(model: str, timeout_seconds: float, max_retries: int)` at `llm.anthropic`), `configs/app.yaml` (`llm.anthropic.model: claude-opus-5-5`, `timeout_seconds: 30`, `max_retries: 2`), `app/main.py` (lifespan: `get_llm_client()` constructed at startup; `await client.aclose()` on shutdown), `tests/unit/test_config.py` (backend set now includes `anthropic`)

**Interfaces:**
- `app/llm/base.py` adds to `LLMGenerateRequest`: `effort: Optional[Literal["low","medium","high"]] = None`. Adds to `BaseLLMClient`:
  - `async def generate_structured(self, request: LLMGenerateRequest, schema: type[T]) -> T` where `T` is a Pydantic `BaseModel` subclass; raises `LLMStructuredOutputError` when the output does not validate, `LLMUnavailableError` on transport failure/timeouts.
  - `async def aclose(self) -> None` (default no-op).
- `app/llm/errors.py`: `class LLMError(Exception)`, `class LLMUnavailableError(LLMError)`, `class LLMStructuredOutputError(LLMError)`, `class LLMConfigurationError(LLMError)`.
- `app/llm/anthropic_client.py`: `class AnthropicClient(BaseLLMClient)`, `backend = "anthropic"`, uses `anthropic.AsyncAnthropic(timeout=…, max_retries=…)`. Constructor raises `LLMConfigurationError("…ANTHROPIC_API_KEY…")` when no credential source resolves. Must not send `temperature`/`top_p` (rejected by current API); maps `request.effort` to `output_config={"effort": …}` (omit when `None`); system prompt (+ summary) goes in `system` as a cacheable block (`cache_control: {"type":"ephemeral"}`); streaming via `client.messages.stream(...)` yielding text deltas; structured via the SDK's structured-outputs helper. **Before writing this file the implementer invokes the `claude-api` skill and follows `python/claude-api/README.md`, `streaming.md` and `tool-use.md` (structured outputs) exactly — do not write SDK calls from memory.** Handle `stop_reason == "refusal"` by raising `LLMUnavailableError` so callers use their fallback.
- `app/llm/scripted.py`: `class ScriptedLLMClient(BaseLLMClient)` — constructed with `text_responses: list[str]` and `structured_responses: list[BaseModel | dict]`, pops in order, records every request in `.calls`; used by unit tests and offline evals.
- `get_llm_client()` returns `AnthropicClient` for `"anthropic"`.

- [ ] **Step 1: Write failing tests**: `test_llm_factory.py::test_anthropic_backend_without_key_fails_fast` (monkeypatch env: `LLM__BACKEND=anthropic`, no `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`, `HOME` to tmp → constructing raises `LLMConfigurationError` whose message contains `ANTHROPIC_API_KEY`); `test_scripted_client_structured_roundtrip` (schema `class P(BaseModel): sugar: str` → returns `P(sugar="it")` and records the request); `test_anthropic_client.py::test_request_has_no_temperature_and_maps_effort` (inject a fake `AsyncAnthropic` whose `messages.create` captures kwargs; assert `"temperature" not in kwargs`, `kwargs["output_config"]["effort"] == "low"`, `kwargs["model"] == "claude-opus-5-5"`); `::test_refusal_raises_unavailable`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement per Interfaces.
- [ ] **Step 4:** Run → PASS. Also `uv run pytest -q` all unit tests PASS.
- [ ] **Step 5: Live smoke (only if the user has provided a key):** `LLM__BACKEND=anthropic uv run python -m scripts.checks.llm_smoke` (create this tiny script: one generate, one stream, one structured call; prints latency + first-token time). If no key, record "skipped — no key" in the task report.
- [ ] **Step 6:** ADR 0005 (one interface, three backends; Claude default for demo per spec §5/§10; why `claude-opus-5-5` + `effort` per call instead of a smaller model; structured outputs for NLU; prompt caching on the fixed system prompt; local SGLang kept).
- [ ] **Step 7: Commit** `feat(llm): structured outputs, Claude backend, scripted fake for tests`

---

### Task 7: Runtime robustness — off-loop ML calls, fp16, warmup, health

**Files:**
- Create: `tests/unit/test_health.py`, `tests/unit/test_embedding_offloop.py`, `docs/adr/0009-offload-blocking-ml-and-warmup.md`
- Modify: `app/rag/embedding_client.py`, `app/rag/reranker.py`, `app/rag/retriever.py` (call sites), `app/main.py`, `app/api/routes.py` (`/health`), `app/core/config.py` + `configs/app.yaml` (`embedding.dtype`, `reranker.dtype`: `float16|float32`; `server.warmup_on_startup: bool`)

**Interfaces:**
- `SentenceTransformerEmbeddingClient.embed_text/embed_batch` run `model.encode` via `asyncio.to_thread`. Reranker exposes `async def arerank(...)` with the same args/return as `rerank`, running `predict` in a thread; retriever awaits `arerank`. Models load with `torch_dtype` from config (`float16` only when device is `cuda`; otherwise force `float32`).
- `/health` JSON adds `"postgres": bool` (from `ping_database()`), top-level `status` = `"ok"` if neo4j and postgres true else `"degraded"`; never raises.
- Lifespan: if `server.warmup_on_startup`, run one `embed_text("khởi động")` and one rerank pair before serving; log duration.

- [ ] **Step 1: Write failing tests**: `test_health.py::test_health_degraded_when_postgres_down` (monkeypatch `ping_database` → False; `TestClient` GET `/health` → 200, `status=="degraded"`, `postgres is False`); `test_embedding_offloop.py::test_encode_runs_in_thread` (fake model whose `encode` records `threading.get_ident()`; assert it differs from the event-loop thread).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Verify VRAM** with dev stack + SGLang running (`--mem-fraction-static 0.6`): start backend with `EMBEDDING__DEVICE=cuda RERANKER__DEVICE=cuda` and fp16 → `nvidia-smi` total < 12 GB and first `/chat` after startup < 3 s (warm). Record numbers in the ADR.
- [ ] **Step 6:** ADR 0009 (blocking encode/predict in async code serialised all customers — README §17; fp16 halves VRAM so 7B AWQ + both models fit 12 GB; warmup removes the 35–39 s first-request delay measured in spec §2).
- [ ] **Step 7: Commit** `perf: run embedding/rerank off the event loop, fp16 on GPU, warmup, postgres health`

---

### Task 8: CI pipeline + test strategy ADR

**Files:**
- Create: `.github/workflows/ci.yml`, `docs/adr/0006-test-strategy.md`

**Interfaces:**
- CI jobs: `unit` (ubuntu, `astral-sh/setup-uv`, `uv sync --frozen` without `ml` extra, `EMBEDDING__BACKEND=mock RERANKER__BACKEND=null LLM__BACKEND=mock uv run pytest -q`), `integration` (service containers postgres:16 + neo4j:5 on the test ports, `uv run pytest -m integration -q`).

- [ ] **Step 1:** Make sure unit tests import nothing that needs torch when `embedding.backend=mock` (lazy imports in `embedding_client.py`/`reranker.py`). Verify locally: `uv venv /tmp/ci-venv && UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv sync --frozen && UV_PROJECT_ENVIRONMENT=/tmp/ci-venv EMBEDDING__BACKEND=mock RERANKER__BACKEND=null uv run pytest -q` → all pass.
- [ ] **Step 2:** Write workflow. **Step 3:** ADR 0006 (unit vs integration split, isolated test DBs, ScriptedLLMClient instead of live LLM in CI, live-LLM evals run manually because they cost money).
- [ ] **Step 4: Commit** `ci: unit and integration test workflow`

---

### Task 9: End-to-end regression check and Part 0 report

**Files:**
- Create: `scripts/checks/conversation_probe.py` (the probe from spec §2: sessions a/b/c/d with the same messages; prints intent, router type, latency, answer), `docs/part0-report.md`

- [ ] **Step 1:** `docker compose up -d postgres neo4j && uv run python -m scripts.bootstrap --reset && uv run python run.py` then `uv run python scripts/checks/conversation_probe.py` with `LLM__BACKEND=mock`.
- [ ] **Step 2: Expected:** every turn returns 200; intents identical to the mock-run baseline in spec §2 (order, order, order, order, order, order / faq×4 / consultant×3, order / order, order, order, order, ignore); faq_0010-style question "Có giao hàng không?" now answers with the new copy (self-delivery, 15.000đ). Known H1–H9 bugs are expected to remain (Part 1 fixes them).
- [ ] **Step 3:** If a key exists, repeat with `LLM__BACKEND=anthropic` and record per-turn latency.
- [ ] **Step 4:** Write `docs/part0-report.md`: what changed, measured numbers (VRAM, cold/warm latency, probe output), list of ADRs 0001–0009 with one-line "why" each, anything skipped and why. This is the report the user receives (spec §11).
- [ ] **Step 5: Commit** `docs: part 0 report and conversation probe`
