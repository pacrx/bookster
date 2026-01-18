from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, time, yaml, re, json
from pathlib import Path
from typing import Any, Optional, Dict, List, Literal

from sqlalchemy import create_engine, text
from neo4j import GraphDatabase
from simpleeval import simple_eval

import httpx

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.domain.relationships import ALLOWED_REL_TYPES

app = FastAPI(title="Bookster Orchestrator", version="0.6.0")

CONFIG_DIR = os.getenv("CONFIG_DIR", "/app/config")

def load_yaml(name: str) -> dict:
    p = Path(CONFIG_DIR) / name
    if not p.exists():
        raise HTTPException(500, f"Missing config file: {p}")
    return yaml.safe_load(p.read_text())

runtime_cfg = load_yaml("runtime.yaml")
inv_cfg = load_yaml("invariants.yaml")
ctx_cfg = load_yaml("context.yaml")
tags_cfg = load_yaml("world_state_tags.yaml")
pov_cfg = load_yaml("pov_voice_packets.yaml")

# Path to invariants config (used for UI-based edits)
INVARIANTS_PATH = Path(CONFIG_DIR) / "invariants.yaml"

def reload_invariants() -> None:
    """Reload invariants.yaml into memory after UI edits."""
    global inv_cfg
    inv_cfg = yaml.safe_load(INVARIANTS_PATH.read_text()) if INVARIANTS_PATH.exists() else {"invariants": {}}

OUT_ROOT = Path(runtime_cfg["storage"]["outputs_root"])
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- Postgres ---
db_url = os.getenv(runtime_cfg["storage"]["db_url_env"], os.getenv("DB_URL"))
if not db_url:
    raise RuntimeError("DB_URL not set")
engine = create_engine(db_url, pool_pre_ping=True)

# --- Neo4j ---
neo4j_uri = os.getenv(runtime_cfg["knowledge_graph"]["neo4j_uri_env"], os.getenv("NEO4J_URI"))
neo4j_user = os.getenv(runtime_cfg["knowledge_graph"]["neo4j_user_env"], os.getenv("NEO4J_USER"))
neo4j_pass = os.getenv(runtime_cfg["knowledge_graph"]["neo4j_password_env"], os.getenv("NEO4J_PASSWORD"))
if not (neo4j_uri and neo4j_user and neo4j_pass):
    raise RuntimeError("Neo4j env vars not set")
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

# --- Chroma (prose memory) ---
chroma_url = os.getenv(runtime_cfg["retrieval"]["chroma_url_env"], os.getenv("CHROMA_URL"))
if not chroma_url:
    raise RuntimeError("CHROMA_URL not set")

from urllib.parse import urlparse
u = urlparse(chroma_url)
chroma_client = chromadb.HttpClient(host=u.hostname, port=u.port or 8000) #inside compose network
#chroma_client = chromadb.HttpClient(host="chroma", port=8000)  # inside compose network
embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
embedder = SentenceTransformerEmbeddingFunction(model_name=embed_model)

PROSE_COLLECTION = os.getenv("CHROMA_PROSE_COLLECTION", "prose_memory")
prose_collection = chroma_client.get_or_create_collection(
    name=PROSE_COLLECTION,
    embedding_function=embedder
)

AUTO_APPROVE_TIER = int(os.getenv("PROPOSAL_AUTO_APPROVE_TIER", "1"))

def db_bootstrap():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS projects (
          project_id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          premise TEXT NOT NULL,
          genre TEXT NOT NULL,
          release_version TEXT NOT NULL,
          created_at BIGINT NOT NULL
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS canon_versions (
          project_id TEXT NOT NULL,
          canon_version INTEGER NOT NULL,
          chapter_number INTEGER NOT NULL,
          inworld_time TEXT NULL,
          created_at BIGINT NOT NULL,
          approved BOOLEAN NOT NULL DEFAULT TRUE,
          PRIMARY KEY (project_id, canon_version)
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS proposals (
          id BIGSERIAL PRIMARY KEY,
          project_id TEXT NOT NULL,
          kind TEXT NOT NULL,                 -- "entity" | "relationship"
          payload JSONB NOT NULL,
          trust_tier INTEGER NOT NULL DEFAULT 2,
          status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING|APPROVED|REJECTED
          created_at BIGINT NOT NULL,
          decided_at BIGINT NULL
        );
        """))

@app.on_event("startup")
def on_startup():
    db_bootstrap()

# ---------------- Models ----------------
class ProjectInitReq(BaseModel):
    project_id: str
    title: str
    premise: str
    genre: str = "fantasy"
    style_guide: str = "Write in clear prose. Preserve POV voice. No new canon facts without approval."
    release_version: str = "V1_DRAFT"

class LogicValidateReq(BaseModel):
    invariant_name: str
    scene: Dict[str, Any]
    # Optional: allow invariants to use kg facts
    kg_view: Optional[Dict[str, Any]] = None

class ContextBundleReq(BaseModel):
    project_id: str
    chapter_number: int
    pov: str = "omniscient_neutral"
    pov_guid: Optional[str] = None
    strict_epistemic: Optional[bool] = None
    # Provide a distilled style query if you have it; otherwise system will build one
    style_query: Optional[str] = None

RelType = Literal["KNOWS", "OWNS", "ALLY_OF", "ENEMY_OF", "LOCATED_IN", "MEMBER_OF", "HAS_TITLE"]
assert set(RelType.__args__) == set(ALLOWED_REL_TYPES), "RelType Literal is out of sync with domain ALLOWED_REL_TYPES"

class BibleRel(BaseModel):
    target: str
    type: RelType
    from_chapter: int = 1
    to_chapter: Optional[int] = None
    properties: Dict[str, Any] = {}

class BibleEntity(BaseModel):
    guid: str
    name: str
    valid_from_chapter: int = 1
    valid_to_chapter: Optional[int] = None
    reveal_tag: str = "REVEALED"
    reveal_from_chapter: Optional[int] = None
    entity_type: str = "Entity"
    properties: Dict[str, Any] = {}
    relationships: List[BibleRel] = []

class WorldBibleGrouped(BaseModel):
    project_id: str
    characters: List[BibleEntity] = []
    locations: List[BibleEntity] = []
    factions: List[BibleEntity] = []
    artifacts: List[BibleEntity] = []
    entities: List[BibleEntity] = []

class ProseUpsertReq(BaseModel):
    project_id: str
    chapter_number: int
    pov: Optional[str] = None
    text: str
    label: Optional[str] = None  # e.g., "chapter1_opening"

class ProposalCreateReq(BaseModel):
    project_id: str
    kind: Literal["entity", "relationship"]
    payload: Dict[str, Any]
    trust_tier: int = 2

class ProposalDecisionReq(BaseModel):
    id: int

# ------------- Invariants: robust nested resolver (dict + list) -------------
VAR_PATTERN = re.compile(r"^\$\{(.+)\}$")

def get_nested_var(data: Any, path: str, default=None):
    """
    Supports:
      - dict keys: a.b.c
      - list indices: characters.0.name
    """
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
            else:
                return default
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return default
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return default
        else:
            return default
    return cur

def resolve_ref(v: Any, scene: Dict[str, Any], kg_view: Optional[Dict[str, Any]]):
    if not isinstance(v, str):
        return v
    m = VAR_PATTERN.match(v.strip())
    if not m:
        return v
    inner = m.group(1).strip()
    # Supported roots: scene., kg.
    if inner.startswith("scene."):
        return get_nested_var(scene, inner.replace("scene.", "", 1))
    if inner.startswith("kg."):
        return get_nested_var(kg_view or {}, inner.replace("kg.", "", 1))
    return v

def eval_expr(expr: str, names: Dict[str, Any]) -> Any:
    return simple_eval(expr, names=names)

# ---------------- Helpers ----------------
def latest_canon_meta(project_id: str, chapter_number: int) -> dict:
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT project_id, canon_version, chapter_number, inworld_time, created_at, approved
            FROM canon_versions
            WHERE project_id=:pid AND chapter_number <= :ch
            ORDER BY canon_version DESC
            LIMIT 1
        """), {"pid": project_id, "ch": chapter_number}).mappings().first()
    if not row:
        raise HTTPException(404, "No canon metadata found for this project/chapter")
    return dict(row)

def kg_time_sliced_view(project_id: str, chapter_number: int, pov_guid: Optional[str], strict_ep: bool) -> dict:
    with driver.session() as session:
        if strict_ep and pov_guid:
            nodes_q = """
            MATCH (p:Entity {guid:$pov, project_id:$pid})
            MATCH (p)-[k:KNOWS]->(e:Entity {project_id:$pid})
            WHERE e.valid_from_chapter <= $ch
              AND (e.valid_to_chapter IS NULL OR e.valid_to_chapter >= $ch)
              AND k.from_chapter <= $ch
              AND (k.to_chapter IS NULL OR k.to_chapter >= $ch)
              AND coalesce(e.reveal_from_chapter, e.valid_from_chapter) <= $ch
            RETURN DISTINCT e {.*} AS node
            """
            nodes = [r["node"] for r in session.run(nodes_q, pid=project_id, ch=chapter_number, pov=pov_guid).data()]

            rels_q = """
            MATCH (p:Entity {guid:$pov, project_id:$pid})
            MATCH (p)-[k:KNOWS]->(a:Entity {project_id:$pid})
            WHERE a.valid_from_chapter <= $ch
            AND (a.valid_to_chapter IS NULL OR a.valid_to_chapter >= $ch)
            AND k.from_chapter <= $ch
            AND (k.to_chapter IS NULL OR k.to_chapter >= $ch)
            AND coalesce(a.reveal_from_chapter, a.valid_from_chapter) <= $ch

            WITH collect(DISTINCT a) AS known, p
            WITH known + [p] AS visible

            UNWIND visible AS x
            UNWIND visible AS y
            MATCH (x)-[r]->(y)
            WHERE (r.from_chapter IS NULL OR r.from_chapter <= $ch)
            AND (r.to_chapter IS NULL OR r.to_chapter >= $ch)
            AND type(r) <> "KNOWS"

            RETURN DISTINCT
            x.guid AS source,
            type(r) AS type,
            y.guid AS target,
            properties(r) AS props
            """
            rels = session.run(rels_q, pid=project_id, ch=chapter_number, pov=pov_guid).data()
            return {"nodes": nodes, "relationships": rels}

        nodes_q = """
        MATCH (e:Entity {project_id:$pid})
        WHERE e.valid_from_chapter <= $ch
          AND (e.valid_to_chapter IS NULL OR e.valid_to_chapter >= $ch)
          AND coalesce(e.reveal_from_chapter, e.valid_from_chapter) <= $ch
        RETURN e {.*} AS node
        """
        nodes = [r["node"] for r in session.run(nodes_q, pid=project_id, ch=chapter_number).data()]

        rels_q = """
        MATCH (a:Entity {project_id:$pid})-[r]->(b:Entity {project_id:$pid})
        WHERE a.valid_from_chapter <= $ch
          AND (a.valid_to_chapter IS NULL OR a.valid_to_chapter >= $ch)
          AND b.valid_from_chapter <= $ch
          AND (b.valid_to_chapter IS NULL OR b.valid_to_chapter >= $ch)
          AND (r.from_chapter IS NULL OR r.from_chapter <= $ch)
          AND (r.to_chapter IS NULL OR r.to_chapter >= $ch)
        RETURN a.guid AS source, type(r) AS type, b.guid AS target, properties(r) AS props
        """
        rels = session.run(rels_q, pid=project_id, ch=chapter_number).data()
        return {"nodes": nodes, "relationships": rels}

def upsert_entities(session, project_id: str, entities: List[BibleEntity], default_type: str):
    for e in entities:
        etype = e.entity_type if e.entity_type and e.entity_type != "Entity" else default_type

        # Prevent accidental overwriting of guid inside props
        props = dict(e.properties or {})
        props.pop("guid", None)

        rfc = e.reveal_from_chapter
        if rfc is None:
            rfc = (10**9) if e.reveal_tag == "UNREVEALED" else e.valid_from_chapter

        session.run("""
            MERGE (n:Entity {project_id:$pid, guid:$guid})
            SET n.project_id=$pid,
                n.name=$name,
                n.entity_type=$etype,
                n.valid_from_chapter=$vfc,
                n.valid_to_chapter=$vtc,
                n.reveal_tag=$reveal,
                n.reveal_from_chapter=$rfc
            SET n += $props
            RETURN n.guid AS guid
        """, pid=project_id, guid=e.guid, name=e.name, etype=etype,
           vfc=e.valid_from_chapter, vtc=e.valid_to_chapter, reveal=e.reveal_tag, rfc=rfc,
           props=props)

        for r in e.relationships:
            rel_props = dict(r.properties or {})
            rel_props.pop("guid", None)
            # Deterministic rel_guid for bible ingestion
            rel_guid = f"bible:{project_id}:{e.guid}:{r.type}:{r.target}:{r.from_chapter}"

            upsert_temporal_relationship(
                session=session,
                project_id=project_id,
                src_guid=e.guid,
                tgt_guid=r.target,
                rel_type=str(r.type),
                from_chapter=r.from_chapter,
                to_chapter=r.to_chapter,
                props=rel_props,
                rel_guid=rel_guid,
            )


def chroma_upsert_prose(req: ProseUpsertReq):
    doc_id = f"{req.project_id}:{req.chapter_number}:{req.label or int(time.time())}"
    meta = {"project_id": req.project_id, "chapter_number": req.chapter_number, "pov": req.pov or ""}
    prose_collection.upsert(
        ids=[doc_id],
        documents=[req.text],
        metadatas=[meta]
    )
    return {"status": "ok", "id": doc_id}

def chroma_query_prose(project_id: str, query_text: str, top_k: int = 3):
    # Filter to project only
    res = prose_collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where={"project_id": project_id}
    )
    snippets = []
    # Chroma returns lists per query
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        snippets.append({"text": doc, "meta": meta, "distance": dist})
    return snippets

def build_style_query(req: ContextBundleReq, kg_view: dict) -> str:
    # Distilled query: POV + goal keywords + salient entities.
    # We avoid feeding entire chapter text to reduce noise.
    pov = req.pov
    ch = req.chapter_number
    entities = [n.get("name", "") for n in kg_view.get("nodes", []) if n.get("name")]
    entities = [e for e in entities if e][:8]
    ent_str = ", ".join(entities)
    return f"Writing style reference for POV={pov}, chapter={ch}. Key entities: {ent_str}. Tone and voice continuity."

def upsert_temporal_relationship(
    session,
    project_id: str,
    src_guid: str,
    tgt_guid: str,
    rel_type: str,
    from_chapter: int,
    to_chapter: Optional[int],
    props: Dict[str, Any],
    rel_guid: str,
):
    if src_guid == tgt_guid and rel_type not in {"KNOWS"}:
        raise HTTPException(400, f"Self-relationship not allowed for type={rel_type}")

    # 1) Relationship type allowlist
    if rel_type not in ALLOWED_REL_TYPES:
        raise HTTPException(400, f"Relationship type not allowed: {rel_type}")

    # 2) Chapter sanity
    fc = int(from_chapter or 1)
    tc = int(to_chapter) if to_chapter is not None else None

    if fc < 1:
        raise HTTPException(400, f"from_chapter must be >= 1 (got {fc})")
    if tc is not None and tc < fc:
        raise HTTPException(400, f"to_chapter must be >= from_chapter (got {tc} < {fc})")

    # 3) Sanitize props
    safe_props = dict(props or {})
    safe_props.pop("guid", None)

    # 4) Existence check (better errors; prevents silent no-op)
    exists_q = """
    MATCH (src:Entity {project_id:$pid, guid:$src})
    MATCH (tgt:Entity {project_id:$pid, guid:$tgt})
    RETURN count(src) AS src_ok, count(tgt) AS tgt_ok
    """

    # 5) Temporal relationship write (idempotent by rel_guid)
    cypher = f"""
    MATCH (src:Entity {{project_id:$pid, guid:$src}})
    MATCH (tgt:Entity {{project_id:$pid, guid:$tgt}})

    OPTIONAL MATCH (src)-[existing:`{rel_type}` {{rel_guid:$rel_guid}}]->(tgt)
    WITH src, tgt, existing
    CALL {{
        WITH src, tgt, existing
        WITH src, tgt WHERE existing IS NULL

        OPTIONAL MATCH (src)-[r:`{rel_type}`]->(tgt)
        WHERE r.rel_guid <> $rel_guid
          AND r.from_chapter <= $fc
          AND (r.to_chapter IS NULL OR r.to_chapter >= $fc)
        SET r.to_chapter = $fc - 1

        CREATE (src)-[rel:`{rel_type}` {{
            rel_guid: $rel_guid,
            from_chapter: $fc,
            to_chapter: $tc,
            created_at: datetime()
        }}]->(tgt)
        SET rel += $props

        RETURN 1 AS created
    }}
    RETURN coalesce(created, 0) AS created;
    """

    def _write(tx):
        # Existence validation
        row = tx.run(
            exists_q,
            pid=project_id,
            src=src_guid,
            tgt=tgt_guid,
        ).single()

        if not row or row["src_ok"] == 0 or row["tgt_ok"] == 0:
            raise HTTPException(
                400,
                f"Missing entity for relationship write: project_id={project_id} type={rel_type} src={src_guid} tgt={tgt_guid}"
            )

        # Relationship upsert
        return tx.run(
            cypher,
            pid=project_id,
            src=src_guid,
            tgt=tgt_guid,
            fc=fc,
            tc=tc,
            props=safe_props,
            rel_guid=rel_guid,
        ).single()

    result = session.execute_write(_write)
    return result["created"] if result else 0

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time()), "embed_model": embed_model}


# ---------------- Directory endpoints for author UI ----------------
@app.get("/kg/rel-types")
def kg_rel_types():
    """Return the authoritative allowlist of relationship types."""
    return {"items": sorted(list(ALLOWED_REL_TYPES))}


# Backwards-compatible alias (some UIs may call underscore version)
@app.get("/kg/rel_types")
def kg_rel_types_alias():
    return kg_rel_types()


@app.get("/kg/entities")
def kg_entities(project_id: str, q: Optional[str] = None, limit: int = 200):
    """Fast directory service for the UI to discover entities by name/GUID."""
    q = (q or "").strip().lower()
    lim = max(1, min(int(limit), 500))

    with driver.session() as session:
        if q:
            rows = session.run(
                """
                MATCH (e:Entity {project_id:$pid})
                WHERE toLower(coalesce(e.name,'')) CONTAINS $q OR toLower(e.guid) CONTAINS $q
                RETURN e.guid AS guid, e.name AS name, e.entity_type AS entity_type
                ORDER BY e.name
                LIMIT $lim
                """,
                pid=project_id,
                q=q,
                lim=lim,
            ).data()
        else:
            rows = session.run(
                """
                MATCH (e:Entity {project_id:$pid})
                RETURN e.guid AS guid, e.name AS name, e.entity_type AS entity_type
                ORDER BY e.name
                LIMIT $lim
                """,
                pid=project_id,
                lim=lim,
            ).data()

    return {"items": rows}


# ---------------- Invariants management (author UI) ----------------
@app.get("/invariants/list")
def invariants_list():
    invs = inv_cfg.get("invariants", {}) or {}
    return {"items": [{"name": k, **(v or {})} for k, v in sorted(invs.items())]}


class InvariantUpsertReq(BaseModel):
    name: str
    invariant: Dict[str, Any]


@app.post("/invariants/upsert")
def invariants_upsert(req: InvariantUpsertReq):
    """Upsert a single invariant into invariants.yaml.

    This is intentionally simple and opinionated for author use. Advanced editing can still be
    done in the YAML file directly.
    """
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(400, "Invariant name is required")

    doc = yaml.safe_load(INVARIANTS_PATH.read_text()) if INVARIANTS_PATH.exists() else {"invariants": {}}
    doc.setdefault("invariants", {})
    doc["invariants"][name] = req.invariant or {}
    INVARIANTS_PATH.write_text(yaml.safe_dump(doc, sort_keys=False))
    reload_invariants()
    return {"status": "ok", "name": name}


# ---------------- AI assist endpoints (optional) ----------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")


class AIDraftReq(BaseModel):
    project_id: str
    chapter_number: int
    pov: str = "omniscient_neutral"
    pov_guid: Optional[str] = None
    strict_epistemic: Optional[bool] = None
    prompt: str


@app.post("/ai/ollama/draft")
def ai_ollama_draft(req: AIDraftReq):
    """Draft a scene with Ollama (local on the host).

    NOTE: This assumes Ollama is reachable from Docker via host.docker.internal.
    """
    # Build the same bundle the author would see, and give it to the model as context.
    bundle = context_bundle(ContextBundleReq(
        project_id=req.project_id,
        chapter_number=req.chapter_number,
        pov=req.pov,
        pov_guid=req.pov_guid,
        strict_epistemic=req.strict_epistemic,
    ))

    system = (
        "You are a writing assistant. Use ONLY the provided world facts and POV limits. "
        "Do not invent new canon. If something is unknown, leave it ambiguous or propose a question."
    )
    user = (
        "WORLD_CONTEXT_JSON:\n" + json.dumps(bundle, ensure_ascii=False) +
        "\n\nSCENE_REQUEST:\n" + (req.prompt or "")
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }

    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        text_out = (data.get("message") or {}).get("content") or ""
        return {"status": "ok", "model": OLLAMA_MODEL, "text": text_out}
    except Exception as e:
        raise HTTPException(502, f"Ollama draft failed: {e}")


class AIReviewReq(BaseModel):
    project_id: str
    chapter_number: int
    pov: str = "omniscient_neutral"
    pov_guid: Optional[str] = None
    strict_epistemic: Optional[bool] = None
    draft_text: str


@app.post("/ai/claude/review")
def ai_claude_review(req: AIReviewReq):
    """Review a draft with Claude (Anthropic API)."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set")

    bundle = context_bundle(ContextBundleReq(
        project_id=req.project_id,
        chapter_number=req.chapter_number,
        pov=req.pov,
        pov_guid=req.pov_guid,
        strict_epistemic=req.strict_epistemic,
    ))

    prompt = (
        "You are a strict narrative continuity editor.\n"
        "Given the world context JSON and the draft, identify: (1) contradictions, (2) spoilers/leaks "
        "under strict POV, (3) proposed canon updates needed.\n"
        "Return: a bullet list of issues, and a bullet list of suggested proposals (entity/relationship) "
        "in plain English (not JSON).\n\n"
        "WORLD_CONTEXT_JSON:\n" + json.dumps(bundle, ensure_ascii=False) +
        "\n\nDRAFT:\n" + (req.draft_text or "")
    )

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        with httpx.Client(timeout=120) as client:
            r = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        # Anthropic returns content as a list of blocks
        blocks = data.get("content", [])
        text_out = "".join([b.get("text", "") for b in blocks if b.get("type") == "text"]) or ""
        return {"status": "ok", "model": CLAUDE_MODEL, "text": text_out}
    except Exception as e:
        raise HTTPException(502, f"Claude review failed: {e}")

@app.post("/projects/init")
def init_project(req: ProjectInitReq):
    now = int(time.time())
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO projects (project_id, title, premise, genre, release_version, created_at)
            VALUES (:pid, :t, :p, :g, :rv, :ts)
            ON CONFLICT (project_id) DO NOTHING
        """), {"pid": req.project_id, "t": req.title, "p": req.premise, "g": req.genre, "rv": req.release_version, "ts": now})

        conn.execute(text("""
            INSERT INTO canon_versions (project_id, canon_version, chapter_number, inworld_time, created_at, approved)
            VALUES (:pid, 1, 0, NULL, :ts, TRUE)
            ON CONFLICT (project_id, canon_version) DO NOTHING
        """), {"pid": req.project_id, "ts": now})

    proj_dir = OUT_ROOT / req.project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "style_guide.md").write_text(req.style_guide)

    with driver.session() as session:
        session.run("MERGE (p:Project {project_id:$pid}) SET p.created_at=$ts", pid=req.project_id, ts=now)

    return {"status": "created", "project_id": req.project_id, "canon_version": 1}

@app.post("/kg/upsert")
def kg_upsert(payload: WorldBibleGrouped):
    with driver.session() as session:
        session.run("MERGE (p:Project {project_id:$pid})", pid=payload.project_id)
        upsert_entities(session, payload.project_id, payload.characters, "character")
        upsert_entities(session, payload.project_id, payload.locations, "location")
        upsert_entities(session, payload.project_id, payload.factions, "faction")
        upsert_entities(session, payload.project_id, payload.artifacts, "artifact")
        upsert_entities(session, payload.project_id, payload.entities, "entity")

    total = sum(len(x) for x in [payload.characters, payload.locations, payload.factions, payload.artifacts, payload.entities])
    return {"status": "ok", "project_id": payload.project_id, "entities_upserted": total}

@app.post("/prose/upsert")
def prose_upsert(req: ProseUpsertReq):
    return chroma_upsert_prose(req)

@app.post("/logic/validate")
def logic_validate(req: LogicValidateReq):
    inv = inv_cfg.get("invariants", {}).get(req.invariant_name)
    if not inv:
        raise HTTPException(404, f"Unknown invariant: {req.invariant_name}")

    variables = inv.get("variables", {})
    names: Dict[str, Any] = {}

    for k, v in variables.items():
        rv = resolve_ref(v, req.scene, req.kg_view)
        # Expression if contains operators; else literal resolved value
        if isinstance(rv, str) and any(op in rv for op in ["+", "-", "*", "/", "and", "or", ">", "<", "==", "(", ")"]):
            names[k] = eval_expr(rv, names | {"scene": req.scene, "kg": req.kg_view or {}})
        else:
            names[k] = rv

    rule = inv.get("rule")
    if not rule:
        raise HTTPException(400, f"Invariant missing rule: {req.invariant_name}")

    violated = bool(eval_expr(rule, names | {"scene": req.scene, "kg": req.kg_view or {}}))

    if violated:
        return {
            "severity": inv.get("severity", "BLOCKER"),
            "invariant": req.invariant_name,
            "inputs": {"scene": req.scene},
            "variables": names,
            "rule": rule,
            "why": inv.get("why", "Invariant violated."),
            "suggestions": inv.get("suggestions", []),
        }
    return {"severity": "OK", "invariant": req.invariant_name, "variables": names}

@app.post("/proposals/create")
def proposals_create(req: ProposalCreateReq):
    now = int(time.time())

    # Validate relationship type early (applies to both auto-approved and pending proposals)
    if req.kind == "relationship":
        rel_type = (req.payload or {}).get("type")
        if rel_type not in ALLOWED_REL_TYPES:
            raise HTTPException(400, f"Relationship type not allowed: {rel_type}")

    # Auto-approve Tier 1 if configured
    if req.trust_tier <= AUTO_APPROVE_TIER:
        with engine.begin() as conn:
            row = conn.execute(text("""
                INSERT INTO proposals (project_id, kind, payload, trust_tier, status, created_at, decided_at)
                VALUES (:pid, :k, :p::jsonb, :t, 'APPROVED', :ca, :da)
                RETURNING id
            """), {
                "pid": req.project_id,
                "k": req.kind,
                "p": json.dumps(req.payload),
                "t": req.trust_tier,
                "ca": now,
                "da": now,
            }).first()
            pid = row[0]

        payload = dict(req.payload or {})
        if req.kind == "relationship":
            payload.setdefault("rel_guid", f"proposal:{pid}")

        apply_proposal_to_kg(req.project_id, req.kind, payload)
        return {"status": "auto_approved", "id": pid}

    # Otherwise PENDING
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO proposals (project_id, kind, payload, trust_tier, status, created_at)
            VALUES (:pid, :k, :p::jsonb, :t, 'PENDING', :ca)
            RETURNING id
        """), {
            "pid": req.project_id,
            "k": req.kind,
            "p": json.dumps(req.payload),
            "t": req.trust_tier,
            "ca": now,
        }).first()

    return {"status": "pending", "id": row[0]}

@app.get("/proposals/list")
def proposals_list(project_id: str, status: str = "PENDING", limit: int = 50):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, project_id, kind, payload, trust_tier, status, created_at, decided_at
            FROM proposals
            WHERE project_id=:pid AND status=:st
            ORDER BY created_at ASC
            LIMIT :lim
        """), {"pid": project_id, "st": status, "lim": limit}).mappings().all()
    return {"items": [dict(r) for r in rows]}

def apply_proposal_to_kg(project_id: str, kind: str, payload: Dict[str, Any]):
    with driver.session() as session:
        if kind == "entity":
            entity = BibleEntity(**payload)
            upsert_entities(session, project_id, [entity], default_type=entity.entity_type or "entity")
            return

        if kind != "relationship":
            raise HTTPException(400, f"Unknown proposal kind: {kind}")

        src = payload["source_guid"]
        tgt = payload["target_guid"]
        rel_type = payload["type"]

        fc = int(payload.get("from_chapter", 1))
        tc = payload.get("to_chapter")
        tc = int(tc) if tc is not None else None

        props = dict(payload.get("properties", {}) or {})
        props.pop("guid", None)

        rel_guid = payload.get("rel_guid")
        if not rel_guid:
            raise HTTPException(400, "Relationship payload must include rel_guid for idempotency")

        upsert_temporal_relationship(
            session=session,
            project_id=project_id,
            src_guid=src,
            tgt_guid=tgt,
            rel_type=str(rel_type),
            from_chapter=fc,
            to_chapter=tc,
            props=props,
            rel_guid=rel_guid,
        )

@app.post("/proposals/approve")
def proposals_approve(req: ProposalDecisionReq):
    now = int(time.time())
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id, project_id, kind, payload, trust_tier, status
            FROM proposals WHERE id=:id
        """), {"id": req.id}).mappings().first()

        if not row:
            raise HTTPException(404, "Proposal not found")
        if row["status"] != "PENDING":
            return {"status": "noop", "message": f"Already {row['status']}"}

        conn.execute(text("""
            UPDATE proposals SET status='APPROVED', decided_at=:da WHERE id=:id
        """), {"id": req.id, "da": now})

    payload = dict(row["payload"] or {})

    if row["kind"] == "relationship":
        rel_type = payload.get("type")
        if rel_type not in ALLOWED_REL_TYPES:
            raise HTTPException(400, f"Relationship type not allowed: {rel_type}")

        payload.setdefault("rel_guid", f"proposal:{row['id']}")

    apply_proposal_to_kg(row["project_id"], row["kind"], payload)
    return {"status": "approved", "id": req.id}

@app.post("/context/bundle")
def context_bundle(req: ContextBundleReq):
    canon_meta = latest_canon_meta(req.project_id, req.chapter_number)

    packets = pov_cfg.get("packets", {})
    pov_packet = packets.get(req.pov) or packets.get("omniscient_neutral", {})

    strict_default = bool(ctx_cfg.get("epistemic_filtering", {}).get("enabled", True) and
                          ctx_cfg.get("epistemic_filtering", {}).get("strict", True))
    strict_ep = req.strict_epistemic if req.strict_epistemic is not None else strict_default
        
    if strict_ep and not req.pov_guid:
        raise HTTPException(status_code=400, detail="Strict epistemic mode requires a pov_guid.")

    kg_view = kg_time_sliced_view(req.project_id, req.chapter_number, req.pov_guid, strict_ep)

    # Prose memory: retrieve top snippets via distilled query
    style_query = req.style_query or build_style_query(req, kg_view)
    prose_snips = chroma_query_prose(req.project_id, style_query, top_k=3)

    bundle = {
        "chapter_number": req.chapter_number,
        "canon_meta": canon_meta,
        "pov": req.pov,
        "pov_guid": req.pov_guid,
        "epistemic_filtering": {"strict": strict_ep},
        "pov_style_packet": pov_packet,
        "kg": kg_view,  # nodes + relationships + rel props
        "prose_memory": {
            "query": style_query,
            "snippets": prose_snips
        },
        "invariants": inv_cfg.get("invariants", {}),
        "release_version": tags_cfg["release_gating"]["current_release_version"],
        "manifest": {
            "included": ["canon_meta", "pov_style_packet", "kg.nodes", "kg.relationships", "prose_memory.snippets", "invariants"],
            "note": "Hard facts from KG + soft style from Chroma. Query is distilled to avoid noise.",
        }
    }
    return bundle