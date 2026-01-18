import os
import re
import json
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st
import yaml

ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8000")

st.set_page_config(page_title="Bookster Writer UI", layout="wide")
st.title("Bookster Writer UI (v6.3)")

# ---------------- HTTP helpers ----------------
def post(path: str, payload: dict, timeout: int = 30) -> httpx.Response:
    return httpx.post(f"{ORCH_URL}{path}", json=payload, timeout=timeout)


def get(path: str, params: Optional[dict] = None, timeout: int = 30) -> httpx.Response:
    return httpx.get(f"{ORCH_URL}{path}", params=params or {}, timeout=timeout)


# ---------------- Sidebar globals ----------------
with st.sidebar:
    st.header("Project Settings")
    ACTIVE_PROJECT = st.text_input("Active Project ID", "my_series")
    ACTIVE_CHAPTER = st.number_input("Current Chapter", min_value=1, value=1, step=1)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Refresh UI"):
            st.cache_data.clear()
            st.rerun()
    with col_b:
        st.caption("Uses cached directory calls")


# ---------------- Cached directory lookups ----------------
@st.cache_data(ttl=10)
def search_entities(project_id: str, q: str = "", limit: int = 200) -> Dict[str, str]:
    """Returns {guid: 'Name (type)'} for selectboxes."""
    try:
        r = get("/kg/entities", params={"project_id": project_id, "q": q, "limit": limit}, timeout=20)
        if r.status_code == 200:
            items = r.json().get("items", [])
            out: Dict[str, str] = {}
            for e in items:
                guid = e.get("guid")
                if not guid:
                    continue
                name = e.get("name") or guid
                et = e.get("entity_type") or "Entity"
                out[guid] = f"{name} ({et})"
            return out
        st.warning(f"Entity directory error: HTTP {r.status_code}")
    except Exception as e:
        st.warning(f"Cannot reach Orchestrator (/kg/entities): {e}")
    return {}


@st.cache_data(ttl=30)
def get_rel_types() -> List[str]:
    """Authoritative relationship allowlist from the orchestrator."""
    try:
        r = get("/kg/rel_types", timeout=15)
        if r.status_code == 200:
            return r.json().get("items", [])
        # fallback to the other spelling
        r2 = get("/kg/rel-types", timeout=15)
        if r2.status_code == 200:
            return r2.json().get("items", [])
    except Exception:
        pass

    # No local fallback: domain is authoritative.
    return []


@st.cache_data(ttl=10)
def list_invariants() -> List[Dict[str, Any]]:
    try:
        r = get("/invariants/list", timeout=15)
        if r.status_code == 200:
            return r.json().get("items", [])
    except Exception:
        pass
    return []


# ---------------- Tabs ----------------
tabs = st.tabs([
    "Health",
    "Project Init",
    "World Zero Upsert",
    "Prose Memory",
    "Proposals",
    "Context Bundle + AI",
    "Logic Validator",
])


# ---------------- Tab 0: Health ----------------
with tabs[0]:
    st.subheader("Service health")
    if st.button("Run health check"):
        try:
            r = get("/health", timeout=10)
            st.write(f"HTTP {r.status_code}")
            if r.headers.get("content-type", "").startswith("application/json"):
                st.json(r.json())
            else:
                st.code(r.text)
        except Exception as e:
            st.error(f"Health check failed: {e}")


# ---------------- Tab 1: Project Init ----------------
with tabs[1]:
    st.subheader("Initialize a project")
    project_id = st.text_input("Project ID", ACTIVE_PROJECT)
    title = st.text_input("Title", "My Series")
    premise = st.text_area("Premise", "A hidden heir discovers a world-law that breaks the empire.")
    genre = st.text_input("Genre", "fantasy")

    if st.button("Create / Initialize"):
        r = post(
            "/projects/init",
            {"project_id": project_id, "title": title, "premise": premise, "genre": genre},
            timeout=30,
        )
        st.write(f"HTTP {r.status_code}")
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 2: World Zero Upsert ----------------
with tabs[2]:
    st.subheader("World Zero seed (YAML → /kg/upsert)")
    st.caption("This seeds your Knowledge Graph. After this, the UI can search entities by name.")

    default_yaml = f"""project_id: {ACTIVE_PROJECT}
characters:
  - guid: CHAR_ARAGORN
    name: Aragorn
    valid_from_chapter: 1
    reveal_tag: REVEALED
    relationships:
      # Aragorn learns the secret at Chapter 10
      - target: PLOT_TRUE_HEIR
        type: KNOWS
        from_chapter: 10

locations:
  - guid: LOC_MINAS_TIRITH
    name: Minas Tirith
    valid_from_chapter: 1

entities:
  - guid: PLOT_TRUE_HEIR
    name: The True Heir
    valid_from_chapter: 1
    reveal_tag: UNREVEALED
    reveal_from_chapter: 10
"""

    world_yaml = st.text_area("world_zero.yaml", default_yaml, height=320)

    if st.button("Upsert World Zero"):
        try:
            payload = yaml.safe_load(world_yaml)
        except Exception as e:
            st.error(f"Invalid YAML: {e}")
            st.stop()

        r = post("/kg/upsert", payload, timeout=60)
        st.write(f"HTTP {r.status_code}")
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 3: Prose Memory ----------------
with tabs[3]:
    st.subheader("Prose Memory (style continuity)")

    chapter_number = st.number_input("Chapter #", min_value=0, value=int(ACTIVE_CHAPTER), step=1)
    pov = st.text_input("POV label (optional)", "omniscient_neutral")
    label = st.text_input("Label", "ch1_opening")
    text = st.text_area("Paste a passage you love", "The rain came down like a verdict...", height=180)

    if st.button("Save snippet"):
        r = post(
            "/prose/upsert",
            {
                "project_id": ACTIVE_PROJECT,
                "chapter_number": int(chapter_number),
                "pov": pov,
                "label": label,
                "text": text,
            },
            timeout=60,
        )
        st.write(f"HTTP {r.status_code}")
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 4: Proposals ----------------
with tabs[4]:
    st.subheader("Proposals (author-friendly governance)")
    st.caption("Create or approve changes without writing JSON or memorizing GUIDs.")

    # Queue
    st.markdown("### Queue")
    status_filter = st.selectbox("Queue filter", ["PENDING", "APPROVED", "REJECTED"], index=0)
    try:
        r_list = get("/proposals/list", params={"project_id": ACTIVE_PROJECT, "status": status_filter}, timeout=20)
    except Exception as e:
        st.error(f"Cannot load proposals: {e}")
        r_list = None

    if r_list is not None:
        if r_list.status_code == 200:
            items = r_list.json().get("items", [])
            if not items:
                st.info("No proposals in this filter.")
            for item in items:
                with st.expander(f"Proposal #{item['id']} — {item['kind'].upper()} — {item['status']}"):
                    st.json(item.get("payload", {}))
                    if item["status"] == "PENDING":
                        if st.button(f"Approve #{item['id']}", key=f"approve_{item['id']}"):
                            rr = post("/proposals/approve", {"id": item["id"]}, timeout=30)
                            if rr.status_code == 200:
                                st.success("Approved.")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(f"Approve failed: HTTP {rr.status_code}")
                                st.code(rr.text)
        else:
            st.error(f"Failed to load proposals: HTTP {r_list.status_code}")
            st.code(r_list.text)

    st.divider()

    # Create
    st.markdown("### Create a proposal")
    mode = st.radio("What are you adding?", ["Entity", "Relationship"], horizontal=True)
    trust_tier = st.selectbox("Trust tier", [1, 2, 3], index=1, help="1=minor, 2=normal, 3=canon-shifting")

    if mode == "Entity":
        with st.form("entity_form"):
            e_name = st.text_input("Name", "")
            e_type = st.selectbox("Type", ["character", "location", "faction", "artifact", "entity"], index=0)
            e_guid = st.text_input("Internal ID (optional)", "", help="Leave blank to auto-generate from name.")
            e_desc = st.text_area("Description / notes (optional)", "")

            if st.form_submit_button("Submit entity"):
                if not e_name.strip():
                    st.error("Name is required.")
                    st.stop()

                raw = (e_guid or e_name).strip().upper()
                safe = re.sub(r"[^A-Z0-9_]+", "_", raw).strip("_")
                prefix = "CHAR" if e_type == "character" else e_type.upper()
                guid = safe if safe.startswith(prefix + "_") else f"{prefix}_{safe}"

                payload_obj = {
                    "guid": guid,
                    "name": e_name.strip(),
                    "entity_type": e_type,
                    "valid_from_chapter": int(ACTIVE_CHAPTER),
                    "properties": {"description": e_desc.strip()} if e_desc.strip() else {},
                    "relationships": [],
                }

                r = post(
                    "/proposals/create",
                    {
                        "project_id": ACTIVE_PROJECT,
                        "kind": "entity",
                        "trust_tier": int(trust_tier),
                        "payload": payload_obj,
                    },
                    timeout=30,
                )

                if r.status_code == 200:
                    st.success("Entity proposal submitted.")
                    st.json(r.json())
                    st.cache_data.clear()
                else:
                    st.error(f"Failed: HTTP {r.status_code}")
                    st.code(r.text)

    else:
        rel_types = get_rel_types()
        if not rel_types:
            st.error("Cannot load relationship types from the Orchestrator. Is it running and updated (v6.3)?")
            st.stop()

        with st.form("rel_form"):
            st.caption("Search entities by name.")

            q_src = st.text_input("Search source", "")
            src_map = search_entities(ACTIVE_PROJECT, q_src)
            if not src_map:
                st.warning("No entities found. Seed World Zero first (Tab 2).")
                st.stop()
            src = st.selectbox("Source", options=list(src_map.keys()), format_func=lambda x: src_map[x])

            rel_type = st.selectbox("Relationship type", rel_types)

            q_tgt = st.text_input("Search target", "")
            tgt_map = search_entities(ACTIVE_PROJECT, q_tgt)
            if not tgt_map:
                st.warning("No entities found for target search.")
                st.stop()
            tgt = st.selectbox("Target", options=list(tgt_map.keys()), format_func=lambda x: tgt_map[x])

            from_ch = st.number_input("From chapter", min_value=1, value=int(ACTIVE_CHAPTER), step=1)

            if st.form_submit_button("Submit relationship"):
                # Guardrail is author-side only; backend still enforces.
                if src == tgt and rel_type != "KNOWS":
                    st.error("Invalid: relationship cannot point to itself (except KNOWS).")
                    st.stop()

                payload_obj = {
                    "source_guid": src,
                    "target_guid": tgt,
                    "type": rel_type,
                    "from_chapter": int(from_ch),
                }

                r = post(
                    "/proposals/create",
                    {
                        "project_id": ACTIVE_PROJECT,
                        "kind": "relationship",
                        "trust_tier": int(trust_tier),
                        "payload": payload_obj,
                    },
                    timeout=30,
                )

                if r.status_code == 200:
                    st.success("Relationship proposal submitted.")
                    st.json(r.json())
                    st.cache_data.clear()
                else:
                    st.error(f"Failed: HTTP {r.status_code}")
                    st.code(r.text)


# ---------------- Tab 5: Context Bundle + AI ----------------
with tabs[5]:
    st.subheader("Context Bundle")
    st.caption("Strict mode filters spoilers based on KNOWS edges.")

    pov_style = st.selectbox("Narrative voice", ["omniscient_neutral", "deep_third", "first_person_past"], index=0)
    strict = st.checkbox("Strict POV knowledge (filter spoilers)", value=True)

    pov_guid = None
    if strict:
        q_pov = st.text_input("Search POV character", "")
        pov_map = search_entities(ACTIVE_PROJECT, q_pov)
        if pov_map:
            pov_guid = st.selectbox("POV character", options=list(pov_map.keys()), format_func=lambda x: pov_map[x])
        else:
            st.warning("No entities found. Seed World Zero first (Tab 2).")

    style_query = st.text_input("Optional style query override (usually blank)", "")

    if "bundle" not in st.session_state:
        st.session_state.bundle = None

    if st.button("Build context bundle"):
        if strict and not pov_guid:
            st.error("Strict mode requires selecting a POV character.")
            st.stop()

        payload = {
            "project_id": ACTIVE_PROJECT,
            "chapter_number": int(ACTIVE_CHAPTER),
            "pov": pov_style,
            "strict_epistemic": bool(strict),
            "style_query": style_query.strip() or None,
        }
        if strict:
            payload["pov_guid"] = pov_guid

        r = post("/context/bundle", payload, timeout=60)
        st.write(f"HTTP {r.status_code}")

        if r.headers.get("content-type", "").startswith("application/json"):
            st.session_state.bundle = r.json()
        else:
            st.session_state.bundle = None
            st.error("Non-JSON response from orchestrator")
            st.code(r.text)

    bundle = st.session_state.bundle
    if bundle:
        nodes = bundle.get("kg", {}).get("nodes", [])
        rels = bundle.get("kg", {}).get("relationships", [])
        name_map = {n.get("guid"): n.get("name", n.get("guid")) for n in nodes if n.get("guid")}

        st.success("Bundle ready")
        st.markdown("### Active relationships")
        if not rels:
            st.info("No relationships visible under this POV at this chapter.")
        for rel in rels:
            s = name_map.get(rel["source"], rel["source"])
            t = name_map.get(rel["target"], rel["target"])
            st.info(f"{s} ➔ {rel['type']} ➔ {t}")

        with st.expander("Raw bundle (debug)"):
            st.json(bundle)

    st.divider()

    # -------- AI Assist (Ollama draft, Claude review) --------
    st.subheader("AI Assist")
    st.caption("Draft locally with Ollama, then review with Claude. Both use the current Context Bundle.")

    if "draft_text" not in st.session_state:
        st.session_state.draft_text = ""
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    scene_prompt = st.text_area(
        "What should happen in this scene?",
        "Rowan crosses the city at night and senses something wrong in the air.",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Draft with Ollama"):
            if not bundle:
                st.error("Build a context bundle first.")
                st.stop()
            r = post(
                "/ai/ollama/draft",
                {"context": bundle, "prompt": scene_prompt, "temperature": 0.7},
                timeout=120,
            )
            if r.status_code == 200:
                st.session_state.draft_text = r.json().get("text", "")
            else:
                st.error(f"Ollama draft failed: HTTP {r.status_code}")
                st.code(r.text)

    with col2:
        if st.button("Review with Claude"):
            if not bundle:
                st.error("Build a context bundle first.")
                st.stop()
            if not st.session_state.draft_text.strip():
                st.error("Draft with Ollama first (or paste a draft below).")
                st.stop()
            r = post(
                "/ai/claude/review",
                {"context": bundle, "draft": st.session_state.draft_text, "instructions": "Find continuity issues, spoilers, and propose fixes."},
                timeout=120,
            )
            if r.status_code == 200:
                st.session_state.review_text = r.json().get("text", "")
            else:
                st.error(f"Claude review failed: HTTP {r.status_code}")
                st.code(r.text)

    st.markdown("### Draft")
    st.session_state.draft_text = st.text_area("Draft text", st.session_state.draft_text, height=220)

    st.markdown("### Review")
    st.text_area("Review notes", st.session_state.review_text, height=220)


# ---------------- Tab 6: Logic Validator ----------------
with tabs[6]:
    st.subheader("Logic Validator")
    st.caption("Use Simple Form mode to avoid JSON. Advanced mode remains available.")

    inv_items = list_invariants()
    inv_names = [i.get("name") for i in inv_items if i.get("name")]

    mode = st.radio("Mode", ["Simple Form", "Advanced JSON"], horizontal=True)

    # ----- Validator: Simple form -----
    if mode == "Simple Form":
        invariant_name = st.selectbox("Invariant", inv_names or [""], index=0)

        st.markdown("### Scene form")
        scene_type = st.selectbox("Scene type", ["combat"], index=0)

        # v6.3 scope: Combat scene
        weapon_drawn = st.checkbox("Weapon drawn", value=False)
        magic_used = st.checkbox("Magic used", value=False)
        character_injured = st.checkbox("Character injured", value=False)

        enemies_count = st.number_input("Enemies count", min_value=0, value=0, step=1)
        mana_spent = st.number_input("Mana spent", min_value=0, value=0, step=1)

        scene_obj = {
            "scene_type": scene_type,
            "weapon_drawn": bool(weapon_drawn),
            "magic_used": bool(magic_used),
            "character_injured": bool(character_injured),
            "enemies": int(enemies_count),
            "mana_spent": int(mana_spent),
        }

        with st.expander("Generated JSON (read-only)"):
            st.json(scene_obj)

        if st.button("Validate"):
            if not invariant_name:
                st.error("No invariants available. Add one below (Manage Invariants).")
                st.stop()
            r = post("/logic/validate", {"invariant_name": invariant_name, "scene": scene_obj}, timeout=30)
            st.write(f"HTTP {r.status_code}")
            st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})

    # ----- Validator: Advanced JSON -----
    else:
        invariant_name = st.text_input("Invariant name", "")
        scene_json = st.text_area("Scene JSON", '{"scene_type":"combat","weapon_drawn":true,"magic_used":false,"enemies":3,"mana_spent":0}', height=160)

        if st.button("Validate (Advanced)"):
            try:
                scene = json.loads(scene_json)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

            r = post("/logic/validate", {"invariant_name": invariant_name, "scene": scene}, timeout=30)
            st.write(f"HTTP {r.status_code}")
            st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})

    st.divider()

    # ----- Invariants editor (author-controlled) -----
    st.subheader("Manage invariants")
    st.caption("Create or update invariants from the UI. This writes to invariants.yaml in the Orchestrator container.")

    with st.expander("Create / Update invariant"):
        existing = {i.get("name"): i for i in inv_items if i.get("name")}
        pick = st.selectbox("Load existing (optional)", ["(new)"] + sorted(existing.keys()))

        seed = existing.get(pick) if pick and pick != "(new)" else {}

        inv_name = st.text_input("Name", seed.get("name", ""))
        inv_sev = st.selectbox("Severity", ["OK", "INFO", "WARN", "BLOCKER"], index=["OK", "INFO", "WARN", "BLOCKER"].index(seed.get("severity", "BLOCKER")) if seed.get("severity", "BLOCKER") in ["OK", "INFO", "WARN", "BLOCKER"] else 3)
        inv_rule = st.text_input("Rule expression (simple_eval)", seed.get("rule", ""), help="Return True to mark violated.")
        inv_why = st.text_area("Why (message)", seed.get("why", "Invariant violated."), height=80)
        inv_suggestions = st.text_area("Suggestions (one per line)", "\n".join(seed.get("suggestions", []) or []), height=120)

        variables_yaml = st.text_area(
            "Variables mapping (optional YAML)",
            yaml.safe_dump(seed.get("variables", {}) or {}, sort_keys=False) if seed else "{}\n",
            height=140,
            help="Example: teleport_used: ${scene.teleportation_used}",
        )

        if st.button("Save invariant"):
            if not inv_name.strip():
                st.error("Name is required.")
                st.stop()
            try:
                variables = yaml.safe_load(variables_yaml) or {}
                if not isinstance(variables, dict):
                    raise ValueError("Variables must be a YAML mapping")
            except Exception as e:
                st.error(f"Invalid variables YAML: {e}")
                st.stop()

            payload = {
                "name": inv_name.strip(),
                "invariant": {
                    "severity": inv_sev,
                    "variables": variables,
                    "rule": inv_rule.strip(),
                    "why": inv_why.strip(),
                    "suggestions": [s.strip() for s in inv_suggestions.splitlines() if s.strip()],
                },
            }

            r = post("/invariants/upsert", payload, timeout=30)
            if r.status_code == 200:
                st.success("Saved.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Save failed: HTTP {r.status_code}")
                st.code(r.text)
