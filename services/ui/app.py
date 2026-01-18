import os
import re
import json
from typing import Dict, List, Any, Optional

import httpx
import streamlit as st
import yaml

ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8000")

st.set_page_config(page_title="Bookster UI", layout="wide")
st.title("Bookster UI (v6.4)")

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

    if st.button("Refresh UI Data"):
        st.cache_data.clear()
        st.rerun()


# ---------------- Cached directories ----------------
@st.cache_data(ttl=10)
def rel_types() -> List[str]:
    try:
        r = get("/kg/rel_types", timeout=10)
        if r.status_code == 200:
            items = r.json().get("items", [])
            return sorted([str(x) for x in items])
    except Exception:
        pass
    # Fallback: should be rare; prefer domain allowlist from orchestrator.
    return ["KNOWS", "OWNS", "ALLY_OF", "ENEMY_OF", "LOCATED_IN", "MEMBER_OF", "HAS_TITLE"]


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
        st.warning(f"Cannot reach Orchestrator: {e}")
    return {}


@st.cache_data(ttl=10)
def invariants_list(project_id: str) -> Dict[str, Any]:
    try:
        r = get("/invariants/list", params={"project_id": project_id}, timeout=20)
        if r.status_code == 200:
            return r.json().get("invariants", {})
    except Exception:
        pass
    return {}


# ---------------- Tabs ----------------

tabs = st.tabs([
    "Health",
    "Project Init",
    "World Zero Upsert",
    "Prose Memory",
    "Proposals",
    "Context Bundle",
    "Logic Validator",
])


# ---------------- Tab 0: Health ----------------
with tabs[0]:
    st.subheader("Health")
    if st.button("Health Check"):
        try:
            r = get("/health", timeout=10)
            st.write(r.status_code)
            st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})
        except Exception as e:
            st.error(f"Health check failed: {e}")


# ---------------- Tab 1: Project Init ----------------
with tabs[1]:
    st.subheader("Initialize Project")

    project_id = st.text_input("Project ID", ACTIVE_PROJECT)
    title = st.text_input("Title", "My Series")
    premise = st.text_area("Premise", "A hidden heir discovers a world-law that breaks the empire.")
    genre = st.text_input("Genre", "fantasy")

    if st.button("Create Project"):
        r = post("/projects/init", {"project_id": project_id, "title": title, "premise": premise, "genre": genre}, timeout=30)
        st.write(r.status_code)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 2: World Zero Upsert ----------------
with tabs[2]:
    st.subheader("World Zero Upsert (YAML → KG)")
    st.caption("Paste YAML. The UI converts it into JSON for /kg/upsert.")

    default_yaml = f"""project_id: {ACTIVE_PROJECT}

characters:
  - guid: CHAR_TEST_POV
    name: Test POV character
    valid_from_chapter: 1
    reveal_tag: REVEALED

locations:
  - guid: LOC_HOME
    name: Dream Park
    valid_from_chapter: 1

entities:
  - guid: PLOT_SECRET
    name: Placeholder Secret
    valid_from_chapter: 1
    reveal_tag: UNREVEALED
    reveal_from_chapter: 10

# Knowledge edge direction: POV KNOWS plot point
relationships:
  - source_guid: CHAR_TEST_POV
    target_guid: PLOT_SECRET
    type: KNOWS
    from_chapter: 10
"""

    world_yaml = st.text_area("world_zero.yaml", default_yaml, height=320)

    if st.button("Upsert KG"):
        try:
            data = yaml.safe_load(world_yaml) or {}
        except Exception as e:
            st.error(f"Invalid YAML: {e}")
            st.stop()

        r = post("/kg/upsert", data, timeout=90)
        st.write(r.status_code)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 3: Prose Memory ----------------
with tabs[3]:
    st.subheader("Prose Memory (style continuity)")

    project_id = st.text_input("Project ID (prose)", ACTIVE_PROJECT)
    chapter_number = st.number_input("Chapter #", min_value=0, value=int(ACTIVE_CHAPTER), step=1)
    pov = st.text_input("POV label", "omniscient_neutral")
    label = st.text_input("Label", "ch1_opening")
    text = st.text_area("Prose excerpt", "The rain came down like a verdict...", height=180)

    if st.button("Upsert Prose Snippet"):
        r = post(
            "/prose/upsert",
            {
                "project_id": project_id,
                "chapter_number": int(chapter_number),
                "pov": pov,
                "label": label,
                "text": text,
            },
            timeout=60,
        )
        st.write(r.status_code)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 4: Proposals ----------------
with tabs[4]:
    st.subheader("Proposals (author-friendly)")
    st.caption("Add facts without writing JSON or memorizing GUIDs.")

    # Queue
    st.markdown("### Queue")
    status_filter = st.selectbox("Queue filter", ["PENDING", "APPROVED", "REJECTED"], index=0)
    r_list = get("/proposals/list", params={"project_id": ACTIVE_PROJECT, "status": status_filter}, timeout=30)

    if r_list.status_code == 200:
        items = r_list.json().get("items", [])
        if not items:
            st.info("No proposals in this filter.")
        for item in items:
            with st.expander(f"Proposal #{item['id']} — {item['kind'].upper()} — {item['status']}"):
                st.json(item["payload"])
                if item["status"] == "PENDING":
                    if st.button(f"Approve #{item['id']}", key=f"approve_{item['id']}"):
                        r = post("/proposals/approve", {"id": item["id"]}, timeout=30)
                        if r.status_code == 200:
                            st.success("Approved.")
                            st.rerun()
                        else:
                            st.error(f"Approve failed: {r.status_code}")
                            st.write(r.text)
    else:
        st.error(f"Failed to load proposals: {r_list.status_code}")
        st.write(r_list.text)

    st.divider()

    st.markdown("### Create a new proposal")
    mode = st.radio("What are you adding?", ["Entity", "Relationship"], horizontal=True)
    trust_tier = st.selectbox("Trust tier", [1, 2, 3], index=1, help="1=minor, 2=normal, 3=canon-shifting")

    if mode == "Entity":
        with st.form("entity_form"):
            e_name = st.text_input("Name")
            e_type = st.selectbox("Type", ["character", "location", "faction", "artifact", "entity"], index=0)
            e_guid = st.text_input("Internal ID (optional)", help="Leave blank to auto-generate from name")
            e_desc = st.text_area("Description/notes (optional)")
            reveal_tag = st.selectbox("Reveal tag", ["REVEALED", "UNREVEALED"], index=0)
            reveal_from = st.number_input("Reveal from chapter (optional)", min_value=1, value=int(ACTIVE_CHAPTER), step=1)

            if st.form_submit_button("Submit entity proposal"):
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
                    "reveal_tag": reveal_tag,
                    "reveal_from_chapter": int(reveal_from) if reveal_tag == "UNREVEALED" else None,
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
                else:
                    st.error(f"Failed: {r.status_code}")
                    st.write(r.text)

    else:
        with st.form("rel_form"):
            st.caption("Search entities by name; the UI uses the KG directory.")

            q_src = st.text_input("Search source")
            src_map = search_entities(ACTIVE_PROJECT, q_src)
            if not src_map:
                st.warning("No entities found. Seed World Zero first (Tab 2).")
                st.stop()
            src = st.selectbox("Source", options=list(src_map.keys()), format_func=lambda x: src_map[x])

            rel_type = st.selectbox("Relationship type", rel_types())

            q_tgt = st.text_input("Search target")
            tgt_map = search_entities(ACTIVE_PROJECT, q_tgt)
            if not tgt_map:
                st.warning("No entities found for target search.")
                st.stop()
            tgt = st.selectbox("Target", options=list(tgt_map.keys()), format_func=lambda x: tgt_map[x])

            from_ch = st.number_input("From chapter", min_value=1, value=int(ACTIVE_CHAPTER), step=1)

            if st.form_submit_button("Submit relationship proposal"):
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
                else:
                    st.error(f"Failed: {r.status_code}")
                    st.write(r.text)


# ---------------- Tab 5: Context Bundle (+ AI buttons) ----------------
with tabs[5]:
    st.subheader("Context Bundle")
    st.caption("Strict mode filters spoilers based on outgoing KNOWS edges from the POV.")

    pov_style = st.selectbox("Narrative voice", ["omniscient_neutral", "deep_third", "first_person_past"], index=0)
    strict = st.checkbox("Strict POV knowledge (filter spoilers)", value=True)

    pov_guid = None
    if strict:
        q_pov = st.text_input("Search POV character")
        pov_map = search_entities(ACTIVE_PROJECT, q_pov)
        if not pov_map:
            st.warning("No entities found. Seed World Zero first (Tab 2).")
        else:
            pov_guid = st.selectbox("POV character", options=list(pov_map.keys()), format_func=lambda x: pov_map[x])

    style_query = st.text_input("Optional style query override (leave blank)")

    bundle: Optional[dict] = None
    if st.button("Build bundle"):
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
            bundle = r.json()
            st.session_state["last_bundle"] = bundle
        else:
            st.error("Non-JSON response from orchestrator")
            st.write(r.text)

    bundle = st.session_state.get("last_bundle")
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

        with st.expander("AI Studio (Ollama draft + Claude review)"):
            st.caption("Ollama runs on your Mac. Claude needs an Anthropic API key in the orchestrator container.")

            user_brief = st.text_area(
                "What should the AI write next?",
                "Write the next scene in this chapter.",
                height=90,
            )

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("Draft with Ollama"):
                    r = post(
                        "/ai/ollama/generate",
                        {
                            "project_id": ACTIVE_PROJECT,
                            "chapter_number": int(ACTIVE_CHAPTER),
                            "bundle": bundle,
                            "instruction": user_brief,
                        },
                        timeout=120,
                    )
                    if r.status_code == 200:
                        st.session_state["ai_draft"] = r.json().get("text", "")
                    else:
                        st.error(f"Ollama error: HTTP {r.status_code}")
                        st.write(r.text)

            with col_b:
                if st.button("Review with Claude"):
                    draft = st.session_state.get("ai_draft", "")
                    if not draft.strip():
                        st.error("No draft to review yet. Click 'Draft with Ollama' first (or paste your draft below).")
                    else:
                        r = post(
                            "/ai/claude/review",
                            {
                                "project_id": ACTIVE_PROJECT,
                                "chapter_number": int(ACTIVE_CHAPTER),
                                "bundle": bundle,
                                "draft_text": draft,
                                "instruction": "Review for canon/logic/POV epistemics. Provide fixes and a revised version.",
                            },
                            timeout=120,
                        )
                        if r.status_code == 200:
                            st.session_state["ai_review"] = r.json().get("text", "")
                        else:
                            st.error(f"Claude error: HTTP {r.status_code}")
                            st.write(r.text)

            draft = st.text_area("Draft (editable)", st.session_state.get("ai_draft", ""), height=220)
            st.session_state["ai_draft"] = draft

            review = st.text_area("Claude review (read-only)", st.session_state.get("ai_review", ""), height=220)
            st.session_state["ai_review"] = review

        with st.expander("Raw bundle (debug)"):
            st.json(bundle)


# ---------------- Tab 6: Logic Validator (simple forms + invariant editor) ----------------
with tabs[6]:
    st.subheader("Logic Validator")
    st.caption("Validate invariants without writing JSON. Use 'Raw mode' only when needed.")

    invs = invariants_list(ACTIVE_PROJECT)
    inv_names = sorted(list(invs.keys())) if invs else []

    col_left, col_right = st.columns([2, 1])

    with col_left:
        mode = st.radio("Mode", ["Simple form", "Raw JSON"], horizontal=True)

    with col_right:
        st.markdown("**Invariant**")
        invariant_name = st.selectbox("", inv_names, index=0 if inv_names else None, placeholder="No invariants found")

    # Simple-form scene builders
    scene: Optional[Dict[str, Any]] = None

    if mode == "Simple form":
        st.markdown("### Scene Type: Combat")
        col1, col2, col3 = st.columns(3)
        with col1:
            weapon_drawn = st.checkbox("Weapon drawn")
            magic_used = st.checkbox("Magic used")
        with col2:
            character_injured = st.checkbox("Character injured")
            enemies = st.number_input("Enemies count", min_value=0, value=0, step=1)
        with col3:
            mana_spent = st.number_input("Mana spent", min_value=0, value=0, step=1)

        scene = {
            "scene_type": "combat",
            "weapon_drawn": bool(weapon_drawn),
            "magic_used": bool(magic_used),
            "character_injured": bool(character_injured),
            "enemies": int(enemies),
            "mana_spent": int(mana_spent),
        }

        st.markdown("#### Generated JSON (hidden in normal use)")
        st.code(json.dumps(scene, indent=2), language="json")

    else:
        scene_json = st.text_area("Scene JSON", '{"scene_type": "combat", "magic_used": true}', height=160)
        if st.button("Parse JSON"):
            try:
                scene = json.loads(scene_json)
                st.success("Parsed.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    if st.button("Validate"):
        if not invariant_name:
            st.error("No invariant selected.")
            st.stop()
        if scene is None:
            st.error("No scene data.")
            st.stop()

        r = post("/logic/validate", {"project_id": ACTIVE_PROJECT, "invariant_name": invariant_name, "scene": scene}, timeout=30)
        st.write(r.status_code)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})

    st.divider()
    st.subheader("Edit invariants from the UI")
    st.caption("This writes to a project-local runtime invariants file (does not overwrite your base invariants.yaml).")

    with st.form("inv_editor"):
        inv_key = st.text_input("Invariant key", "combat_mana_limit")
        severity = st.selectbox("Severity", ["OK", "WARN", "BLOCKER"], index=2)
        why = st.text_input("Why (message)", "Mana spent exceeds allowed limit.")
        rule = st.text_input("Rule expression (true means violated)", "mana_spent > max_mana")

        st.markdown("**Variables (YAML).** Use ${scene.<field>} references.")
        vars_yaml = st.text_area(
            "",
            """max_mana: 50\nmana_spent: ${scene.mana_spent}\n""",
            height=110,
        )

        suggestions_text = st.text_area("Suggestions (one per line)", "Reduce mana_spent\nAdd training scene earlier")

        if st.form_submit_button("Save / update invariant"):
            try:
                variables = yaml.safe_load(vars_yaml) or {}
                if not isinstance(variables, dict):
                    raise ValueError("Variables must be a YAML mapping")
            except Exception as e:
                st.error(f"Invalid variables YAML: {e}")
                st.stop()

            inv_obj = {
                "severity": severity,
                "why": why,
                "variables": variables,
                "rule": rule,
                "suggestions": [s.strip() for s in suggestions_text.splitlines() if s.strip()],
            }

            r = post("/invariants/upsert", {"project_id": ACTIVE_PROJECT, "name": inv_key, "invariant": inv_obj}, timeout=30)
            if r.status_code == 200:
                st.success("Invariant saved.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Save failed: HTTP {r.status_code}")
                st.write(r.text)
