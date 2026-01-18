import os
import re
import json
from typing import Dict, List

import httpx
import streamlit as st
import yaml

ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8000")

st.set_page_config(page_title="Bookster UI", layout="wide")
st.title("Bookster UI (v6.4)")


def post(path: str, payload: dict, timeout: int = 30) -> httpx.Response:
    return httpx.post(f"{ORCH_URL}{path}", json=payload, timeout=timeout)


def get(path: str, params: dict | None = None, timeout: int = 30) -> httpx.Response:
    return httpx.get(f"{ORCH_URL}{path}", params=params or {}, timeout=timeout)


# Sidebar globals (used across all 7 tabs)
with st.sidebar:
    st.header("Project Settings")
    ACTIVE_PROJECT = st.text_input("Active Project ID", "my_series")
    ACTIVE_CHAPTER = st.number_input("Current Chapter", min_value=1, value=1, step=1)
    if st.button("Refresh UI Data"):
        st.cache_data.clear()
        st.rerun()


@st.cache_data(ttl=10)
def search_entities(project_id: str, q: str = "") -> Dict[str, str]:
    """Returns {guid: 'Name (type)'} for selectboxes via /kg/entities."""
    try:
        r = get("/kg/entities", params={"project_id": project_id, "q": q, "limit": 200}, timeout=30)
        if r.status_code == 200:
            items = r.json().get("items", [])
            return {
                e["guid"]: f"{e.get('name', e['guid'])} ({e.get('entity_type', 'entity')})"
                for e in items
                if e.get("guid")
            }
        st.warning(f"Entity directory error: HTTP {r.status_code}")
    except Exception as e:
        st.warning(f"Cannot reach Orchestrator: {e}")
    return {}


@st.cache_data(ttl=30)
def get_rel_types() -> List[str]:
    """
    Fetch relationship types from orchestrator (/kg/rel_types).
    This avoids any hard-coded allowlist in the UI.
    """
    try:
        r = get("/kg/rel_types", timeout=30)
        if r.status_code == 200:
            items = r.json().get("items", [])
            # defensive: ensure list[str]
            return sorted([x for x in items if isinstance(x, str)])
        st.warning(f"Rel types error: HTTP {r.status_code}")
    except Exception as e:
        st.warning(f"Cannot fetch relationship types: {e}")
    return []


tabs = st.tabs(
    [
        "Health",
        "Project Init",
        "World Zero Upsert",
        "Prose Memory",
        "Proposals",
        "Context Bundle",
        "Logic Validator",
    ]
)


# ---------------- Tab 0: Health ----------------
with tabs[0]:
    if st.button("Health Check"):
        try:
            r = get("/health")
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
        r = post("/projects/init", {"project_id": project_id, "title": title, "premise": premise, "genre": genre})
        st.write(r.status_code)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 2: World Zero Upsert ----------------
with tabs[2]:
    st.subheader("World Zero Upsert (YAML → /kg/upsert)")
    st.caption("Tip: define entities first, then relationships. If you include relationships in the same YAML, the server will still enforce that both endpoints exist.")

    default_yaml = f"""project_id: {ACTIVE_PROJECT}

characters:
  - guid: CHAR_ROWAN
    name: Rowan
    valid_from_chapter: 1
    reveal_tag: REVEALED

locations:
  - guid: LOC_HOME
    name: Home
    valid_from_chapter: 1

entities:
  - guid: PLOT_SECRET_ORIGIN
    name: Rowan's origin secret
    valid_from_chapter: 1
    reveal_tag: UNREVEALED
    reveal_from_chapter: 8
"""

    world_yaml = st.text_area("world_zero.yaml", default_yaml, height=320)

    if st.button("Upsert KG"):
        try:
            payload = yaml.safe_load(world_yaml)
        except Exception as e:
            st.error(f"Invalid YAML: {e}")
            st.stop()

        r = post("/kg/upsert", payload, timeout=60)
        st.write(r.status_code)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})


# ---------------- Tab 3: Prose Memory ----------------
with tabs[3]:
    st.subheader("Prose Memory (store excerpts for style continuity)")
    project_id = st.text_input("Project ID (prose)", ACTIVE_PROJECT)
    chapter_number = st.number_input("Chapter #", min_value=0, value=int(ACTIVE_CHAPTER), step=1)
    pov = st.text_input("POV label", "omniscient_neutral")
    label = st.text_input("Label", "ch1_opening")
    text = st.text_area("Prose text to store", "The rain came down like a verdict...", height=180)

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
    st.subheader("⚖️ Proposals (Author-Friendly)")
    st.caption("Add facts to your world without writing JSON or memorizing GUIDs.")

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

    # Create proposals via forms
    st.markdown("### Create a new proposal")
    mode = st.radio("What are you adding?", ["Entity", "Relationship"], horizontal=True)
    trust_tier = st.selectbox("Trust tier", [1, 2, 3], index=1, help="1=minor, 2=normal, 3=canon-shifting")

    if mode == "Entity":
        with st.form("entity_form"):
            e_name = st.text_input("Name", "")
            e_type = st.selectbox("Type", ["character", "location", "faction", "artifact", "entity"], index=0)
            e_guid = st.text_input("Internal ID (optional)", "", help="Leave blank to auto-generate from name.")
            e_desc = st.text_area("Description / notes (optional)", "")

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
            st.caption("Search entities by name. Relationship types are fetched from the Orchestrator.")

            q_src = st.text_input("Search source", "")
            src_map = search_entities(ACTIVE_PROJECT, q_src)
            if not src_map:
                st.warning("No entities found. Seed World Zero first (Tab 2).")
                st.stop()
            src = st.selectbox("Source", options=list(src_map.keys()), format_func=lambda x: src_map[x])

            rel_types = get_rel_types()
            if not rel_types:
                st.warning("No relationship types available. Is the orchestrator running?")
                st.stop()

            rel_type = st.selectbox("Relationship type", rel_types)


            q_tgt = st.text_input("Search target", "")
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


# ---------------- Tab 5: Context Bundle ----------------
with tabs[5]:
    st.subheader("📖 Context Bundle")
    st.caption("Generates the context your AI writer should see. Strict mode filters spoilers based on KNOWS edges.")

    # Persist the last bundle so AI tools can use it
    if "last_bundle" not in st.session_state:
        st.session_state["last_bundle"] = None

    pov_style = st.selectbox(
        "Narrative voice",
        ["omniscient_neutral", "deep_third", "first_person_past"],
        index=0,
        key="ctx_pov_style",
    )
    strict = st.checkbox("Strict POV knowledge (filter spoilers)", value=True, key="ctx_strict")

    pov_guid = None
    if strict:
        q_pov = st.text_input("Search POV character", "", key="ctx_pov_search")
        pov_map = search_entities(ACTIVE_PROJECT, q_pov)
        if not pov_map:
            st.warning("No entities found. Seed World Zero first (Tab 2).")
        else:
            pov_guid = st.selectbox(
                "POV character",
                options=list(pov_map.keys()),
                format_func=lambda x: pov_map[x],
                key="ctx_pov_guid",
            )

    style_query = st.text_input("Optional style query override (usually leave blank)", "", key="ctx_style_query")

    if st.button("Build bundle", key="ctx_build"):
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

            # GUID -> name mapping for author readability
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
        else:
            st.error("Non-JSON response from orchestrator:")
            st.write(r.text)

    st.divider()
    st.subheader("AI Assist")
    st.caption("Draft with local Ollama; then review with Claude (optional).")

    bundle = st.session_state.get("last_bundle")
    if not bundle:
        st.info("Build a Context Bundle first, then the AI tools can use the same context.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Draft a scene with Ollama")
        scene_brief = st.text_area(
            "Scene brief (what happens)",
            "Rowan enters Dream Park at dusk. He senses something is wrong.",
            height=140,
            key="ai_scene_brief",
        )
        if st.button("Draft with Ollama", key="ai_ollama"):
            if not bundle:
                st.error("Please build a Context Bundle first.")
            else:
                rr = post(
                    "/ai/ollama/draft",
                    {
                        "project_id": ACTIVE_PROJECT,
                        "chapter_number": int(ACTIVE_CHAPTER),
                        "bundle": bundle,
                        "scene_brief": scene_brief,
                    },
                    timeout=180,
                )
                if rr.status_code == 200 and rr.headers.get("content-type", "").startswith("application/json"):
                    st.session_state["ai_last_draft"] = rr.json().get("draft", "")
                    st.success("Draft generated.")
                else:
                    st.error(f"Ollama draft failed: HTTP {rr.status_code}")
                    st.write(rr.text)

        st.text_area(
            "Latest draft (editable)",
            value=st.session_state.get("ai_last_draft", ""),
            height=260,
            key="ai_draft_text",
        )

    with col2:
        st.markdown("#### Review with Claude")
        review_focus = st.text_input(
            "Review focus (optional)",
            "Continuity, POV leaks, pacing, and contradictions.",
            key="ai_review_focus",
        )
        if st.button("Review with Claude", key="ai_claude"):
            if not bundle:
                st.error("Please build a Context Bundle first.")
            else:
                draft_text = st.session_state.get("ai_draft_text", "")
                rr = post(
                    "/ai/claude/review",
                    {
                        "project_id": ACTIVE_PROJECT,
                        "chapter_number": int(ACTIVE_CHAPTER),
                        "bundle": bundle,
                        "draft": draft_text,
                        "focus": review_focus,
                    },
                    timeout=180,
                )
                if rr.status_code == 200 and rr.headers.get("content-type", "").startswith("application/json"):
                    st.session_state["ai_last_review"] = rr.json().get("review", "")
                    st.success("Review completed.")
                else:
                    st.error(f"Claude review failed: HTTP {rr.status_code}")
                    st.write(rr.text)

        st.text_area(
            "Review notes",
            value=st.session_state.get("ai_last_review", ""),
            height=420,
            key="ai_review_text",
        )
with tabs[6]:
    st.subheader("Logic Validator")
    invariant = st.text_input("Invariant name", "teleportation_allowed")

    st.markdown("#### Simple form mode")
    st.caption("Use this for common invariants. The UI generates the JSON for you.")

    scene_type = st.selectbox("Scene type", ["custom", "combat"], index=0)

    if scene_type == "combat":
        weapon_drawn = st.checkbox("Weapon drawn", value=False)
        magic_used = st.checkbox("Magic used", value=False)
        character_injured = st.checkbox("Character injured", value=False)
        enemies = st.number_input("Enemies count", min_value=0, value=0, step=1)
        mana_spent = st.number_input("Mana spent", min_value=0, value=0, step=1)

        generated_scene = {
            "scene_type": "combat",
            "weapon_drawn": bool(weapon_drawn),
            "magic_used": bool(magic_used),
            "character_injured": bool(character_injured),
            "enemies": int(enemies),
            "mana_spent": int(mana_spent),
        }

        st.code(json.dumps(generated_scene, indent=2), language="json")
        scene_payload = generated_scene

    else:
        st.markdown("#### Advanced mode (raw JSON)")
        scene_json = st.text_area("Scene JSON", '{"teleportation_used": true}', height=120)
        try:
            scene_payload = json.loads(scene_json)
        except Exception:
            scene_payload = None

    if st.button("Validate"):
        if scene_payload is None:
            st.error("Invalid Scene JSON")
            st.stop()
        r = post("/logic/validate", {"invariant_name": invariant, "scene": scene_payload}, timeout=30)
        st.write(r.status_code)
        st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})
