import os
import re
import json
from typing import Dict, List, Optional

import httpx
import streamlit as st
import yaml

ORCH_URL = os.getenv("ORCH_URL", "http://orchestrator:8000")

st.set_page_config(page_title="Bookster UI", layout="wide")
st.title("Bookster UI (v6.2)")

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


# ---------------- Cached directory lookups ----------------
@st.cache_data(ttl=10)
def search_entities(project_id: str, q: str = "") -> Dict[str, str]:
    """Returns {guid: 'Name (type)'} for selectboxes."""
    q = (q or "").strip()
    try:
        r = get("/kg/entities", params={"project_id": project_id, "q": q, "limit": 200}, timeout=30)
        if r.status_code != 200:
            st.warning(f"Entity directory error: HTTP {r.status_code}")
            return {}
        items = r.json().get("items", [])
        out: Dict[str, str] = {}
        for e in items:
            guid = e.get("guid")
            if not guid:
                continue
            name = e.get("name") or guid
            etype = e.get("entity_type") or "Entity"
            out[guid] = f"{name} ({etype})"
        return out
    except Exception as e:
        st.warning(f"Cannot reach Orchestrator: {e}")
        return {}


@st.cache_data(ttl=60)
def get_rel_types() -> List[str]:
    """Relationship allowlist provided by orchestrator (single source of truth)."""
    try:
        r = get("/kg/rel-types", timeout=30)
        if r.status_code != 200:
            st.warning(f"Relationship allowlist error: HTTP {r.status_code}")
            return ["KNOWS"]
        items = r.json().get("items", [])
        return [str(x) for x in items]
    except Exception as e:
        st.warning(f"Cannot reach Orchestrator for relationship types: {e}")
        return ["KNOWS"]


REL_TYPES = get_rel_types()


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
            r = get("/health")
            st.write(r.status_code)
            if r.headers.get("content-type", "").startswith("application/json"):
                st.json(r.json())
            else:
                st.code(r.text)
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
    st.caption("Seed your initial world bible. Tip: keep secrets as UNREVEALED until the reveal chapter.")

    default_yaml = f"""project_id: {ACTIVE_PROJECT}
characters:
  - guid: CHAR_ARAGORN
    name: Aragorn
    valid_from_chapter: 1
    reveal_tag: REVEALED
    relationships:
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
    st.subheader("Prose Memory (Style Continuity)")
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
    st.subheader("Proposals (Author-Friendly)")
    st.caption("Add facts without typing JSON or memorizing GUIDs.")

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

    # Create proposals
    st.markdown("### Create a new proposal")
    mode = st.radio("What are you adding?", ["Entity", "Relationship"], horizontal=True)
    trust_tier = st.selectbox("Trust tier", [1, 2, 3], index=1, help="1=minor, 2=normal, 3=canon-shifting")

    if mode == "Entity":
        with st.form("entity_form"):
            e_name = st.text_input("Name")
            e_type = st.selectbox("Type", ["character", "location", "faction", "artifact", "entity"], index=0)
            e_guid = st.text_input("Internal ID (optional)", help="Leave blank to auto-generate from name.")
            e_desc = st.text_area("Description / notes (optional)")

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
            st.caption("Search entities by name. The dropdown is powered by your knowledge graph.")

            q_src = st.text_input("Search source")
            src_map = search_entities(ACTIVE_PROJECT, q_src)
            if not src_map:
                st.warning("No entities found. Seed World Zero first (Tab 3).")
                st.stop()
            src = st.selectbox("Source", options=list(src_map.keys()), format_func=lambda x: src_map[x])

            rel_type = st.selectbox("Relationship type", REL_TYPES)

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


# ---------------- Tab 5: Context Bundle ----------------
with tabs[5]:
    st.subheader("Context Bundle")
    st.caption("Strict mode filters spoilers based on KNOWS edges + reveal tags.")

    pov_style = st.selectbox("Narrative voice", ["omniscient_neutral", "deep_third", "first_person_past"], index=0)
    strict = st.checkbox("Strict POV knowledge (filter spoilers)", value=True)

    pov_guid = None
    if strict:
        q_pov = st.text_input("Search POV character")
        pov_map = search_entities(ACTIVE_PROJECT, q_pov)
        if not pov_map:
            st.warning("No entities found. Seed World Zero first (Tab 3).")
        else:
            pov_guid = st.selectbox("POV character", options=list(pov_map.keys()), format_func=lambda x: pov_map[x])

    style_query = st.text_input("Optional style query override (usually leave blank)")

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


# ---------------- Tab 6: Logic Validator + Invariant Editor ----------------
with tabs[6]:
    st.subheader("🛡️ Logic & Rules")

    sub = st.tabs(["Validate a Scene", "Edit Invariants (No Code)"])

    # ------- Subtab: Validate -------
    with sub[0]:
        st.caption("Simple Form mode avoids JSON for common checks. Advanced mode accepts raw Scene JSON.")

        # You can expand this map over time. These are UI-only helpers; the *real* invariant lives in orchestrator YAML.
        COMMON_INVARIANTS = {
            "teleportation_allowed": {
                "teleportation_used": ("bool", False),
            },
        }

        mode = st.radio("Input mode", ["Simple Form", "Advanced JSON"], horizontal=True)
        invariant = st.text_input("Invariant name", "teleportation_allowed")

        scene: Optional[dict] = None

        if mode == "Simple Form":
            spec = COMMON_INVARIANTS.get(invariant)
            if not spec:
                st.info(
                    "No simple form is configured for this invariant yet. "
                    "Use Advanced JSON, or add a form template in app.py (COMMON_INVARIANTS)."
                )
            else:
                st.markdown("### Scene inputs")
                scene = {}
                for field, (ftype, default) in spec.items():
                    if ftype == "bool":
                        scene[field] = st.checkbox(field, value=bool(default), key=f"inv_{invariant}_{field}")
                    elif ftype == "int":
                        scene[field] = int(st.number_input(field, value=int(default), step=1, key=f"inv_{invariant}_{field}"))
                    elif ftype == "float":
                        scene[field] = float(st.number_input(field, value=float(default), step=0.1, key=f"inv_{invariant}_{field}"))
                    elif ftype == "str":
                        scene[field] = st.text_input(field, value=str(default), key=f"inv_{invariant}_{field}")
                    else:
                        st.warning(f"Unsupported field type '{ftype}' for '{field}'.")

                with st.expander("Show generated Scene JSON"):
                    st.code(json.dumps(scene, indent=2), language="json")

            if st.button("Validate", key="validate_form"):
                if scene is None:
                    st.error("No form available for this invariant. Switch to Advanced JSON.")
                    st.stop()
                r = post("/logic/validate", {"invariant_name": invariant, "scene": scene}, timeout=30)
                st.write(r.status_code)
                st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})

        else:
            scene_json = st.text_area("Scene JSON", '{"teleportation_used": true}', height=160)
            if st.button("Validate", key="validate_json"):
                try:
                    scene = json.loads(scene_json)
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    st.stop()
                r = post("/logic/validate", {"invariant_name": invariant, "scene": scene}, timeout=30)
                st.write(r.status_code)
                st.json(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})

    # ------- Subtab: Edit invariants -------
    with sub[1]:
        st.caption(
            "Create or edit invariants from the UI. These are stored in a shared file and reloaded by the orchestrator."
        )

        CUSTOM_INVARIANTS_PATH = os.getenv("CUSTOM_INVARIANTS_PATH", "/shared/invariants.custom.yaml")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Reload invariants in Orchestrator"):
                r = post("/admin/reload-invariants", {}, timeout=30)
                if r.status_code == 200:
                    st.success("Reloaded.")
                    st.json(r.json())
                else:
                    st.error(f"Reload failed: HTTP {r.status_code}")
                    st.write(r.text)

        with col_b:
            if st.button("View current invariants (merged)"):
                r = get("/admin/invariants", timeout=30)
                if r.status_code == 200:
                    invs = r.json().get("invariants", {})
                    st.write(f"Loaded invariants: {len(invs)}")
                    with st.expander("Show all invariants"):
                        st.json(invs)
                else:
                    st.error(f"Cannot fetch invariants: HTTP {r.status_code}")
                    st.write(r.text)

        st.divider()

        def _load_custom() -> dict:
            try:
                if os.path.exists(CUSTOM_INVARIANTS_PATH):
                    with open(CUSTOM_INVARIANTS_PATH, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f.read()) or {}
                        return data
            except Exception as e:
                st.warning(f"Could not read custom invariants: {e}")
            return {}

        def _save_custom(data: dict) -> None:
            os.makedirs(os.path.dirname(CUSTOM_INVARIANTS_PATH), exist_ok=True)
            with open(CUSTOM_INVARIANTS_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        custom_doc = _load_custom()
        custom_invs = (custom_doc or {}).get("invariants", {}) or {}

        st.markdown("### Edit custom invariants")
        existing = [""] + sorted(custom_invs.keys())
        pick = st.selectbox("Edit existing (optional)", options=existing)

        seed = custom_invs.get(pick, {}) if pick else {}

        with st.form("inv_editor"):
            inv_name = st.text_input("Invariant name", value=pick or "")
            severity = st.selectbox("Severity", ["OK", "INFO", "WARN", "BLOCKER"], index=3)
            rule = st.text_input("Rule (expression)", value=str(seed.get("rule", "")))
            why = st.text_area("Why (message shown when violated)", value=str(seed.get("why", "")), height=80)

            variables_yaml = st.text_area(
                "Variables (YAML map)",
                value=yaml.safe_dump(seed.get("variables", {}) or {}, sort_keys=False),
                height=140,
                help="Example: mana_spent: ${scene.mana_spent}"
            )
            suggestions_txt = st.text_area(
                "Suggestions (one per line)",
                value="\n".join(seed.get("suggestions", []) or []),
                height=100,
            )

            save = st.form_submit_button("Save to custom invariants")

        if save:
            name = (inv_name or "").strip()
            if not name:
                st.error("Invariant name is required.")
                st.stop()
            if not rule.strip():
                st.error("Rule is required.")
                st.stop()

            try:
                variables = yaml.safe_load(variables_yaml) or {}
                if not isinstance(variables, dict):
                    raise ValueError("Variables must be a YAML map (key: value).")
            except Exception as e:
                st.error(f"Invalid Variables YAML: {e}")
                st.stop()

            suggestions = [ln.strip() for ln in (suggestions_txt or "").splitlines() if ln.strip()]

            custom_invs[name] = {
                "severity": severity,
                "variables": variables,
                "rule": rule.strip(),
                "why": why.strip() or "Invariant violated.",
                "suggestions": suggestions,
            }

            new_doc = dict(custom_doc or {})
            new_doc["invariants"] = custom_invs
            _save_custom(new_doc)
            st.success(f"Saved '{name}' to {CUSTOM_INVARIANTS_PATH}")
            st.info("Next: click 'Reload invariants in Orchestrator' above.")
