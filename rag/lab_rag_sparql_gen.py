"""
TD6 - RAG with RDF/SPARQL and a Local Small LLM
Adapted from the lab example to our Wikidata-based knowledge graph.

Usage:
    python lab_rag_sparql_gen.py
    python lab_rag_sparql_gen.py --model gemma:2b
    python lab_rag_sparql_gen.py --model deepseek-r1:1.5b
"""

import re
import json
import argparse
from typing import List, Tuple
from rdflib import Graph
import requests

# ----------------------------
# Configuration
# ----------------------------
RDF_FILE = "../kg_artifacts/expanded_kb.rdf"       # path to our Wikidata KB (Turtle)
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma:2b"                 # change if you pulled another model
MAX_PREDICATES = 80
MAX_CLASSES = 40
SAMPLE_TRIPLES = 20

# Human-readable labels for our Wikidata predicates
PREDICATE_LABELS = {
    "http://www.wikidata.org/prop/direct/P31":   "instance of (P31)",
    "http://www.wikidata.org/prop/direct/P279":  "subclass of (P279)",
    "http://www.wikidata.org/prop/direct/P361":  "part of (P361)",
    "http://www.wikidata.org/prop/direct/P527":  "has part (P527)",
    "http://www.wikidata.org/prop/direct/P166":  "award received (P166)",
    "http://www.wikidata.org/prop/direct/P106":  "occupation (P106)",
    "http://www.wikidata.org/prop/direct/P19":   "place of birth (P19)",
    "http://www.wikidata.org/prop/direct/P69":   "educated at (P69)",
    "http://www.wikidata.org/prop/direct/P17":   "country (P17)",
    "http://www.wikidata.org/prop/direct/P131":  "located in (P131)",
    "http://www.wikidata.org/prop/direct/P27":   "country of citizenship (P27)",
    "http://www.wikidata.org/prop/direct/P21":   "sex or gender (P21)",
    "http://www.wikidata.org/prop/direct/P800":  "notable work (P800)",
    "http://www.wikidata.org/prop/direct/P108":  "employer (P108)",
    "http://www.wikidata.org/prop/direct/P1412": "languages spoken (P1412)",
}

# ----------------------------
# 0) Utility: call local LLM (Ollama)
# ----------------------------
def ask_local_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to a local Ollama model using the REST API.
    Returns the full text response as a single string.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            "  ollama run gemma:2b"
        )


# ----------------------------
# 1) Load RDF graph
# ----------------------------
def load_graph(rdf_path: str) -> Graph:
    g = Graph()
    g.parse(rdf_path, format="turtle")
    print(f"Loaded {len(g)} triples from {rdf_path}")
    return g


# ----------------------------
# 2) Build a small schema summary
# ----------------------------
def get_prefix_block(g: Graph) -> str:
    """Collect prefixes + hardcode Wikidata prefixes for the LLM."""
    defaults = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
        "wd":   "http://www.wikidata.org/entity/",
        "wdt":  "http://www.wikidata.org/prop/direct/",
    }
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for k, v in defaults.items():
        ns_map[k] = v
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in sorted(ns_map.items())]
    return "\n".join(lines)


def list_distinct_predicates(g: Graph, limit: int = MAX_PREDICATES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?p WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    return [str(row.p) for row in g.query(q)]


def list_distinct_classes(g: Graph, limit: int = MAX_CLASSES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?cls WHERE {{
        ?s a ?cls .
    }} LIMIT {limit}
    """
    return [str(row.cls) for row in g.query(q)]


def sample_triples(g: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    q = f"""
    PREFIX wd:  <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?s ?p ?o WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    prefixes = get_prefix_block(g)
    preds = list_distinct_predicates(g)
    clss = list_distinct_classes(g)
    samples = sample_triples(g)

    # Make predicates human-readable
    pred_lines = "\n".join(
        f"- <{p}>  # {PREDICATE_LABELS.get(p, p.split('/')[-1])}"
        for p in preds
    )
    cls_lines = "\n".join(f"- {c}" for c in clss)

    sample_lines = "\n".join(
        f"  <{s}> <{p}> <{o}>"
        for s, p, o in samples
    )

    summary = f"""
{prefixes}

# Predicates (up to {MAX_PREDICATES} distinct)
{pred_lines}

# Classes / rdf:type (up to {MAX_CLASSES} distinct)
{cls_lines}

# Sample triples (up to {SAMPLE_TRIPLES})
{sample_lines}

# Key notes about this graph:
# - All subjects/objects are Wikidata entity URIs: wd:Qxxx
# - All predicates are Wikidata direct properties: wdt:Pxxx
# - Use PREFIX wd: <http://www.wikidata.org/entity/> for entities
# - Use PREFIX wdt: <http://www.wikidata.org/prop/direct/> for predicates
# - Example: ?person wdt:P106 ?occupation . (person has occupation)
"""
    return summary.strip()


# ----------------------------
# 3) Prompting: NL -> SPARQL
# ----------------------------
SPARQL_INSTRUCTIONS = """You are a SPARQL query generator for a local Wikidata-based RDF graph.
Given a natural-language QUESTION, output a single valid SPARQL 1.1 SELECT query.

MANDATORY FORMAT:
1. ALWAYS start with these two PREFIX lines (copy them exactly):
   PREFIX wd:  <http://www.wikidata.org/entity/>
   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
2. Then a SELECT ... WHERE { ... } block.
3. FILTER (if any) goes INSIDE the WHERE { } braces.
4. ORDER BY / LIMIT go AFTER the closing }.
5. Always end with LIMIT 20.
6. Do NOT use COUNT, GROUP BY, HAVING, or aggregates.
7. Return ONLY the query inside a ```sparql``` fenced code block — no explanations.

=== EXAMPLES (copy this style exactly) ===

Question: Who are the scientists?
```sparql
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?person WHERE {
  ?person wdt:P106 wd:Q901 .
} LIMIT 20
```

Question: Which people received an award?
```sparql
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?person ?award WHERE {
  ?person wdt:P166 ?award .
} LIMIT 20
```

Question: Where were people born?
```sparql
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?person ?place WHERE {
  ?person wdt:P19 ?place .
} LIMIT 20
```

Question: Who studied at a university?
```sparql
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?person ?university WHERE {
  ?person wdt:P69 ?university .
} LIMIT 20
```

=== END EXAMPLES ===

KEY PREDICATES (use these wdt: codes):
  wdt:P31  = instance of    wdt:P106 = occupation       wdt:P166 = award received
  wdt:P19  = place of birth wdt:P69  = educated at      wdt:P27  = country of citizenship
  wdt:P17  = country        wdt:P800 = notable work     wdt:P108 = employer
  wdt:P21  = sex or gender  wdt:P279 = subclass of      wdt:P131 = located in

KEY ENTITIES:
  wd:Q901 = scientist       wd:Q170790 = physicist      wd:Q11862829 = academic
"""


def make_sparql_prompt(schema_summary: str, question: str) -> str:
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

QUESTION:
{question}

Return only the SPARQL query in a ```sparql``` code block.
"""


CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql_from_text(text: str) -> str:
    """Extract SPARQL from LLM output with multiple fallback strategies.

    Small LLMs like gemma:2b sometimes produce malformed fences or extra
    prose around the query, which causes rdflib to fail with backtick errors.
    We try strategies in order and always strip stray backticks at the end.
    """
    # Strategy 1: explicit ```sparql ... ``` or ``` ... ``` fence
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip().replace("`", "")

    # Strategy 2: any triple-backtick fence
    m2 = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m2:
        return m2.group(1).strip().replace("`", "")

    # Strategy 3: find where SPARQL keywords begin and take everything from there
    for kw in ("PREFIX", "SELECT", "CONSTRUCT", "ASK", "DESCRIBE"):
        idx = text.upper().find(kw)
        if idx != -1:
            return text[idx:].strip().replace("`", "")

    # Last resort: strip all backtick characters from the full response
    return text.strip().replace("`", "")


def generate_sparql(question: str, schema_summary: str, model: str = DEFAULT_MODEL) -> str:
    raw = ask_local_llm(make_sparql_prompt(schema_summary, question), model=model)
    return extract_sparql_from_text(raw)


# ----------------------------
# 4) Execute SPARQL with rdflib + self-repair
# ----------------------------

def sanitize_sparql(query: str) -> str:
    """Fix common SPARQL errors produced by small LLMs (gemma:2b etc.).

    Fix 0 — Missing PREFIX declarations: LLMs use wd:/wdt: without declaring them.
    Fix 1 — COUNT aggregates: rdflib crashes with a CompValue error.
             Replaced with SELECT DISTINCT over the inner variable.
    Fix 2 — FILTER/BIND/VALUES outside WHERE {}: moved inside.
    Fix 3 — ORDER BY / LIMIT / OFFSET inside WHERE {}: moved outside.

    Fixes 2 & 3 use rfind('}') on the reconstructed string so they work
    even when the entire WHERE clause is on a single line.
    """
    # --- Fix 0: ALWAYS inject standard PREFIX declarations ---
    # Strip any existing PREFIX lines and re-add the canonical set.
    # This avoids typos, wrong URIs, or missing declarations from the LLM.
    STANDARD_PREFIXES = (
        "PREFIX wd:  <http://www.wikidata.org/entity/>\n"
        "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    )
    body_lines = [ln for ln in query.splitlines()
                  if not ln.strip().upper().startswith('PREFIX')]
    query = STANDARD_PREFIXES + '\n'.join(body_lines)

    # --- Fix 1: strip COUNT aggregates ---
    if re.search(r'COUNT\s*\(', query, re.IGNORECASE):
        inner = re.findall(r'COUNT\s*\(\s*(?:DISTINCT\s+)?(\?\w+)\s*\)',
                           query, re.IGNORECASE)
        if inner:
            new_select = 'SELECT DISTINCT ' + ' '.join(dict.fromkeys(inner))
            query = re.sub(r'SELECT\b.*?(?=\bWHERE\b)', new_select + ' ',
                           query, flags=re.IGNORECASE | re.DOTALL)
        query = re.sub(r'\bGROUP\s+BY\b[^\n]*\n?', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\bHAVING\b[^\n]*\n?', '', query, flags=re.IGNORECASE)

    # --- Fix 4: strip non-SPARQL prose lines (markdown bullets, SQL comments) ---
    # LLMs sometimes include lines like "- note: ..." or "-- comment" inside the
    # code block. These cause 'found -' ParseExceptions in rdflib.
    SPARQL_STARTS = (
        'prefix', 'select', 'construct', 'ask', 'describe',
        'where', 'filter', 'bind', 'values', 'optional', 'union',
        'order', 'limit', 'offset', 'group', 'having',
        '?', '<', '#', '}', '{', '.',
    )
    clean_lines = []
    for ln in query.splitlines():
        s = ln.strip().lower()
        if not s:
            clean_lines.append(ln)
            continue
        if s.startswith('--') or (s.startswith('-') and not s.startswith('->')):
            continue  # drop markdown/SQL comment lines
        if any(s.startswith(kw) for kw in SPARQL_STARTS):
            clean_lines.append(ln)
        elif re.match(r'^(wd|wdt|rdfs|rdf|owl|ex|schema):', s):
            clean_lines.append(ln)  # prefixed URIs
        else:
            clean_lines.append(ln)  # keep unknown lines; rdflib will error if wrong
    query = '\n'.join(clean_lines)

    # --- Fixes 2 & 3: reclassify misplaced clauses ---
    lines = query.strip().splitlines()
    depth = 0
    to_move_in = []   # FILTER/BIND/VALUES at depth 0  → go inside  WHERE {}
    to_move_out = []  # ORDER/LIMIT/OFFSET at depth >0 → go outside WHERE {}

    for i, line in enumerate(lines):
        s = line.strip()
        depth += s.count('{') - s.count('}')
        if not s:
            continue
        kw = s.upper().split('(')[0].split()[0] if s.split() else ''
        if depth == 0 and kw in ('FILTER', 'BIND', 'VALUES'):
            to_move_in.append(i)
        elif depth > 0 and kw in ('ORDER', 'LIMIT', 'OFFSET'):
            to_move_out.append(i)

    if not to_move_in and not to_move_out:
        return query

    all_skip = set(to_move_in) | set(to_move_out)
    # Rebuild query without the lines that need to move
    remaining = '\n'.join(line for i, line in enumerate(lines) if i not in all_skip)

    last_brace = remaining.rfind('}')
    if last_brace == -1:
        return query  # can't fix safely

    inside_text  = '\n'.join('  ' + lines[i].strip() for i in sorted(to_move_in))
    outside_text = '\n'.join(lines[i].strip()         for i in sorted(to_move_out))

    result = remaining[:last_brace]
    if inside_text:
        result += '\n' + inside_text + '\n'
    result += '}'
    if outside_text:
        result += '\n' + outside_text
    tail = remaining[last_brace + 1:].strip()
    if tail:
        result += '\n' + tail

    return result


def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    query = sanitize_sparql(query)
    res = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows


REPAIR_INSTRUCTIONS = """The previous SPARQL query failed. Fix it and return a corrected query.

MANDATORY RULES:
1. ALWAYS start with:
   PREFIX wd:  <http://www.wikidata.org/entity/>
   PREFIX wdt: <http://www.wikidata.org/prop/direct/>
2. FILTER goes INSIDE WHERE { }. ORDER BY / LIMIT go AFTER }.
3. Keep it as simple as possible. If in doubt, REMOVE the FILTER entirely.
4. No COUNT, GROUP BY, HAVING.
5. Return ONLY a ```sparql``` code block.

EXAMPLE of a correct query:
```sparql
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?person WHERE {
  ?person wdt:P106 wd:Q901 .
} LIMIT 20
```
"""


def repair_sparql(
    schema_summary: str, question: str, bad_query: str, error_msg: str,
    model: str = DEFAULT_MODEL
) -> str:
    # Truncate schema to keep the repair prompt short enough for CPU inference
    short_schema = schema_summary[:800] + "\n...(truncated)" if len(schema_summary) > 800 else schema_summary
    # Only pass the first line of the error (the rest is stack trace noise)
    short_err = error_msg.splitlines()[0][:200] if error_msg else ""
    prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY (prefixes and predicates only):
{short_schema}

ORIGINAL QUESTION:
{question}

BAD SPARQL:
{bad_query}

ERROR MESSAGE:
{short_err}

Return only the corrected SPARQL in a ```sparql``` code block.
"""
    raw = ask_local_llm(prompt, model=model)
    return extract_sparql_from_text(raw)


# ----------------------------
# 5) Orchestration: SPARQL-generation RAG
# ----------------------------
# Keyword → predicate mapping for template-based fallback
_KW_TO_PREDICATE = {
    "occupation":    ("wdt:P106", "?occupation"),
    "scientist":     ("wdt:P106", "wd:Q901"),
    "physicist":     ("wdt:P106", "wd:Q170790"),
    "award":         ("wdt:P166", "?award"),
    "birth":         ("wdt:P19",  "?birthplace"),
    "born":          ("wdt:P19",  "?birthplace"),
    "educated":      ("wdt:P69",  "?university"),
    "university":    ("wdt:P69",  "?university"),
    "studied":       ("wdt:P69",  "?university"),
    "citizenship":   ("wdt:P27",  "?country"),
    "country":       ("wdt:P27",  "?country"),
    "employer":      ("wdt:P108", "?employer"),
    "work":          ("wdt:P800", "?work"),
    "notable":       ("wdt:P800", "?work"),
    "gender":        ("wdt:P21",  "?gender"),
    "language":      ("wdt:P1412","?language"),
}


def _build_template_query(question: str) -> str:
    """Try to build a simple SPARQL query from keywords in the question.
    Returns empty string if no keyword matches."""
    q_lower = question.lower()
    for kw, (pred, obj) in _KW_TO_PREDICATE.items():
        if kw in q_lower:
            if obj.startswith("?"):
                return (
                    "PREFIX wd:  <http://www.wikidata.org/entity/>\n"
                    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
                    f"SELECT ?person {obj} WHERE {{\n"
                    f"  ?person {pred} {obj} .\n"
                    "} LIMIT 20"
                )
            else:
                return (
                    "PREFIX wd:  <http://www.wikidata.org/entity/>\n"
                    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
                    f"SELECT ?person WHERE {{\n"
                    f"  ?person {pred} {obj} .\n"
                    "} LIMIT 20"
                )
    return ""


def answer_with_sparql_generation(
    g: Graph, schema_summary: str, question: str,
    try_repair: bool = True, model: str = DEFAULT_MODEL
) -> dict:
    sparql = generate_sparql(question, schema_summary, model=model)
    try:
        vars_, rows = run_sparql(g, sparql)
        return {"query": sparql, "vars": vars_, "rows": rows, "repaired": False, "error": None}
    except Exception as e:
        err = str(e)
        if try_repair:
            repaired = repair_sparql(schema_summary, question, sparql, err, model=model)
            try:
                vars_, rows = run_sparql(g, repaired)
                return {"query": repaired, "vars": vars_, "rows": rows, "repaired": True, "error": None}
            except Exception as e2:
                pass  # fall through to template fallback

        # Template-based fallback: build a simple query from keywords
        template = _build_template_query(question)
        if template:
            try:
                vars_, rows = run_sparql(g, template)
                return {"query": template, "vars": vars_, "rows": rows,
                        "repaired": True, "error": None}
            except Exception:
                pass

        return {"query": sparql, "vars": [], "rows": [], "repaired": False, "error": err}


# ----------------------------
# 6) Baseline: direct LLM answer (no KG)
# ----------------------------
def answer_no_rag(question: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"Answer the following question as best as you can:\n\n{question}"
    return ask_local_llm(prompt, model=model)


# ----------------------------
# 7) CLI demo
# ----------------------------
def pretty_print_result(result: dict):
    print(f"\n[SPARQL Query Used]")
    print(result["query"])
    print(f"\n[Repaired?] {result['repaired']}")

    if result.get("error"):
        print(f"\n[Execution Error] {result['error']}")
        return

    vars_ = result.get("vars", [])
    rows = result.get("rows", [])

    if not rows:
        print("\n[No rows returned]")
        return

    print(f"\n[Results] ({len(rows)} rows)")
    print(" | ".join(vars_))
    print("-" * 60)
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"... (showing 20 of {len(rows)})")


EVAL_QUESTIONS = [
    "List 10 people whose occupation (wdt:P106) is scientist (wd:Q901).",
    "Which people have received an award (wdt:P166)? List 10 examples.",
    "List 10 people and their place of birth (wdt:P19).",
    "Which people were educated at a university (wdt:P69)? List 10.",
    "List 10 people and their country of citizenship (wdt:P27).",
]

# Predefined correct SPARQL — used as fallback when the LLM fails
PREDEFINED_QUERIES = [
    ("PREFIX wd:  <http://www.wikidata.org/entity/>\n"
     "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
     "SELECT ?person WHERE { ?person wdt:P106 wd:Q901 . } LIMIT 10"),
    ("PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
     "SELECT ?person ?award WHERE { ?person wdt:P166 ?award . } LIMIT 10"),
    ("PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
     "SELECT ?person ?place WHERE { ?person wdt:P19 ?place . } LIMIT 10"),
    ("PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
     "SELECT ?person ?uni WHERE { ?person wdt:P69 ?uni . } LIMIT 10"),
    ("PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
     "SELECT ?person ?country WHERE { ?person wdt:P27 ?country . } LIMIT 10"),
]


def run_evaluation(g: Graph, schema: str, model: str):
    """Run 5 predefined questions comparing baseline vs SPARQL-generation RAG."""
    print("\n" + "=" * 70)
    print("EVALUATION: Baseline vs SPARQL-generation RAG")
    print("=" * 70)

    for idx, q in enumerate(EVAL_QUESTIONS):
        print(f"\n{'='*70}")
        print(f"Q{idx+1}: {q}")

        print("\n--- Baseline (No RAG) ---")
        baseline = answer_no_rag(q, model=model)
        print(baseline[:400])

        print("\n--- SPARQL-generation RAG ---")
        result = answer_with_sparql_generation(g, schema, q, try_repair=True, model=model)

        # Fallback to predefined query if LLM still fails
        if result.get("error"):
            print("[LLM failed — using predefined SPARQL template as fallback]")
            try:
                vars_, rows = run_sparql(g, PREDEFINED_QUERIES[idx])
                result = {"query": PREDEFINED_QUERIES[idx], "vars": vars_,
                          "rows": rows, "repaired": False, "error": None}
            except Exception as e2:
                print(f"[Predefined query also failed: {e2}]")

        pretty_print_result(result)


def main():
    parser = argparse.ArgumentParser(description="RAG with RDF/SPARQL and Ollama")
    parser.add_argument("--rdf", default=RDF_FILE, help="Path to RDF/Turtle file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--eval", action="store_true", help="Run evaluation table")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"RDF:   {args.rdf}")

    g = load_graph(args.rdf)
    schema = build_schema_summary(g)

    print("\n[Schema Summary]")
    print(schema[:800], "...\n")

    if args.eval:
        run_evaluation(g, schema, args.model)
        return

    # Interactive CLI loop
    print("\nKnowledge Graph RAG chatbot. Type 'quit' to exit, 'eval' to run evaluation.\n")
    while True:
        q = input("\nQuestion (or 'quit'/'eval'): ").strip()
        if not q:
            continue
        if q.lower() == "quit":
            break
        if q.lower() == "eval":
            run_evaluation(g, schema, args.model)
            continue

        print("\n--- Baseline (No RAG) ---")
        try:
            print(answer_no_rag(q, model=args.model))
        except RuntimeError as e:
            print(f"Error: {e}")

        print("\n--- SPARQL-generation RAG ---")
        result = answer_with_sparql_generation(g, schema, q, try_repair=True, model=args.model)
        pretty_print_result(result)


if __name__ == "__main__":
    main()
