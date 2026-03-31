# Knowledge Graph Construction, Reasoning & RAG

**Authors:** Aharouni Refael, Aye William (DIA1)

A full pipeline from web crawling to RAG-powered question answering over a Wikidata knowledge graph.

---

## Project Structure

```
WebDataMining/
├── notebooks/
│   ├── Lab1_Web_Crawling_NER.ipynb          # Lab 1 - Web crawling & NER
│   ├── Lab4_KB_Construction_Expansion.ipynb  # Lab 4 - SPARQL KB expansion
│   ├── Lab5_Reasoning_KGE.ipynb             # Lab 5 - SWRL reasoning & KGE
│   └── Lab6_RAG.ipynb                       # Lab 6 - RAG over RDF/SPARQL
├── data/
│   ├── extracted_knowledge.csv   <- NER entities (Lab 1 output)
│   ├── triples.csv               <- relation triples (Lab 1 output)
│   ├── crawler_output.jsonl      <- raw crawl data
│   ├── entity_mapping.csv        <- entity alignment (Lab 4)
│   ├── predicate_candidates.csv  <- predicate alignment (Lab 4)
│   └── expanded_kb.csv           <- KB in CSV form (Lab 4)
├── kg_artifacts/
│   ├── initial_kb.ttl            <- initial RDF graph from Lab 1 NER
│   ├── expanded_kb.rdf           <- full expanded KB, Turtle format (~52k triples)
│   ├── alignment.ttl             <- predicate alignment (Wikidata -> schema.org/foaf)
│   └── family.owl                <- OWL ontology for SWRL reasoning (Lab 5)
├── kge_data/
│   ├── train.txt                 <- 80% training split
│   ├── valid.txt                 <- 10% validation split
│   └── test.txt                  <- 10% test split
├── rag/
│   └── lab_rag_sparql_gen.py     <- RAG CLI script (Lab 6)
├── README.md
├── requirements.txt
└── .gitignore
```

> The `_other/` folder (practice labs TD2, TD3 and lab PDFs) is gitignored.

---

## Installation

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd WebDataMining
pip install -r requirements.txt
```

### 2. Download SpaCy model

```bash
python -m spacy download en_core_web_trf
```

### 3. Install Ollama (for Lab 6 RAG)

Download from https://ollama.com, then:

```bash
ollama serve          # Start the Ollama server
ollama pull gemma:2b  # Download the model (~2 GB)
```

---

## How to Run Each Module

### Lab 1 - Web Crawling & NER

Open `notebooks/Lab1_Web_Crawling_NER.ipynb` and run all cells.

**Output:** `data/extracted_knowledge.csv`, `data/triples.csv`, `kg_artifacts/initial_kb.ttl`

### Lab 4 - KB Construction & SPARQL Expansion

Open `notebooks/Lab4_KB_Construction_Expansion.ipynb` and run all cells.

Requires internet (queries Wikidata SPARQL endpoint).

**Output:** `kg_artifacts/expanded_kb.rdf`, `kg_artifacts/expanded_kb.nt`, `data/expanded_kb.csv`

### Lab 5 - SWRL Reasoning & KGE

Open `notebooks/Lab5_Reasoning_KGE.ipynb` and run all cells (requires Lab 4 output).

**Warning:** KGE training runs on CPU and takes ~10-30 min per model.

**Output:** `kge_data/train.txt`, `kge_data/valid.txt`, `kge_data/test.txt`, metrics table

### Lab 6 - RAG Demo

#### Interactive CLI

```bash
cd rag
python lab_rag_sparql_gen.py --model gemma:2b
```

#### Run evaluation table (5 questions, baseline vs RAG)

```bash
cd rag
python lab_rag_sparql_gen.py --model gemma:2b --eval
```

#### Notebook

Open `notebooks/Lab6_RAG.ipynb` and run all cells.

---

## RAG Demo Screenshot

Below is a sample session from the interactive CLI (`lab_rag_sparql_gen.py`):

```
$ python lab_rag_sparql_gen.py --model gemma:2b

Loading RDF graph from ../kg_artifacts/expanded_kb.rdf ...
  Graph loaded: 52184 triples
Building schema summary...

RAG SPARQL Demo  (model: gemma:2b)
Type your question, 'eval' to run the evaluation table, or 'quit' to exit.

Your question: What are the occupations of James Clerk Maxwell?

[Baseline - no RAG]
  James Clerk Maxwell was a Scottish mathematician and physicist...

[RAG - SPARQL generation]
  Generated SPARQL:
    SELECT ?occupation WHERE {
      ?person rdfs:label "James Clerk Maxwell"@en .
      ?person wdt:P106 ?occupation .
    } LIMIT 10

  Results (2 rows):
    occupation
    wd:Q901        (scientist)
    wd:Q170790     (physicist)

Your question: quit
```

---

## Hardware Requirements

| Module | Minimum | Recommended |
|--------|---------|-------------|
| Lab 1 (crawling) | 4 GB RAM | 8 GB RAM |
| Lab 4 (SPARQL) | 4 GB RAM + internet | 8 GB RAM |
| Lab 5 (KGE) | 8 GB RAM, CPU | 16 GB RAM, GPU |
| Lab 6 (RAG) | 8 GB RAM (Ollama gemma:2b) | 16 GB RAM |

Tested on: Windows 11, Intel CPU, 16 GB RAM, no GPU.
KGE training time on CPU: ~10-30 min/model at 500 epochs.

---

## Knowledge Graph Files

| File | Description |
|------|-------------|
| `kg_artifacts/initial_kb.ttl` | Initial RDF graph from Lab 1 NER (~text triples) |
| `kg_artifacts/expanded_kb.rdf` | Full expanded KB, Turtle format (~52k triples) |
| `kg_artifacts/alignment.ttl` | Predicate alignment: Wikidata properties to schema.org/foaf |
| `kg_artifacts/family.owl` | OWL ontology for SWRL demo (Lab 5) |
| `kge_data/train.txt` | KGE training triples (80%) |
| `kge_data/valid.txt` | KGE validation triples (10%) |
| `kge_data/test.txt` | KGE test triples (10%) |
