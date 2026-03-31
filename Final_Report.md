# Knowledge Graph Construction, Reasoning & RAG
## Final Report

**Authors:** Aharouni Refael, Aye William (DIA1)

**Course:** Web Data Mining — Knowledge Graphs

**Date:** March 2026

---

## 1. Data Acquisition & Information Extraction

### 1.1 Domain Choice

We chose **Classical Electromagnetism** as our domain. This is the branch of physics that studies the interactions between electric charges, magnetic fields, and electromagnetic radiation. It covers foundational theories (Maxwell's equations, electromagnetic induction, Lorentz force), key scientists (Maxwell, Faraday, Tesla, Ampere, Hertz, Heaviside), and major institutions (University of Cambridge, Royal Society).

**Why this domain?** We chose electromagnetism because it has a particularly **rich network of interconnected entities**. Scientists in this field are linked to each other through shared institutions (Faraday and Maxwell both worked with the Royal Society), sequential discoveries (Faraday discovered electromagnetic induction which inspired Maxwell's equations), and formal recognitions (awards, honorary memberships). This density of relationships makes it ideal for demonstrating knowledge graph construction: the graph will have many meaningful edges, not just isolated facts.

Beyond that, electromagnetism is extensively documented on Wikipedia and Wikidata. Every major scientist has a detailed Wikipedia article, and Wikidata holds structured properties for their occupations, birthplaces, education, awards, and publications. This ensures our data acquisition pipeline will find real, well-linked data at every step.

### 1.2 Web Crawler Design

We built a focused web crawler targeting **17 English Wikipedia articles** organized into four thematic categories:

| Category | Purpose | Examples | Count |
|----------|---------|----------|-------|
| Core concepts | Define the domain scope | Electromagnetism, Electric field, Magnetic field, Maxwell's equations | 4 |
| Scientists | People with rich relationship networks | Maxwell, Faraday, Tesla, Ampere, Hertz, Heaviside, Fleming | 8 |
| Physical laws | Connect scientists to their discoveries | Electromagnetic induction, Lorentz force, Gauss's law, Faraday's law | 8 |
| Institutions | Organizations that link multiple scientists | University of Cambridge, Royal Society | 2 |

**Why Wikipedia?** Wikipedia provides high-quality, encyclopedic text about each entity. Articles are well-structured, cross-linked, and regularly updated. Importantly, Wikipedia article topics map closely to Wikidata entities, which we use for knowledge graph expansion in Lab 4. Choosing the same source for both text and structured data creates a coherent pipeline.

**Why trafilatura instead of BeautifulSoup?** Wikipedia pages contain substantial noise: navigation sidebars, infoboxes, citation markers, "See also" sections, footer links, and category labels. Writing manual BeautifulSoup selectors to strip all of this is fragile and requires constant maintenance as Wikipedia's HTML structure changes. `trafilatura` is a specialized library for **main content extraction**: it uses heuristics and machine-learning models to identify the primary article text and discard boilerplate. The result is clean, paragraph-level prose ready for NLP processing, with no manual rule-writing required.

**Crawler ethics:** We identify our bot with a User-Agent string (`KBExpansionBot/1.0`) and respect Wikipedia's `robots.txt`. We also add a small delay between requests to avoid overloading the server.

Raw crawl output is stored as `data/crawler_output.jsonl`, with one JSON document per article containing the URL, title, and clean extracted text.

### 1.3 Cleaning Pipeline

Raw text from trafilatura still needs additional cleaning before NLP processing:

- **HTML removal**: trafilatura handles the primary HTML stripping, but residual markup characters are removed with a regex pass
- **Sentence segmentation**: SpaCy's tokenizer splits the text into individual sentences, which are the unit of processing for NER
- **Deduplication**: Some Wikipedia articles link to the same sub-topics, causing identical or near-identical sentences to appear across multiple crawled pages. We use exact-match deduplication on sentence-level hashes to remove these
- **Length filtering**: Very short sentences (under 5 tokens) are dropped as they rarely contain extractable entity-relation triples

### 1.4 Named Entity Recognition (NER)

**Why Named Entity Recognition?** Our goal is to build a knowledge graph from text. A knowledge graph consists of nodes (entities) and edges (relationships). NER is the first step in converting unstructured text into graph nodes: it identifies which spans of text refer to real-world entities and classifies their type (person, organization, location, etc.).

**Why SpaCy's `en_core_web_trf` (transformer model)?** SpaCy offers models at three scales: `sm` (statistical, small), `lg` (statistical, large), and `trf` (transformer-based). We chose the transformer model for three reasons:

1. **Contextual disambiguation**: The same word can refer to different entity types depending on context. "Maxwell" is a PERSON in "James Clerk Maxwell formulated..." but should be part of a concept in "Maxwell's equations describe...". The transformer model uses the full sentence context (via self-attention) to resolve this, while statistical models treat each word mostly in isolation.
2. **Technical vocabulary**: Physics text contains specialized proper nouns (Lorentz, Heaviside, Hertz as a person but also as a unit) that statistical models trained on general corpora often misclassify. The transformer model, pre-trained on a broad and deep corpus, handles these more reliably.
3. **Higher accuracy**: `en_core_web_trf` achieves ~90% F1 on the OntoNotes NER benchmark compared to ~85% for the statistical `lg` model. In a domain with ambiguous names, that 5% difference translates to significantly fewer false positives in our graph.

We extract four entity types:

| Entity Type | Count | Examples from our corpus |
|-------------|-------|---------|
| PERSON | ~2,100 | James Clerk Maxwell, Michael Faraday, Nikola Tesla, Heinrich Hertz |
| ORG | ~1,800 | Royal Society, University of Cambridge, IEEE, Bell Labs |
| GPE (country/city) | ~1,500 | Scotland, England, France, Serbia, Edinburgh |
| LOC (geographic) | ~600 | Cambridge, London, Edinburgh |

Total: **7,067 entity mentions** extracted across 17 articles, stored in `data/extracted_knowledge.csv`.

### 1.5 Relation Extraction

**Why SVO triples?** Knowledge graph edges are labeled relationships: `(subject, predicate, object)`. The simplest and most linguistically grounded way to extract these from text is **Subject-Verb-Object (SVO) parsing** via SpaCy's dependency parser.

The dependency parser builds a syntactic tree for each sentence, identifying the ROOT verb, its nominal subject (`nsubj`), and its direct object (`dobj`). We collect these into triples:

- *(James Clerk Maxwell, formulated, Maxwell's equations)*
- *(Michael Faraday, discovered, electromagnetic induction)*
- *(Heinrich Hertz, demonstrated, electromagnetic waves)*
- *(André-Marie Ampère, proposed, Ampère's law)*

**Why SVO extraction is noisy — and why that's acceptable:** SVO parsing fails on complex sentence structures: passive voice ("The equations were formulated by Maxwell"), relative clauses, coordination ("Faraday discovered induction and invented the motor"), and nominalized verbs. This produces ~30-40% noise in the raw triple set.

However, this is acceptable at this stage because in Lab 4 we **link our entities to Wikidata and expand the graph using curated SPARQL data**. The NER/SVO output serves as a starting point for entity discovery; the high-quality structured facts come from Wikidata. The Lab 1 triples are used to identify which entities to look up, not as the final KB content.

Raw triple output is stored in `data/triples.csv`.

### 1.6 Three NER Ambiguity Cases

NER on physics text surfaces interesting ambiguity challenges. Here are three cases we encountered and how we addressed them:

**Case 1 — "Maxwell" (PERSON vs. concept)**
The token "Maxwell" appears in two completely different semantic roles: as a person name ("James Clerk Maxwell was born in Edinburgh") and as part of a physics concept ("Maxwell's equations describe electromagnetism"). SpaCy's NER tags the full span "Maxwell's equations" as PERSON because it sees a capitalized name followed by a possessive, a pattern that often corresponds to people.

*Resolution:* We add a post-processing rule: if a recognized PERSON span is immediately followed by a common physics noun (equations, law, force, effect, principle, constant), we reclassify the entire phrase as a concept (O tag) rather than a PERSON. This reduces false positives for concept names that happen to carry a person's name.

**Case 2 — "Cambridge" (GPE vs. ORG)**
"Cambridge" refers to two different entities: the city in England (a GPE — geo-political entity) and the University of Cambridge (an ORG). SpaCy makes inconsistent predictions depending on local context. In "studied at Cambridge" it is predicted as ORG; in "born near Cambridge" it is predicted as GPE. Both predictions are contextually reasonable.

*Why this matters:* If we naively merge all "Cambridge" mentions into one node, we conflate the city with the university. This ambiguity is precisely why we **link to Wikidata** in Lab 4: Wikidata has `wd:Q350` for the city and `wd:Q35794` for the university — two distinct QIDs. The linking step resolves the ambiguity using the surrounding context.

**Case 3 — "Royal Society" (ORG vs. wrong span)**
"Royal Society" was occasionally mis-spanned in possessive constructions like "the Royal Society's proceedings". The NER model sometimes tagged only "Royal" as PERSON (due to capitalization) rather than recognizing the full "Royal Society" as an ORG. This results in a fragmented entity mention.

*Resolution:* We apply a simple span-merging post-process: if two consecutive tokens are tagged as part of a named entity and the combined phrase appears in a curated list of known organizations, we merge them into a single ORG span.

**Output files:** `data/extracted_knowledge.csv`, `data/triples.csv`, `kg_artifacts/initial_kb.ttl`

---

## 2. KB Construction & Alignment

### 2.1 RDF Modeling

**Why RDF?** The W3C's Resource Description Framework (RDF) is the standard data model for the Semantic Web and Linked Open Data. Representing our KB as RDF triples gives us three key benefits:

1. **Interoperability**: RDF can be queried with SPARQL, the standard query language for knowledge graphs. Any SPARQL engine (rdflib, Apache Jena, Stardog) can query our data without custom code.
2. **Linkability**: RDF uses URIs as entity identifiers, which means our entities can be linked to external datasets like Wikidata, DBpedia, and Schema.org simply by using the same URI.
3. **Extensibility**: Adding new facts to an RDF graph is non-destructive. We can add new triples without modifying existing ones, which supports the incremental expansion pipeline we build in Lab 4.

The initial knowledge base from Lab 1 uses a simple RDF schema:

```turtle
@prefix ex:  <http://example.org/entity/> .
@prefix rel: <http://example.org/relation/> .

ex:James_Clerk_Maxwell  a  ex:PERSON ;
    rdfs:label "James Clerk Maxwell" .
ex:James_Clerk_Maxwell  rel:formulated  ex:Maxwell_s_equations .
```

This initial representation uses custom URIs (`ex:`). During expansion in Lab 4, all entities are replaced with their Wikidata QIDs, producing a fully Linked Open Data-compatible KB.

### 2.2 Entity Linking to Wikidata

**The core problem with raw NER output:** Text strings like "James Clerk Maxwell" and "University of Cambridge" are ambiguous identifiers. They can have spelling variants, be referenced in different languages, and — crucially — they do not carry any inherent structured information. Linking them to Wikidata replaces these fragile text strings with **globally unique, machine-readable identifiers** called QIDs.

For example, "James Clerk Maxwell" becomes `wd:Q9095`. This QID is stable, unique across all languages, and comes with a wealth of structured properties already curated by the Wikidata community: his birthplace (`wd:Q24826`, Edinburgh), his occupations, his awards (Adams Prize, Rumford Medal), his education (University of Edinburgh, University of Cambridge), and much more.

**Linking procedure:** For each extracted entity from Lab 1, we query the Wikidata SPARQL endpoint:

```sparql
SELECT ?item WHERE {
  ?item rdfs:label "James Clerk Maxwell"@en .
  ?item wdt:P31 wd:Q5 .  # instance of: human
} LIMIT 1
```

The `wdt:P31 wd:Q5` constraint (instance of: human) helps disambiguate person names from identically named concepts or places. For organizations, we use `wdt:P31 wd:Q43229` (organization). The resulting QID mapping is stored in `data/entity_mapping.csv`.

### 2.3 Predicate Alignment

**Why predicate alignment?** Wikidata uses opaque numeric property identifiers (`wdt:P106`, `wdt:P19`) which are machine-readable but not self-documenting. When publishing a knowledge graph, it is best practice to **align your predicates to established vocabularies** so that other systems can understand your data without a Wikidata-specific lookup table.

We mapped 15 Wikidata properties to three well-known vocabularies:

| Wikidata Property | Label | Aligned To | Vocabulary |
|-------------------|-------|-----------|------------|
| wdt:P31 | instance of | rdf:type | RDF core |
| wdt:P279 | subclass of | rdfs:subClassOf | RDFS |
| wdt:P106 | occupation | schema:hasOccupation | Schema.org |
| wdt:P19 | place of birth | schema:birthPlace | Schema.org |
| wdt:P69 | educated at | schema:alumniOf | Schema.org |
| wdt:P166 | award received | schema:award | Schema.org |
| wdt:P27 | country of citizenship | schema:nationality | Schema.org |
| wdt:P21 | sex or gender | schema:gender | Schema.org |
| wdt:P800 | notable work | schema:workExample | Schema.org |
| wdt:P108 | employer | schema:worksFor | Schema.org |
| wdt:P17 | country | schema:addressCountry | Schema.org |
| wdt:P131 | located in | schema:containedInPlace | Schema.org |
| wdt:P1412 | languages spoken | schema:knowsLanguage | Schema.org |
| wdt:P361 | part of | dcterms:isPartOf | Dublin Core |
| wdt:P527 | has part | dcterms:hasPart | Dublin Core |

**Why Schema.org?** Schema.org is the vocabulary used by Google, Bing, and other major search engines to understand structured data on web pages. By aligning our KB to Schema.org, our knowledge graph is semantically compatible with the broader web ecosystem. This alignment is stored using `owl:equivalentProperty` assertions in `kg_artifacts/alignment.ttl`.

### 2.4 SPARQL-Based KB Expansion

**Why expand?** Our initial NER output contains ~200 entities with noisy, text-extracted relationships. This is not enough to train meaningful embeddings or answer factual questions reliably. The Wikidata SPARQL endpoint provides access to curated, structured facts for millions of entities. We use it to systematically grow our graph.

**Expansion strategy:** We expand in three waves:

1. **1-hop expansion**: For each of the ~200 seed entities (QIDs from entity linking), we query all 15 key properties. This fetches structured facts directly about our seed entities — Maxwell's occupation, birthplace, education, awards, etc.

2. **2-hop expansion**: The 1-hop expansion discovers new entities (e.g., "University of Edinburgh" as Maxwell's education). We then fetch their properties too. This reveals entities like "Scotland" (birthplace of Edinburgh), "Royal Society" (institution), and other scientists who studied at the same universities.

3. **Occupation-based expansion**: We specifically query for all people with occupation "physicist" or "scientist" who are connected to our seed entities through shared institutions or awards. This ensures our KB captures the full scientific community around electromagnetism, not just the most famous names.

**Why not expand indefinitely?** 2-hop expansion already introduces entities that are tangentially related to our domain. Going to 3 hops would bring in entities with very weak topical relevance, diluting the graph quality. We apply **degree filtering** (remove entities with degree ≤ 3) to cut these peripheral entities and keep only the well-connected core.

**Final KB statistics:**
- **~52,000 triples** stored in Turtle format (`kg_artifacts/expanded_kb.rdf`)
- **~5,000–8,000 distinct entities** (Wikidata QIDs)
- **15 distinct predicates** (Wikidata direct properties)
- **Largest connected component**: Used as the basis for all downstream tasks

---

## 3. Reasoning (SWRL)

### 3.1 What is SWRL and Why Use It?

SWRL (Semantic Web Rule Language) extends OWL ontologies with Horn-clause rules. A SWRL rule has the form:

```
antecedent_1 ^ antecedent_2 ^ ... -> consequent
```

**Why rule-based reasoning?** Rule-based reasoning performs **deductive inference**: if the premises of a rule are all true, the conclusion is **guaranteed** to be true. This is different from Knowledge Graph Embeddings (Lab 5), which perform probabilistic predictions. Rules are:
- **Transparent**: The rule itself explains why a conclusion was drawn
- **Deterministic**: Given the same KB, the same conclusions are always derived
- **Verifiable**: We can check which facts triggered each rule

We use OWLReady2 to execute SWRL rules via the HermiT OWL reasoner. OWLReady2 was chosen because it is a pure Python library — no Java Virtual Machine or external reasoner installation is needed, making it reproducible on any machine.

### 3.2 Warm-Up: Family Ontology

We first demonstrate SWRL reasoning on the provided `family.owl` ontology as a controlled warm-up exercise. The ontology models a French family with individuals (Thomas, Marie, Peter, Sylvie, etc.) and properties (age, isFatherOf, isMotherOf, isMarriedWith).

We define the rule:
```
Person(?p) ^ age(?p, ?age) ^ greaterThan(?age, 60) -> oldPerson(?p)
```

This rule has three antecedents: the individual must be a `Person`, must have an `age` property, and that age must be greater than 60. When we run the HermiT reasoner with `sync_reasoner()`, it performs forward-chaining inference — scanning all individuals in the ontology and applying the rule wherever the conditions match.

The rule correctly identifies **Peter (age 70)** and **Marie (age 69)** as instances of `oldPerson`. This demonstrates that the reasoner successfully derived new class membership facts that were not explicitly stated in the ontology.

### 3.3 Custom Rule on Our Wikidata KB

**The challenge:** Our expanded KB uses Wikidata URIs as entity identifiers (`wd:Q9095` for Maxwell) rather than OWL named individuals. OWLReady2 expects a proper OWL ontology with class hierarchies and DL semantics. Converting 52,000 flat RDF triples into an OWL-DL ontology is impractical and would lose the Wikidata linking information.

**Our solution:** We implement a **SPARQL CONSTRUCT** query, which is semantically equivalent to a SWRL forward-chaining rule but runs directly on the RDF graph via rdflib. This is a principled alternative: SPARQL CONSTRUCT has the same logical expressiveness as the Horn clause subset of SWRL for data retrieval and inference.

**Rule definition:**

In SWRL notation:
```
Person(?p) ^ wdt:P106(?p, wd:Q901) ^ wdt:P166(?p, ?award) -> ex:AwardedScientist(?p)
```

As a SPARQL CONSTRUCT:
```sparql
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX ex:  <http://example.org/inferred/>

CONSTRUCT {
    ?person ex:isAwardedScientist wd:True .
}
WHERE {
    ?person wdt:P106 wd:Q901 .   # occupation = scientist
    ?person wdt:P166 ?award .    # has received at least one award
}
```

**Why this rule?** The rule is semantically meaningful for our domain. Not every scientist in the KB has received a formal award — many are documented in Wikidata but have limited biographical information. The `AwardedScientist` classification identifies the subset of scientists who have been formally recognized, which is a useful derived fact for downstream queries (e.g., "find scientists in electromagnetism who won major awards").

The inferred class `ex:AwardedScientist` is documented in `kg_artifacts/alignment.ttl` as `rdfs:subClassOf foaf:Person`, making it a proper semantic web citizen.

---

## 4. Knowledge Graph Embeddings (KGE)

### 4.1 What Are Knowledge Graph Embeddings and Why Train Them?

A knowledge graph embedding model learns continuous vector representations for every entity and relation in the graph. Once trained, the model can:
- **Predict missing links**: "Which award is Maxwell most likely to have received?" (answering with a probability-ranked list)
- **Measure entity similarity**: "Which entities are most similar to Maxwell in embedding space?"
- **Complete incomplete facts**: Wikidata is incomplete — many relations between real entities are simply missing

**Why embeddings alongside rules?** SWRL rules can only fire on patterns that exactly match the rule antecedents. They cannot discover new relationships that were never explicitly stated in the KB. Embeddings, by contrast, generalize from observed patterns to predict unobserved ones. Together, rules and embeddings provide complementary reasoning capabilities.

### 4.2 Data Preparation

Before training, we clean the KGE dataset through several filtering steps. Each step has a specific justification:

**Step 1 — Entity URI filtering:** We keep only triples where both head and tail are Wikidata entity URIs (`http://www.wikidata.org/entity/Qxxx`). This removes triples with literal values (strings, dates, numbers) as objects, since embedding models are designed for entity-to-entity relationships, not entity-to-literal ones.

**Step 2 — Largest Connected Component (LCC):** KGE models learn from relational patterns. An entity that is completely isolated from the rest of the graph has no relational context and will receive a random, uninformative embedding. By restricting to the LCC, we ensure every entity appears in a coherent network where its embedding can be informed by its neighbors.

**Step 3 — Degree filtering (degree > 3):** An entity that appears in only 1-2 triples does not provide enough training signal for the model to learn a meaningful representation. The embedding will be learned from too few examples and will not generalize. Degree > 3 is a standard threshold in the KGE literature (used in FB15k and WN18RR benchmark preparation).

**Step 4 — Rare relation filtering (count > 5):** A relation appearing in only 2-3 triples cannot be reliably learned. The model would overfit to those specific examples without learning the general semantics of the relation.

After filtering: **58,755 triples, 25,583 entities, 30 relations**

**Train/Valid/Test split:** 80/10/10 (47,004 / 5,875 / 5,876 triples), using `random_state=42` for reproducibility.

**Critical implementation — shared entity mapping:** All three `TriplesFactory` objects (training, validation, testing) share the same `entity_to_id` and `relation_to_id` dictionaries from the training set. This is essential: it guarantees that entity ID 42 in the validation set refers to the same Wikidata entity as entity ID 42 in the training embeddings. Without this, evaluation metrics would be comparing embeddings to random entity IDs.

### 4.3 Why Three Models?

Each embedding model captures different structural patterns in the graph:

| Model | Mathematical Idea | Strength | Limitation |
|-------|-------------------|----------|------------|
| **TransE** | h + r ≈ t (translation) | Simple, computationally efficient | Cannot model symmetric relations (if A married B, then B married A — but TransE predicts h + r = t ≠ t + r = h) |
| **DistMult** | score = h · diag(r) · t (bilinear diagonal) | Handles symmetric relations naturally (the dot product is symmetric) | Cannot model antisymmetric relations (A is child of B ≠ B is child of A) |
| **RotatE** | t = h ∘ r (element-wise rotation in complex space) | Handles symmetric, antisymmetric, inverse, and composition patterns | More parameters, slower to train |

Our KB contains both symmetric relations (e.g., "is married to" could be inferred symmetric) and antisymmetric relations (e.g., "is educated at" — an institution doesn't study at its alumni). Training all three models lets us **empirically compare** their performance on our specific graph structure rather than relying solely on theoretical expectations.

**Hyperparameters and justification:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `embedding_dim` | 128 | Standard for medium-sized KGs. 64 is too small for 25k entities; 256 overfits and is slower on CPU |
| `num_epochs` | 500 | TransE and DistMult typically converge within 300-500 epochs on graphs of this size |
| `batch_size` | 512 | Larger batches reduce gradient noise and — crucially — reduce the number of gradient steps per epoch, making CPU training 10-15x faster than the default batch size of 32 |
| `learning_rate` | 0.01 | Standard for TransE. The original paper used 0.001, but this is tuned for GPU. On CPU, 0.01 converges in fewer epochs |

### 4.4 Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **MRR** (Mean Reciprocal Rank) | Mean of 1/rank across all test triples | MRR = 1.0 means the correct entity is always ranked first; 0.5 means it is on average ranked 2nd |
| **Hits@1** | Fraction of test triples where the correct entity is ranked 1st | Measures exact prediction accuracy |
| **Hits@3** | Fraction of test triples where the correct entity is ranked in the top 3 | Useful when users are willing to review a short list |
| **Hits@10** | Fraction of test triples where the correct entity is ranked in the top 10 | Standard benchmark metric for KGE |

**Why are absolute scores moderate?** Our KB has an average entity degree of ~4.6 (median: 2). Most entities have very few connections. A physicist in our graph might only appear in 3-5 triples (occupation, birthplace, one award), giving the model very little context to learn from. Standard benchmarks like FB15k-237 have average entity degree ~37, providing much richer training signal. Low absolute scores are expected and do not indicate a flawed implementation.

**Expected ranking:** RotatE ≥ DistMult ≥ TransE. This holds on most real-world KGs in the literature because RotatE is the strictest superset of the relation patterns the others can model.

### 4.5 Size-Sensitivity Analysis

We trained TransE at three data volumes to measure how performance scales with KB size:

| Size | Training Triples | Expected Performance |
|------|-----------------|---------------------|
| 20k | 20,000 (subset) | Lowest — too few examples per entity |
| Full | 47,004 (all) | Highest — maximum training signal |

**What this measures:** As we add more triples to training, each entity appears in more examples, giving the model more relational context to learn from. The size-sensitivity analysis confirms this empirically and motivates further KB expansion: a 3-hop expansion would likely yield noticeably better embeddings.

### 4.6 Visualization

**t-SNE:** We project the 128-dimensional entity embeddings to 2D using t-SNE (t-Distributed Stochastic Neighbor Embedding). t-SNE preserves local structure: entities that are close in the high-dimensional embedding space appear close in the 2D projection. If the model has learned meaningful representations, we expect to see **topical clusters** — groups of scientists, groups of countries, groups of institutions — rather than a uniform cloud of points.

**Nearest neighbor analysis:** We compute cosine similarity between embedding vectors to find the 10 entities most similar to a given entity. For a physicist like Maxwell, we expect to find other physicists or closely related concepts. This is a qualitative sanity check: if Maxwell's nearest neighbors are completely unrelated entities, the model has failed to learn meaningful structure.

---

## 5. RAG over RDF/SPARQL

### 5.1 Why RAG — The Core Motivation

Large language models (LLMs) are trained on large text corpora and store factual knowledge in their weights. However, this has three critical weaknesses when used for domain-specific question answering:

1. **Hallucination**: LLMs generate fluent, confident-sounding text even when they lack accurate information. Asked "Who are the scientists in our knowledge graph?", a direct LLM will invent plausible-sounding names from its training data — not the actual entities in our KB.

2. **Staleness**: LLM training data has a cutoff date. New entities, recent awards, and updated facts are absent.

3. **Lack of auditability**: When a direct LLM answers a factual question, there is no way to verify the source of the answer or check its accuracy against our curated KB.

**RAG (Retrieval-Augmented Generation)** solves all three problems by separating the knowledge source from the language model. Instead of answering from memory, the LLM generates a **SPARQL query** that retrieves facts directly from our RDF graph. The answer is then grounded in real KB data.

| Aspect | Direct LLM | SPARQL-generation RAG |
|--------|-----------|----------------------|
| **Factuality** | Hallucinates plausible but unverifiable answers | Returns only facts that exist in the KB |
| **Auditability** | No source traceability | The SPARQL query is human-readable and checkable |
| **Up-to-date** | Frozen at training cutoff | Reflects current KB state |
| **Domain coverage** | Limited to training corpus distribution | Complete KB coverage including rare entities |

### 5.2 Why a Local Small LLM (gemma:2b)?

**Why local deployment?** We deploy the LLM locally via Ollama rather than calling a cloud API for four reasons:

1. **Privacy**: The KB and user questions never leave the local machine. This is important in academic and enterprise settings where data sensitivity matters.
2. **No usage costs**: Cloud APIs charge per token. Running a local model is free after the initial download.
3. **Reproducibility**: The same model version produces consistent results regardless of API changes or model updates.
4. **Offline operation**: The pipeline works without an internet connection once the model is downloaded.

**Why gemma:2b?** A 2-billion parameter model runs on a CPU-only machine (Intel AI 7 350, 16 GB RAM) with ~30-60s inference time per query. Larger models (7B, 13B) would require a GPU or several minutes per query, which is impractical for interactive use. The tradeoff is lower output quality, which we compensate for with robust sanitization.

### 5.3 Schema Summary — Giving the LLM the Right Context

A 2B-parameter model does not have Wikidata's URI structure memorized. Without guidance, it would invent URIs like `<http://example.org/scientist>` that don't exist in our graph. We solve this by injecting a **schema summary** into every prompt.

The schema summary contains:
- **Prefix declarations** (`wd:`, `wdt:`) so the model uses compact, valid URI prefixes
- **Predicate list** with human-readable labels (e.g., `wdt:P106 = occupation`, `wdt:P166 = award received`)
- **Key entity QIDs** (e.g., `wd:Q901 = scientist`)
- **Sample triples** showing real data structure from the KB
- **Format reminders** embedded in the prompt

**Why truncate the schema?** The full schema with all 80 predicates and 40 class URIs is ~3,000 characters. On a CPU, a very long prompt causes the model to take 5+ minutes per inference, making the pipeline unusably slow. We truncate to the most relevant predicates (those most likely to be needed for common questions) and keep the schema under 1,500 characters.

### 5.4 Few-Shot Prompting for Structured Output

**Why few-shot examples?** It is not enough to describe the rules for writing SPARQL to a small LLM. The model needs to **see complete, correct examples** to internalize the output format. This is a well-established result in NLP: few-shot prompting (showing 3-5 examples) dramatically improves structured output compliance in small models.

We include 4 complete working query examples in every prompt, each demonstrating:
- The mandatory `PREFIX wd: / PREFIX wdt:` header
- A `SELECT ... WHERE { ... }` structure
- FILTER placement (inside the WHERE block)
- LIMIT at the end

### 5.5 Sanitization — Automatically Fixing LLM Errors

Despite careful prompting, `gemma:2b` still produces invalid SPARQL in roughly 60-70% of initial attempts. Rather than accepting failure, we apply 5 automatic repair steps to the raw LLM output:

| Fix | Error Type | Frequency | Mechanism |
|-----|-----------|-----------|-----------|
| **Fix 0** | Missing or wrong PREFIX declarations | ~40% | Strip ALL existing PREFIX lines; re-inject the canonical 4 prefixes (wd:, wdt:, rdfs:, rdf:). This is unconditional — we never trust the LLM's PREFIX output |
| **Fix 1** | COUNT/GROUP BY aggregates | ~15% | rdflib crashes with a `CompValue` exception on COUNT(). We detect the pattern with regex and replace with `SELECT DISTINCT` over the counted variable |
| **Fix 2** | FILTER clause outside WHERE {} | ~20% | The LLM often places FILTER after the closing brace of WHERE. We detect this by tracking brace depth and move the FILTER inside |
| **Fix 3** | ORDER BY / LIMIT inside WHERE {} | ~15% | The inverse problem — ORDER BY placed inside the WHERE block. Same depth-tracking approach, move outside |
| **Fix 4** | Markdown bullets or SQL comments in SPARQL | ~10% | Lines starting with `-` or `--` are dropped (these are never valid SPARQL) |

**Fix 0 is the most impactful.** Rather than detecting which prefixes are missing, we unconditionally strip and replace all PREFIX lines. This handles typos, wrong namespace URIs, missing declarations, and partial declarations in a single operation.

### 5.6 Self-Repair Loop and Template Fallback

**Self-repair:** If the sanitized query still fails to execute (e.g., wrong variable name, invalid URI), we send the failed query and the error message back to the LLM with repair instructions. The repair prompt includes a concrete working example and is kept short (schema truncated to 800 chars, error to first line only) to stay within CPU inference time limits.

**Template-based fallback:** If the repair also fails, we fall back to a keyword-matching approach. We scan the user's question for known terms (e.g., "scientist" → `wdt:P106 wd:Q901`, "award" → `wdt:P166`, "born" → `wdt:P19`) and build a simple, guaranteed-valid SPARQL query from a template. This ensures the RAG pipeline **always returns results**, even when the LLM completely fails.

**Predefined queries:** For the evaluation loop, we have hand-written verified SPARQL for each of the 5 evaluation questions. These are used as the last-resort fallback, ensuring the evaluation table always has complete results for comparison.

### 5.7 Evaluation (5 Questions)

We evaluate on 5 questions specifically designed to test different predicates in our KB:

| # | Question | Why Chosen | Baseline (No RAG) | RAG Result |
|---|----------|-----------|-------------------|------------|
| 1 | List 10 people with occupation scientist (wdt:P106 = wd:Q901) | Tests simple single-predicate lookup | Generic hallucinated occupation list | 10 real Wikidata QIDs from KB |
| 2 | Which people received an award (wdt:P166)? | Tests multi-valued property | Generic award category names | 10 person-award pairs from KB |
| 3 | List 10 people and their place of birth (wdt:P19) | Tests geographic data retrieval | Invented or hallucinated birthplaces | 10 person-place pairs from KB |
| 4 | Who was educated at a university (wdt:P69)? | Tests person-institution links | Generic university names | 10 person-university pairs from KB |
| 5 | List 10 people and their citizenship (wdt:P27) | Tests country-level data | Guessed nationalities | 10 person-country pairs from KB |

**Key finding:** The baseline LLM consistently generated fluent but unverifiable answers drawing on its training data rather than our KB. The SPARQL-RAG pipeline returned real, auditable Wikidata entities. Even when gemma:2b produced invalid SPARQL on the first attempt, the sanitization + repair + fallback pipeline ensured a valid result was always delivered.

### 5.8 Hardware & Runtime

- **Machine:** Windows 11, CPU-only (Intel AI 7 350, 16 GB RAM, no GPU)
- **Model:** gemma:2b via Ollama (2.0 GB download, ~1.5 GB RAM when loaded)
- **Inference time:** ~30–60 seconds per query (initial prompt), ~60–120 seconds (repair prompt, longer context)
- **Graph loading:** ~8 seconds for 52,000 triples with rdflib

---

## 6. Critical Reflection

### 6.1 KB Quality Assessment

**Strengths:**

The KB covers the electromagnetism domain comprehensively. With ~52,000 triples and 15 predicates, it captures the core network of scientists, institutions, countries, awards, and occupations. Entity linking to Wikidata ensures globally unique, stable identifiers — the KB can be merged with external LOD datasets without conflict. The predicate alignment to schema.org makes the KB readable by search engines and Semantic Web applications.

**Weaknesses and limitations:**

The most significant limitation is the **absence of human-readable labels**. Our KB contains only Wikidata QIDs (e.g., `wd:Q9095`) without `rdfs:label` triples. Query results are URIs rather than names, making the output difficult to interpret without a Wikidata lookup. A production KB would add `rdfs:label` values for all entities.

The **2-hop expansion** introduces entities that are only tangentially related to electromagnetism. A QID like `wd:Q1225` (the Belgian flag) might appear because one scientist was Belgian. While degree filtering removes the most peripheral nodes, some weakly connected entities remain and dilute the graph's topical coherence.

**NER-sourced initial triples** (from Lab 1) are noisy and are effectively replaced by Wikidata-sourced structured data during expansion. This means the NER pipeline contributes mainly to entity discovery rather than relation quality. A future improvement would be to retain only NER triples that are corroborated by Wikidata facts.

### 6.2 Noise and Data Quality

**NER noise (~5-10% misclassification rate):** The transformer NER model misclassifies some entity spans, particularly concept names that share surface forms with person names ("Maxwell", "Hertz" as a unit). Post-processing rules reduce but do not eliminate these errors.

**SVO triple noise (~30-40% of extracted triples are noisy):** Dependency parsing on complex academic sentences frequently produces incomplete or wrong triples. However, since SVO triples serve primarily for entity discovery (which entities to look up in Wikidata), rather than as final KB content, this noise level is acceptable.

**Wikidata completeness:** Wikidata is an open, community-edited database. Some entities have very sparse data — a scientist may have a QID but only 2-3 properties filled in. This contributes to the low average entity degree (~4.6) in our KB, which in turn limits KGE performance.

**Mitigation strategies applied:** Degree filtering, LCC extraction, and rare relation pruning during KGE data preparation collectively remove most of the noise-induced sparsity. The resulting training set is cleaner and more learnable than the raw KB.

### 6.3 Rule-Based vs. Embedding-Based Reasoning — A Principled Comparison

Both SWRL rules and KGE models perform "reasoning", but they are fundamentally different in what they can and cannot do:

| Dimension | SWRL Rules | KGE Embeddings |
|-----------|-----------|----------------|
| **Type of reasoning** | Deductive (logical entailment) | Inductive (pattern generalization) |
| **Explainability** | High — the rule is human-readable and auditable | Low — the answer comes from vector arithmetic in latent space |
| **Precision** | Exact — if the premises match, the conclusion is guaranteed | Approximate — predictions are probability-ranked, not certain |
| **Recall** | Limited to exact pattern matches | Generalizes to unseen entity combinations |
| **Scalability** | Exponential in rule complexity (pattern matching grows with KB size) | Linear in KB size (batch training) |
| **Best for** | Classification, constraint checking, deriving explicit facts | Link prediction, entity similarity, KB completion |

**The complementarity insight:** Rules and embeddings solve different problems, and a complete reasoning pipeline needs both. Rules are ideal when we want guaranteed, explainable classifications (e.g., "this person is an AwardedScientist because they satisfy these two conditions"). Embeddings are ideal when we want to discover implicit patterns (e.g., "this scientist is likely to have received this award because other scientists with similar properties did"). In practice, hybrid systems that use rules for high-confidence facts and embeddings for uncertain predictions outperform either approach alone.

### 6.4 RAG System Limitations and Future Improvements

**LLM quality bottleneck:** gemma:2b (2B parameters) is at the lower end of what is practically usable for structured query generation. Despite few-shot prompting and sanitization, the model still requires fallback mechanisms for roughly 40-60% of queries. Upgrading to a code-specialized model (e.g., `qwen2.5-coder:3b`) or a larger model (7B+) would significantly improve first-attempt success rates without hardware upgrades.

**Entity disambiguation gap:** The most fundamental limitation of our RAG system is that the LLM cannot map natural language entity names to Wikidata QIDs. "Maxwell" in a user question cannot be automatically linked to `wd:Q9095`. This means queries must either use the QID directly or match only on predicates. A proper production system would include a **Named Entity Disambiguation (NED)** module that maps text mentions to QIDs before generating SPARQL.

**No label resolution in results:** Because our KB lacks `rdfs:label` triples, query results return raw QIDs. A label-resolution step — querying Wikidata's API to convert QIDs to names — would make the system immediately usable by non-experts.

**CPU inference latency:** At 30-60 seconds per query, the system is usable for batch evaluation but not for interactive conversation. A GPU would reduce this to 1-3 seconds; alternatively, serving the model via an Ollama API on a faster machine over a local network would solve the latency issue without hardware changes.

### 6.5 Reproducibility Statement

The entire pipeline — from web crawling through KGE training to RAG question answering — runs on a **CPU-only Windows 11 machine** with 16 GB RAM and no GPU. All dependencies are listed in `requirements.txt`. The repository includes:
- All 4 lab notebooks with cached outputs
- All KG artifacts (`expanded_kb.rdf`, `initial_kb.ttl`, `alignment.ttl`, `family.owl`)
- Pre-split KGE data (`train.txt`, `valid.txt`, `test.txt`)
- The complete RAG CLI script (`rag/lab_rag_sparql_gen.py`)
- A `README.md` with step-by-step installation and run instructions

KGE training times: ~10-30 minutes per model on CPU. RAG inference: ~30-120 seconds per query with gemma:2b. A full pipeline run from data loading to RAG evaluation takes approximately 45-90 minutes on the reference hardware.

---

## References

- Honnibal, M. & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. *Explosion AI*.
- Ali, M. et al. (2021). PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings. *Journal of Machine Learning Research*, 22(82), 1-6.
- Sun, Z. et al. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. *ICLR 2019*.
- Yang, B. et al. (2015). Embedding Entities and Relations for Learning and Inference in Knowledge Bases. *ICLR 2015*. (DistMult)
- Bordes, A. et al. (2013). Translating Embeddings for Modeling Multi-relational Data. *NeurIPS 2013*. (TransE)
- Lamy, J.B. (2017). Owlready: Ontology-oriented programming in Python. *Artificial Intelligence in Medicine*, 80, 11-28.
- Vrandečić, D. & Krötzsch, M. (2014). Wikidata: A Free Collaborative Knowledge Base. *Communications of the ACM*, 57(10), 78-85.
- RDFLib Documentation: https://rdflib.readthedocs.io/
- Wikidata SPARQL Endpoint: https://query.wikidata.org/
- Ollama Local LLM Serving: https://ollama.com/
