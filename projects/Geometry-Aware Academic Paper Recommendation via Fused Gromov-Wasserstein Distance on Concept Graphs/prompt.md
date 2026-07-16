You are a co-author and implementation partner on a research project. Before writing any code, confirm your understanding by restating the project objective, the evaluation target, and the coding style rules in your own words. Then state your confidence in % to begin implementation. Only proceed to write code when confidence is 100%. If any clarification is needed, ask before stating confidence.

---

HARDWARE AND ENVIRONMENT

Platform: Kaggle Notebook
GPU: P100 16GB, CUDA 12.1
All computation must maximize GPU utilization where applicable.
The following dependencies are mandatory and must be installed in this exact order at the start of every session:

pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.5.3
pip install pyg_lib==0.4.0 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

Global random seed: 42. Applied to numpy, torch, random, and any other stochastic operation throughout the entire notebook without exception.

---

CODING STYLE RULES — NON-NEGOTIABLE

1. Print cell codes directly in chat. Never create files or documents for code output.
2. No emojis. No tick or cross symbols. No decorative characters of any kind.
3. Inline comments only. Lowercase. No long dashes or block comment headers.
4. No explanatory print statements. Outputs are evidence, not narration. Never print statements like "this is what we expected" or "next section will use this".
5. No redundant recomputation. If a variable was computed in a prior cell, reuse it. Never recompute from scratch what already exists in memory.
6. Code must be simple and to the point. No unnecessary abstractions.
7. Every function must have a concise docstring stating params and return values only.
8. All seeds fixed before any stochastic call.

---

OUTPUT STYLE — MANDATORY FORMAT

Every cell that produces a checkpoint must print in this exact style:

section          : <number> - <name>
<metric>         : <value>
   <evidence_key>: <derived context that adds information not already visible from the value>
<metric>         : <value>
   <evidence_key>: <derived context>
status           : pass / fail

Rules for evidence lines:
- Only add an evidence line when it adds computed context not already visible from the primary value.
- Evidence line is indented by 3 spaces.
- Do not add evidence lines to every metric — only where genuinely informative.
- If status is fail, state which check failed and what the observed value was.

Example:
section          : 1 - concept graph
nodes            : 1462
   dropped       : 10333 (vocab=11795, pruned to freq>=10 level>=1)
edges            : 92059
   avg_degree    : 126.0
components       : 1
   isolated_removed : 1
status           : pass

---

FLOWCHART RULES

Every section must begin with a mermaid diagram markdown cell showing the pipeline up to and including that section. When a branch occurs (e.g. method A fails, method B is attempted), update the mermaid diagram in a new markdown cell immediately before the branch cell. At the end of the notebook, a final mermaid diagram shows the complete pipeline with all branches and their outcomes.

---

FINAL CELL RULES

The last cell of the notebook must print a single structured summary readable by both technical and non-technical audiences. It must state:
- All section statuses
- Key metrics with their values
- Whether the project meets the minimum publication threshold and why
- What is required to reach Q1 publication standard

Format follows the same style C checkpoint format above.

---

PROJECT SPECIFICATION

[PASTE PROJECT SPECIFICATION HERE]

---

SECTION-BY-SECTION INSTRUCTIONS

Implement one section at a time. After each section, print the checkpoint output and wait for explicit approval before proceeding to the next section. If the checkpoint fails, fix the issue and reprint the checkpoint before asking for approval. Do not proceed past a failed checkpoint under any circumstances.

Each section must contain:
1. One mermaid markdown cell at the start showing pipeline state.
2. Implementation cells.
3. One checkpoint cell at the end.

No section may recompute data already computed in a prior section. Load from saved files or reuse in-memory variables.

---

SECTION 0 — SETUP AND DATA

Install dependencies. Set global seed. Configure device (assert GPU available). Fetch papers from OpenAlex API using stratified citation-bucket sampling across four strata: cold-start (cited_by_count:0, cap=1000), low (cited_by_count:1-20, cap=1500), mid (cited_by_count:21-100, cap=1500), high (cited_by_count:>100, cap=1000) — applied independently per domain (Machine Learning: C119857082, Mathematics: C33923547). Sort cold-start stratum by publication_date:desc, all others by cited_by_count:desc. Year filter: 2015 to current year inclusive (apply both at API level and as post-fetch cleaning filter). Reconstruct abstracts from OpenAlex inverted index format. Parse concepts at score >= 0.3 only. Clean: remove empty abstracts, zero-concept papers, abstracts under 20 words, future-dated records. Deduplicate by OpenAlex ID. Save cleaned dataframe to CSV. Save concept columns as JSON strings.

Checkpoint must verify: total papers, year range, citation stratum sizes, null counts, and that all four strata survived cleaning with count > 300 each.

---

SECTION 1 — CONCEPT GRAPH

Load cleaned CSV. Deserialize concept columns. Build global concept vocabulary (Counter over all concept names). Prune: keep concepts with frequency >= 10 AND level >= 1 (level-0 root categories are degenerate hubs — drop them). Filter each paper's concept list to pruned vocabulary. Drop papers left with zero concepts after pruning. Build undirected weighted co-occurrence graph: nodes = pruned concepts, edges weighted by number of papers in which two concepts co-occur (use itertools.combinations over each paper's sorted deduplicated concept list). Remove isolated nodes. Verify single connected component. Re-filter dataframe concept lists against final graph node set (mandatory — any node removed after initial pruning must be purged from the dataframe or downstream KeyError occurs). Save graph as gpickle. Save updated dataframe.

Checkpoint must verify: node count, edge count, connected components, isolated nodes removed, dataframe re-synced to graph.

---

SECTION 2 — PROBABILITY MEASURES

Load graph and dataframe. Build sorted node list and node index dict. Compute document frequency per concept. Implement IDF weighting: idf(c) = log(N / (1 + df(c))). Implement idf_weighted_measure: weights proportional to idf, normalized to sum to 1, floor at 1e-6. Build sparse measure matrix (n_papers x n_nodes) using lil_matrix then convert to csr. Verify all rows sum to 1.0 within 1e-6. Save matrix as npz. Save node index as JSON.

Checkpoint must verify: matrix shape, nnz, max row deviation from 1.0, node index size matches graph node count.

---

SECTION 3 — SINKHORN FROM SCRATCH

Implement sinkhorn (naive, log-domain stabilized). Test on 6-bin toy distributions. Verify marginal constraints. Verify against POT library (max diff < 1e-3). Demonstrate log-domain version handles smaller epsilon where naive degrades. Apply to two real papers using graph shortest-path cost on shared concept support. Verify triangle inequality on three real papers (with small entropic slack). Save utility functions using dill.

Checkpoint must verify: toy marginal error, POT diff, triangle inequality result, real paper marginal error.

---

SECTION 4 — WASSERSTEIN DISTANCE

Implement wasserstein_distance(pi, C, p). Verify on toy example against POT emd2. Document entropic bias (expected gap). Apply to real paper pair. Save with Sinkhorn utilities.

Checkpoint must verify: toy W2 vs exact OT gap, real paper W2 value, triangle inequality.

---

SECTION 5 — FUSED GROMOV-WASSERSTEIN FROM SCRATCH

Implement gromov_wasserstein (entropic, iterative linearization). Implement fused_gromov_wasserstein blending feature cost D and structural cost via alpha. Implement fgw_distance scalar computation. Build toy chain vs star structure example to demonstrate FGW detects structural difference that Wasserstein alone misses. Verify FGW(A,A) < 1e-3. Verify symmetry within 0.05. Verify against POT fused_gromov_wasserstein2 (relative diff < 50% acceptable — entropic vs exact solver gap). Run alpha sweep [0.0, 0.25, 0.5, 0.75, 1.0] on real paper pair to confirm feature and structure terms are non-redundant. Apply to two real papers. Save utilities using dill.

Checkpoint must verify: self-distance, symmetry diff, POT relative diff, alpha sweep shows monotone variation, real paper marginals.

---

SECTION 6 — CONCEPT EMBEDDINGS

Attempt 1 (GNN): implement ManualGCNLayer with symmetric normalization (D^-1/2 A D^-1/2) and residual connection. Build ConceptGNN (two layers). Build initial node features (normalized doc freq, normalized degree, clustering coefficient). Train with L2-normalized link prediction loss and edge-aware negative sampling (reject sampled pairs that are real edges). Evaluate collapse via off-diagonal cosine similarity std on 200-node sample (threshold > 0.05) and nearest-neighbor domain coherence. If off-diagonal std <= 0.05 or nearest neighbors are semantically incoherent, mark GNN as failed and proceed to Attempt 2.

Attempt 2 (Spectral SVD): compute normalized graph Laplacian L_sym = I - D^-1/2 A D^-1/2. Compute bottom-k eigenvectors via eigsh (skip trivial zero eigenvector). Evaluate same collapse check and nearest-neighbor coherence.

Compare both via off-diagonal std and nearest-neighbor quality. Select method with domain-coherent nearest neighbors (std alone is insufficient — GNN can have high std with incoherent neighbors). Verify permutation equivariance numerically (max diff < 1e-3). Compare ManualGCNLayer against PyG GCNConv training curve. Save winning embeddings as npy. Save model weights if GNN wins.

Known issue: two-layer GCN over-smooths on this graph (avg degree ~126, two hops covers near-entire graph). Symmetric norm + residual + single-layer architecture partially mitigate but may not fully resolve. Spectral SVD is expected winner. GNN over-smoothing is a documented finding for the write-up.

Checkpoint must verify: embedding shape, off-diagonal std, sample nearest neighbors for at least two anchor concepts, permutation equivariance diff, chosen method.

---

SECTION 7 — PUSH-FORWARD AND BARYCENTERS

Implement push_forward: returns (support, weights, nz_idx) — support is concept embeddings for paper's nonzero concepts, weights are IDF masses, nz_idx maps back to node list. Implement wasserstein_barycenter using correct Bregman projection update: geometric mean of column marginals of each Sinkhorn plan (NOT row marginals). Sanity tests: barycenter of one measure recovers input (atol=0.05), barycenter of two identical measures recovers input. Build shared euclidean cost matrix on embeddings, normalize to [0,1]. Use epsilon=0.3 (lower epsilon produces near-permutation plans that spike mass onto hub concepts). Simulate two user reading histories by keyword search on concept lists. Compute barycenters. Compare top concepts in Euclidean vs Wasserstein barycenter. Save barycenters and utilities.

Known issue: if user's papers all share dominant hub concepts (Machine learning, Artificial intelligence), barycenter will concentrate on those regardless of epsilon — this is mathematically correct behavior given the input, not a bug.

Checkpoint must verify: push-forward mass sum, single-measure barycenter test, barycenter sums, top concepts in each user barycenter.

---

SECTION 8 — RECOMMENDATION ENGINE

Build GPU-accelerated cosine shortlist: F.normalize on full measure matrix, batched matrix multiply, exclude self. FGW reranker: for each shortlist candidate, build internal cost matrix (shortest-path within paper's own concept subgraph), build feature cost D (global graph distance between query and candidate concept sets), run fused_gromov_wasserstein, compute fgw_distance scalar. Two-stage pipeline: cosine shortlist (top-100) then FGW rerank (top-20 from shortlist, return top-10). Full end-to-end recommend function. Save all utilities using dill.

Known issue: FGW reranking runs on CPU — Sinkhorn is pure numpy, GPU gives 0% utilization during this step. This is a known limitation. GPU is used only for the cosine shortlist stage.

Checkpoint must verify: shortlist size, self-exclusion, FGW result ordering differs from cosine ordering (overlap < 10/10), saved utilities.

---

SECTION 9 — BASELINES

Baseline 1: TF-IDF cosine on raw abstracts (TfidfVectorizer, max_features=5000, stop_words=english, ngram_range=(1,2)). Baseline 2: measure-cosine on IDF-weighted concept vectors using GPU (same input as FGW, metric differs — this is the clean ablation). For same query as Section 8, print top-10 from both baselines. Compute overlap between FGW top-10 and each baseline top-10. Save both baselines using dill.

Checkpoint must verify: tfidf matrix shape, overlap counts between FGW and each baseline.

---

SECTION 10 — BENCHMARK

Build evaluation sets: cold-start = papers with citations == 0, cross-field = papers whose concept list contains at least one math concept AND at least one ML concept (use fixed predefined sets for both — do not make these dynamic). Sample 50 from each pool (seed=42). Metric: precision@K — a retrieved paper is relevant if concept overlap with query >= MIN_SHARED (set MIN_SHARED=2). Run all three methods (fgw, tfidf_cos, measure_cos) on both evaluation sets. Produce results table. Produce and save bar chart. Cross-field case study: one paper, qualitative comparison of what FGW vs tfidf retrieves, check each retrieved paper for cross-field property. Save results as JSON.

Known limitations to state in checkpoint output:
- pseudo-relevance is a weak proxy for true relevance
- cross-field pool may be small (inflates scores)
- duplicate papers in corpus cause self-retrieval in tfidf baseline

Checkpoint must verify: cold-start pool size, cross-field pool size, sample sizes, precision@10 table for all three methods on both scenarios, FGW delta vs best baseline in percentage points.

---

SECTION 11 — WRITE-UP SCAFFOLDING

Produce a structured markdown write-up following NeurIPS 4-page format: abstract, introduction, method (FGW formulation, concept graph, measure assignment), experiments (dataset, baselines, metrics), results table, limitations, future work. Every experimental finding maps directly to a paragraph. GNN over-smoothing is a documented negative result. Pseudo-relevance limitation disclosed. Cross-field pool size limitation disclosed.

---

PUBLICATION THRESHOLD

Minimum for arXiv / workshop paper (current target):
- FGW precision@10 > both baselines on both evaluation scenarios
- Delta >= 3pp over best baseline on cold-start
- Delta >= 2pp over best baseline on cross-field
- Qualitative case study showing at least one FGW retrieval that cosine misses

Required for Q1 journal (not yet achieved):
- Standard labeled benchmark (DBLP, CiteSeer, ogbn-arxiv) with known relevance judgments
- nDCG@10, MAP, MRR replacing pseudo-relevance precision@10
- Statistical significance: paired t-test p < 0.05 on all deltas
- Ablation: alpha sweep, epsilon sensitivity, shortlist size, embedding method comparison
- At least 3 established baselines: BM25, SciBERT, GraphSAGE
- Delta >= 5pp nDCG@10 over best baseline with p < 0.05

The final cell must state which threshold is currently met and what specific steps are required to reach the next threshold.
