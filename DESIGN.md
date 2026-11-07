# Exact worlds, explicit causal assumptions

## Mechanisms are inputs, not discoveries

A version-1 model supplies independent finite exogenous distributions and topologically ordered deterministic endogenous equations. Sharing an exogenous name across equations means sharing the same draw, not independent copies. The graph, functional tables and noise distributions are assumptions. Enumeration answers queries conditional on those assumptions; it cannot establish them from an observational dataset.

The internal `Model` contains frozen tuple-backed noise and node records. `Model.from_dict` validates dimensions, names, strict fields, probabilities and resource limits. Queries revalidate supplied models, even preconstructed records. `as_dict` returns fresh lists. Input probability sums must be within 1e-12 of one and are normalized by their actual sum; arbitrary unnormalized weights are not accepted. Canonical sorted JSON over that normalized model produces a SHA-256 identity checksum.

## Table indexing

For an input list with cardinalities c0,c1,... and corresponding values v0,v1,..., start index=0 and repeatedly set `index = index*cardinality + value`. The final input changes fastest. A table with no inputs contains one constant output. An intervention bypasses indexing for its target node and assigns the requested constant, removing **all** of that node's incoming influences. Other equations are unchanged.

Exogenous values are fixed for a complete world. Every endogenous node is evaluated once in topological order. Nodes may still depend on an intervened node's new value, so the downstream effect propagates normally. Nodes earlier than an intervention or otherwise unaffected retain their original mechanisms, rather than being reconditioned on the action as an observed event.

## Ordinary query versus counterfactual

For each positive-prior exogenous assignment u, its log prior is the sum of named-noise log probabilities. Zero-probability states contribute no positive-prior worlds, although they count against Cartesian-slot bounds.

For an ordinary query:

1. Evaluate the model under the requested interventions.
2. Keep worlds whose **post-action** endogenous values match evidence.
3. Accumulate the requested output tuple using those worlds' prior masses and normalize by evidence mass.

No interventions means ordinary observational conditioning.

For a counterfactual:

1. Evaluate the **unmodified factual model** for every positive-prior assignment.
2. Keep assignments whose factual values match evidence. This is abduction.
3. Evaluate the intervened model with each **same assignment**, not a new noise sample.
4. Normalize predicted output masses by factual evidence mass. Do not multiply by evidence again.

The mechanism is deterministic conditional on u, so the posterior weight on a retained assignment is just its prior mass divided by total retained mass. With no factual evidence, counterfactual prediction matches intervention prediction. With no action, it matches observational conditioning. If the factual action value already holds, imposing that same action leaves the observed conditional outcome distribution unchanged. Tests exercise these identities, including degenerate mechanisms.

The original `infer` API accepts one action world plus factual evidence. The separate `joint_counterfactual` API extends this to 1..4 independently specified named action worlds. Abduct once using the unmodified model, evaluate each action with that retained assignment, concatenate the requested values across scenarios, and group by the **entire tuple** before normalizing by factual evidence mass. There is one prior weight per assignment, not one per scenario; no second evidence multiplication and no multiplication of marginals. Empty actions reuse the same factual equations, so repeated worlds are perfectly coupled. No arbitrary nested counterfactual formulas, cross-world evidence or general twin-network identification algorithm is implemented.

A scenario has exactly `name`, `query`, `do`; name uniqueness, each query and action, combined output Cartesian size (at most 4,096 rows) and trace work are checked before evaluation. Queries are nested under names in each result row, so the same node can appear in multiple worlds without key collision. Scenario order and query order determine enumeration order. The joint trace records factual values and a separate fresh prediction dictionary for every scenario.

## Numerical semantics

World weights are kept as finite log priors. Query/evidence support is determined by structural equality and strictly positive noise probabilities, not an exponentiated mass threshold. Groups of compatible worlds use log-sum-exp; output probabilities use relative exponentials against the largest matching log prior.

Thus a mass such as `(1e-300)^4` can have reported probability 0 while its log remains finite. Conditioning on that possible event still produces a normalized distribution. Structural zero support is a separate error for evidence, or `probability: 0, log_probability: null` for an impossible output row. A finite `log_probability` and stored probability 0 means underflow, not impossibility. Rounding of logs, sums and exponentials remains; this is not rational symbolic arithmetic. Tiny positive masses may be lost in sums against much larger masses.

Impossible evidence is not automatically softened or repaired. The engine rejects it, rather than inventing an epsilon, returning a uniform distribution or changing the model. Counterfactual factual evidence and ordinary post-action evidence deliberately have different failure contexts in error messages.

## Work, memory and output bounds

Let W be the Cartesian noise-world slots, E the number of exogenous variables, N the number of endogenous nodes, I the total number of mechanism inputs, K the output joint rows and M the positive-prior worlds matching evidence.

- Model validation costs the table entries plus node/input metadata, before enumeration.
- An ordinary query evaluates at most W worlds. A counterfactual evaluates at most W factual worlds and M action worlds. Each evaluation costs `O(E+N+I)`, including model-cardinality lookups and truth-table indexing. There is no variable-elimination or pruning claim.
- Aggregation retains at most W log masses in groups plus an additional at-most-W matching-log list and K output rows: `O(W+K+E+N+I)` working memory, apart from the input tables.
- Traces add `O(M*(E+N))` output/memory for noise assignments and factual/predicted values. The trace preflight cap conservatively uses `W*N` times one or two worlds, regardless of zero-probability bins or evidence selectivity.
- The maximum W is 65,536, not a promise of constant latency. Max tables, inputs, query rows and trace bounds are in the README. Counterfactual mode can do more work than an intervention query with the same returned distribution.

For the multi-world API, let S be scenario count (1..4) and D the total number of requested coordinates across scenarios (at most 32). It performs at most `W` factual and `S*M` action evaluations. Joint-key construction/output costs `O(S*M*N + K*D)` as a conservative bound since each query length is at most N; model evaluations cost `O((W+S*M)*(E+N+I))`. Grouped log lists hold at most `W` entries, not `S*W` entries. Non-trace working/output memory is `O(W+K*D+S*N+E+I)` excluding input tables. Traces add `O(M*(E+(S+1)*N))`. Its preflight uses `W*N*(S+1) <= 32768`, even if evidence would select only one world. Outputs count every complete scenario evaluation, including repeated or empty actions: no caching or speed claim.

Outputs expose actual factual/action evaluation counts. In ordinary mode, the evaluated model is counted in the action column even when no intervention was supplied. These counts measure complete model evaluations, not table operations, probability normalizations, machine instructions or wall-clock performance. The demo's 100-world example is exhaustive, not a sample of 100 people.

## Independent checks

The confounded fixture is checked against a separate `Fraction` loop over all integer noise assignments, using direct context/action/outcome conditions rather than the production evaluator. A second test covers all 4 binary X mechanisms and 16 binary Y mechanisms over two independent binary noise variables, comparing intervention and factual-evidence counterfactual answers to rational unit-level enumeration. This is 64 mechanism pairs, not a proof for arbitrary models.

Other tests cover nonbinary mixed-radix order, shared versus independent noise semantics, replacement of multiple equations, joint marginalization/query order, observational-equivalence counterexamples, the abduction/action/prediction consistency identities, log-space rare evidence, impossible events and trace posterior reconstruction. The world-cap boundary is executed at 65,536 slots. Oversized products/tables/query outputs/traces fail before any equation evaluation. CLI tests reject malformed/duplicate-key/oversized input and repeated assignment names without partial output.

The figure generator requires the complete current deterministic demo and rejects changed probabilities, mechanisms, traces or checks. Its eight bars share the [0,1] domain. Tests verify every coordinate/value and the full numeric/model/trace metadata, in addition to byte-identical regeneration. This is a reproducibility contract for fixed development counterexamples, not an empirical benchmark or source of causal validity.

## Cross-world coupling counterexample

The two new four-world fixtures share independent U,V with P(0)=P(1)=.5 and X=V. Stable Y=U gives `(Y0,Y1)=(U,U)`; flip Y=X xor U gives `(Y0,Y1)=(U,1-U)`. V randomizes X independently of U, so every observed `(X,Y)` pair has probability .25 in both. Under each fixed action, both mechanisms return Y=1 with probability .5. Therefore even the complete observed joint and all single-action marginals agree. Stable has response masses (.5,0,0,.5), while flip has (0,.5,.5,0), in order (00,01,10,11). A product of action marginals would give .25 to every response type, incorrectly imposing independent potential outcomes.

For factual X=0,Y=0, both fixtures require U=0,V=0, but stable predicts (0,0) and flip predicts (0,1). These are consequences of two explicit structural assumptions, not identified response-type estimates. No monotonicity is assumed. The fixtures are first hand-specified examples, not selected through mechanism or seed search. The complete demo includes the mechanisms, observations, intervention results, both joint queries and traces, all response masses and the naive product. The second SVG requires that complete unchanged demo; its eight bars and .25 reference lines use one fixed [0,1] scale.

Multi-world tests independently enumerate all 64 Boolean mechanism pairs using rational masses and direct table indexing under two actions, with empty and selected factual evidence. They also check each joint marginal against original single-world inference, repeated-world diagonal support, scenario/query permutation, nonbinary and multiple-node actions, no cross-scenario contamination, trace reconstruction/copy isolation, rare possible support and impossible evidence. Combined bounds are preflighted, and both 4,096 joint rows and the 32,768 trace node-slot boundary are actually executed. Figure tests check all coordinates, metadata/models/traces, changed-input rejection and byte-identical regeneration. The multi-world stage brings the suite to 64 tests before the sharp-bound extension below.

## Sharp binary response bounds from marginals

The multi-world API uses a specified SCM coupling. The independent `response_bounds(p0,p1,monotone=False)` calculation considers **every** binary response joint with the two supplied action marginals. It does not infer those marginals from observations or import a chosen SCM's dependence.

Let a=p0, b=p1 and t=P(Y0=1,Y1=1). The four cells in (00,01,10,11) order must be `(1-a-b+t, b-t, a-t, t)`. Their sum is one; their Y0/Y1 marginals are a/b. Nonnegativity requires and is sufficient for `max(0,a+b-1) <= t <= min(a,b)`. Thus the two endpoint witnesses are feasible and their convex mixtures cover every compatible joint. This is a one-dimensional feasible set, not a numerically solved general linear program.

For each t:

- Helped is `b-t`; harmed is `a-t`; changed is `a+b-2*t`; unchanged is `1-a-b+2*t`.
- Mean effect is `b-a`, independent of t. Positive mean alone does not rule out harmed units.
- Since D=Y1-Y0 is -1,0,1, its second moment is the changed fraction. Therefore `Var(D)=a+b-2*t-(b-a)**2`.
- Covariance is `t-a*b`.
- Sufficiency is `(b-t)/(1-a)` when a<1; necessity is `(b-t)/b` when b>0. These are defined cross-world conditional probabilities, not estimates from people selected into an observational action group. A zero denominator makes the quantity undefined, not zero.

Every listed defined quantity is affine in t; evaluating its two endpoint values therefore gives sharp lower/upper bounds. Each endpoint names the witness that attains it. Maxima/minima of different quantities can require different witnesses; the report does not claim all interval endpoints jointly attainable. The API does not bound arbitrary nonlinear objectives.

Monotonicity means Y1>=Y0 for every positive-mass response type. It sets the harmed cell to zero, so t=a. This is feasible exactly when b>=a. It then fixes the unique joint `(1-b, b-a, 0, a)`. When a=b it forces every unit unchanged, although the unrestricted equal-marginal model allows all units changed. The API rejects b<a exactly, without tolerance or silently moving a marginal. Monotonicity is optional and visible in output, never implied by b>=a alone.

All arithmetic above uses `Fraction` over the supplied finite Python int/float. This is exact for the represented binary value, not a claim that decimal text such as .1 equals 1/10. Exact fraction strings accompany every displayed input, bound, parameter endpoint, witness cell and witness marginal. Display floats can round, including a signed zero for a tiny negative covariance; the fraction remains authoritative. Classification of impossible conditioning uses exact zero, not an exponentiated threshold. The calculation uses a fixed number of rational operations; integer bit complexity is not claimed constant. It does not enumerate noise worlds or use the SCM world/trace caps.

The independent bound reference enumerates every four-cell vector with denominator eight: 165 joints, grouped by all 81 marginal pairs. For each pair and each assumption mode it computes empirical extrema of response-cell probabilities, direct centered second-moment variance/covariance and conditional ratios from the joint cells, then compares those exact rational extrema to production. Endpoints of these grid marginals are on the same grid, so this enumeration can verify sharpness, not merely containment. Separate tests verify every witness's nonnegativity, total mass, exact action marginals and attainment of its named endpoint, convex interior mixtures, action relabeling, degenerate corners, smallest-subnormal positive denominators, underflowed float display with retained exact fraction, incompatible monotonicity and strict CLI behavior. The original stable/flip SCMs are matched to the two .5/.5 endpoint witnesses.

The third deterministic demo uses .5/.5 to connect to the existing mechanisms and .25/.75 to expose positive mean with possible negative unit responses. No data/seed/mechanism search selects these quarters. Its six stacked columns are explicit attaining witnesses, not confidence bands or sampled population estimates. All 24 component masses, exact joint and range metadata, assumption flags and fixed [0,1] domains are verified, with full unchanged-demo checks and byte-identical SVG regeneration. The sharp-bound stage brings the suite to 90 tests before the graphical extension below.

## DAG separation and a moral-graph certificate

The new graph-only format supplies unique named nodes and directed edges. Validation caps 32 nodes/128 edges before processing, rejects duplicate/unknown/self edges, and uses a Kahn traversal to reject cycles without requiring input topological order. Canonical node/edge order makes identity hashes and output independent of input ordering. `graph_from_model` first revalidates the full SCM, then includes every exogenous root plus every equation input edge, including shared latent causes. It does not marginalize latent nodes or pretend the endogenous-only graph is sufficient.

For disjoint nonempty X/Y sets and conditioning set Z, optional intervention targets remove all incoming arrows first. In that mutilated DAG, take the ancestor closure A of X union Y union Z, including those sets themselves. Restrict to A, make directed adjacencies undirected and join each pair of parents of every retained child. Only after ancestor restriction and moralization are Z and its incident moral edges removed. Moralizing the full original graph first would wrongly open an unconditioned collider whose child is not ancestral to the query; the tests explicitly catch this ordering error.

A deterministic multi-source breadth-first search from X records every reachable node and a predecessor map. X/Y are d-separated iff no Y node is reachable. A connected result reconstructs one path to the first reachable target in canonical Y order. Its edges carry all original directed/collider-child origins; coparent bridges can skip a collider node, so it is explicitly a moral-graph witness and not an original-DAG active path. A separated result's complete reachable set contains X, excludes Y/Z and is closed under surviving adjacency, which supplies a checkable cut certificate. Results retain original/effective edges/hashes, removed arrows, ancestors/pruned nodes, moral origins, conditioned-node removal and all reachable nodes.

With V nodes, E effective edges and d_c parents of child c, graph construction takes O(V+E+sum_c d_c^2) operations apart from canonical sorting. The moral graph can be dense with O(V^2) edges; origin provenance can require O(E+sum_c d_c^2) records. Production never enumerates paths, worlds or truth-table assignments. Query sets are list/tuple inputs with unique known names and pairwise disjoint X/Y/Z; cuts may overlap query or conditioning sets because surgery and conditioning are distinct operations. Cuts specify graph targets only, not assigned values. A fixed intervention can make a target constant and create extra distributional independences not captured by this graphical sufficient rule.

The independent reference does not moralize or build the ancestor query closure. It constructs the underlying undirected skeleton, enumerates simple X/Y paths, and tests each internal vertex directly: noncolliders must avoid Z, while colliders must be in Z or have a directed descendant in Z. It calculates descendant sets from outgoing arrows after the same specified surgery, using no production helper. Across all 64 edge subsets consistent with one four-node topological order, 49,152 singleton-pair/conditioning/all-surgery queries and 7,040 disjoint set queries agree. These are exhaustive checks of that finite family, not a proof for every graph. Non-topologically ordered input, multiple sources/targets, multiple moral-edge origins, all eight demo certificates, 32-node/128-edge boundaries, invalid graphs/queries, conversion and strict CLI errors are also tested.

The fourth fixed SVG displays eight chain/fork/collider/surgery answers and moral witnesses without presenting measured correlations. Its tests verify every status, graph edge, arrow endpoint, node position, removed-edge styling, complete unchanged demo metadata and byte-identical generation. The graphical stage brings the suite to 115 tests before the backdoor extension below. In a separate fixture, X and Y are graph-connected but a fair-noise XOR mechanism makes their computed joint uniform, directly checking the warning that connection does not imply dependence. No graph search selects the motifs and no faithfulness is assumed.

The criterion follows the directed global Markov property and ancestral-moral equivalence described in [Lauritzen's author-hosted Causal Inference from Graphical Models](https://web.math.ku.dk/~richard/BSc/Lauritzen.pdf). Its implication is conditional on a correctly supplied Markov DAG, including relevant latent causes. It neither tests observational independence nor identifies causal structure or probabilities. All implementation, fixtures and diagrams are original to this repository.

## A sufficient backdoor criterion and bounded subset search

The backdoor checker takes one treatment X, one distinct outcome Y and an adjustment set Z excluding those endpoints. It computes strict descendants of X in the original DAG. Condition one is Z intersect descendants(X) empty. Separately it removes every outgoing arrow of X and calls ancestral-moral d-separation of X/Y given Z in that graph. Both conditions and their evidence remain visible even when one fails. Outgoing-edge deletion is specific to this sufficient criterion; it is not do(X), which cuts incoming arrows. Original descendants must be computed before deleting outgoing edges, or a mediator would wrongly escape exclusion.

The single-set result includes both conditions, the original graph/hash, removed outgoing edges and the full moralization/cut certificate. A formula string is included only for a passing set, with the explicit qualification that the graph's causal assumptions and observed support/positivity must hold. No probabilities or data estimates are computed and no data availability is inferred. Standard backdoor descendant exclusion is intentionally conservative: rejection is not a claim that no more general adjustment functional could work.

The set search requires an explicit available-candidate list with at most 10 names. It never silently adds a latent common cause to that list. It enumerates all 2^k subsets in cardinality then canonical-name order, including empty adjustment, without early exit or upward-closure pruning. Validity is not monotone under inclusion: conditioning on a collider can turn a valid empty set into an invalid singleton. Every tested set gets compact condition/ancestor/reachability/witness evidence; the original and outgoing-cut graphs are shared once rather than copying full moral provenance into every row. Full provenance remains recoverable with the single-set API.

All passing sets are returned. To find inclusion-minimal ones, scan cardinality-first passing sets and retain a set only if it contains no already-retained minimal set. This is sound because any valid strict subset contains an inclusion-minimal valid subset, whether or not intervening supersets are valid. The minimum cardinality is reported separately. With k candidates the graph-check work is 2^k times a separation calculation; a simple minimality pass can take O(4^k*k). Both work and output are exponential in k, which is why the search preflights the 10-name cap. The single-set checker is not capped at 10 and can inspect all 30 non-endpoint nodes of a 32-node graph.

The independent reference evaluates the original graph, not the outgoing-cut graph: enumerate simple paths that begin with an arrow into X, then test each local noncollider/collider using independently computed directed descendant sets. Exclude original treatment descendants separately. On the 64 DAGs consistent with a fixed four-node order, all 3,072 endpoint/adjustment combinations agree with the checker, and all 768 explicit-candidate searches agree on every valid and inclusion-minimal set. Other tests execute the 1,024-subset boundary, reject 11 candidates before any separation call, retain failed descendant and separation conditions together, distinguish minimum cardinality from inclusion minimality, and cover strict CLI errors/order/RNG isolation.

A supplied-SCM integration check converts the existing confounding fixture with all exogenous nodes preserved, finds Z as its valid candidate adjustment, then independently computes sum_z P(Y=1|X=x,Z=z)P(Z=z) from the observational joint. Every relevant X/Z cell has positive support; the result is .60/.50 for x=0/1 and matches direct intervention enumeration. This is a synthetic positive-support check, not a claim that the checker verifies positivity for arbitrary graphs or datasets.

The fifth SVG fixes one collider graph and all eight candidate subsets, plus summaries of confounding, mediator rejection and an unavailable common cause. Tests verify all eight pass/fail/minimal labels, five arrows/node coordinates, full unchanged graph/subset metadata and byte regeneration. Total suite: 138 tests. No fixture is selected through graph search; no empirical effect estimate is made. A missing backdoor set is not reported as a proof of non-identifiability.

The sufficient criterion, descendant exclusion and adjustment functional follow [Pearl's author-hosted discussion](https://bayes.cs.ucla.edu/BOOK-09/ch11-3-1.pdf) and [Poole/Mackworth's author-hosted AI lecture](https://www.cs.ubc.ca/~poole/aibook/3e/slides/ch11/lect_11_3_h.pdf). This is an independent implementation, not a complete general adjustment criterion, frontdoor search or do-calculus engine.

## Source and limits

The standard SCM intervention and counterfactual semantics are independently implemented from Pearl's [author-hosted R-485 report](https://ftp.cs.ucla.edu/pub/stat_ser/r485.pdf), section 2 and equation 3. No source text or implementation is incorporated. No general automatic effect identification beyond the supplied-DAG sufficient backdoor criterion, graph estimation, model selection, medical recommendation, policy optimization or fairness certificate follows from these computations.
