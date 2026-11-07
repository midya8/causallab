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

For the multi-world API, let S be scenario count (1..4) and D the total number of requested coordinates across scenarios (at most 32). It performs at most W factual and S*M action evaluations. Joint-key construction/output costs `O(S*M*N + K*D)` as a conservative bound since each query length is at most N; model evaluations cost `O((W+S*M)*(E+N+I))`. Grouped log lists hold at most W entries, not S*W entries. Non-trace working/output memory is `O(W+K*D+S*N+E+I)` excluding input tables. Traces add `O(M*(E+(S+1)*N))`. Its preflight uses `W*N*(S+1) <= 32768`, even if evidence would select only one world. Outputs count every complete scenario evaluation, including repeated or empty actions: no caching or speed claim.

Outputs expose actual factual/action evaluation counts. In ordinary mode, the evaluated model is counted in the action column even when no intervention was supplied. These counts measure complete model evaluations, not table operations, probability normalizations, machine instructions or wall-clock performance. The demo's 100-world example is exhaustive, not a sample of 100 people.

## Independent checks

The confounded fixture is checked against a separate `Fraction` loop over all integer noise assignments, using direct context/action/outcome conditions rather than the production evaluator. A second test covers all 4 binary X mechanisms and 16 binary Y mechanisms over two independent binary noise variables, comparing intervention and factual-evidence counterfactual answers to rational unit-level enumeration. This is 64 mechanism pairs, not a proof for arbitrary models.

Other tests cover nonbinary mixed-radix order, shared versus independent noise semantics, replacement of multiple equations, joint marginalization/query order, observational-equivalence counterexamples, the abduction/action/prediction consistency identities, log-space rare evidence, impossible events and trace posterior reconstruction. The world-cap boundary is executed at 65,536 slots. Oversized products/tables/query outputs/traces fail before any equation evaluation. CLI tests reject malformed/duplicate-key/oversized input and repeated assignment names without partial output.

The figure generator requires the complete current deterministic demo and rejects changed probabilities, mechanisms, traces or checks. Its eight bars share the [0,1] domain. Tests verify every coordinate/value and the full numeric/model/trace metadata, in addition to byte-identical regeneration. This is a reproducibility contract for fixed development counterexamples, not an empirical benchmark or source of causal validity.

## Cross-world coupling counterexample

The two new four-world fixtures share independent U,V with P(0)=P(1)=.5 and X=V. Stable Y=U gives `(Y0,Y1)=(U,U)`; flip Y=X xor U gives `(Y0,Y1)=(U,1-U)`. V randomizes X independently of U, so every observed `(X,Y)` pair has probability .25 in both. Under each fixed action, both mechanisms return Y=1 with probability .5. Therefore even the complete observed joint and all single-action marginals agree. Stable has response masses (.5,0,0,.5), while flip has (0,.5,.5,0), in order (00,01,10,11). A product of action marginals would give .25 to every response type, incorrectly imposing independent potential outcomes.

For factual X=0,Y=0, both fixtures require U=0,V=0, but stable predicts (0,0) and flip predicts (0,1). These are consequences of two explicit structural assumptions, not identified response-type estimates. No monotonicity is assumed. The fixtures are first hand-specified examples, not selected through mechanism or seed search. The complete demo includes the mechanisms, observations, intervention results, both joint queries and traces, all response masses and the naive product. The second SVG requires that complete unchanged demo; its eight bars and .25 reference lines use one fixed [0,1] scale.

Multi-world tests independently enumerate all 64 Boolean mechanism pairs using rational masses and direct table indexing under two actions, with empty and selected factual evidence. They also check each joint marginal against original single-world inference, repeated-world diagonal support, scenario/query permutation, nonbinary and multiple-node actions, no cross-scenario contamination, trace reconstruction/copy isolation, rare possible support and impossible evidence. Combined bounds are preflighted, and both 4,096 joint rows and the 32,768 trace node-slot boundary are actually executed. Figure tests check all coordinates, metadata/models/traces, changed-input rejection and byte-identical regeneration. Total suite: 64 tests.

## Source and limits

The standard SCM intervention and counterfactual semantics are independently implemented from Pearl's [author-hosted R-485 report](https://ftp.cs.ucla.edu/pub/stat_ser/r485.pdf), section 2 and equation 3. No source text or implementation is incorporated. No automatic effect identification, graph estimation, model selection, medical recommendation, policy optimization or fairness certificate follows from these computations.
