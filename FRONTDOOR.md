# A mediator can identify an effect

## Scope and assumptions

This implements the established sufficient frontdoor criterion for one treatment X, one distinct outcome Y and a disjoint mediator set Z in a supplied causal DAG. It is not general identification, a necessary criterion for every possible frontdoor functional, graph discovery or an empirical independence test. Relevant latent common causes must remain explicit nodes. Candidate availability is declared by the caller, not inferred from names.

All three conditions remain visible even on failure:

1. Z intercepts every directed X-to-Y path.
2. There is no open proper backdoor path from X to Z.
3. X blocks every proper backdoor path from Z to Y.

A proper backdoor path starts with an arrow into its source set and has exactly one node from that source set. This matters when Z has multiple nodes. The criteria and functional follow Pearl's corrected chapter3, Definition3.3.3/Theorem3.3.4, with the set/proper-path formulation made explicit in the CIfly operator documentation. This code is an independent bounded implementation, not CIfly's maximum-set algorithm.

## Graph certificates

Directed BFS stops at mediators. A surviving X/Y path refutes condition1 and is returned as a canonical shortest directed witness in normalized node order. The reachable set includes reached mediators but does not expand them.

For condition2, remove outgoing arrows from X and check d-separation of X and Z with no conditioning. For condition3, remove outgoing arrows from every member of Z and check d-separation of Z and Y given X. These are criterion test graphs, not intervention graphs: do(X) instead cuts incoming arrows. Both full ancestral-moral certificates remain available, including graph hashes, ancestors, retained moral edges, their directed/coparent origins and a moral path when connected. A moral path is not represented as an original-DAG active path.

Empty mediators have vacuous second/third conditions, represented by null separation certificates. The first condition then passes only when there is no directed X/Y path. The resulting functional reduces to the marginal P(Y), expressing a graph-implied absence of a causal effect, not absence of observed association.

`frontdoor_sets` checks every subset of at most10 explicitly available candidates, including empty adjustment. Every subset retains the three booleans and directed/moral failure witnesses. All passing sets, inclusion-minimal sets and minimum size are returned separately. A selected set's full certificates come from `frontdoor_criterion`. The single-set checker shares the original32-node/128-edge DAG bounds and is not limited to10 mediators. No passing set in these candidates is not a proof of non-identifiability.

## Exact functional of a supplied joint

`frontdoor_effect` first requires the graph criterion to pass. Its observed joint must contain exactly X, Z and Y. Each variable has2..4 states, there are2..8 variables and at most4,096 full Cartesian probability cells. Variable order is explicit and the final index changes fastest. Inputs are integers or `[numerator,positive denominator]`, never floats, booleans, decimal strings or approximate counts. Components are bounded by1,000,000; normalized denominator LCM must be at most1,000,000. Probabilities are nonnegative and sum exactly to1. The normalized report orders variables as X, sorted Z, Y, independently of the input ordering.

Every declared treatment/mediator cell must have strictly positive P(x,z). This deliberately uses the classical sufficient support condition, not a query-specific weakening. A missing cell is rejected rather than filling an undefined0/0 conditional with zero. Zero outcome cells are allowed when P(x,z) remains positive.

For every x and y the report retains:

```
P(y | do(x)) = sum_z P(z | x) sum_xprime P(y | xprime,z) P(xprime)
```

Every inner conditional, treatment weight, weighted inner term, inner sum, mediator weight and final contribution is an exact rational pair. Support masses, canonical observed joint, its hash and the complete graph certificate accompany the result. Intermediate fraction sizes and full evidence records still cost time and space; a cell cap is not a universal RAM or speed guarantee.

A positive observed joint does not validate the causal graph, its latent-variable assumptions or compatibility with a supplied SCM. The causal interpretation is conditional on those assumptions. This is a deterministic functional of supplied exact probabilities, not an estimator learned from samples, a confidence interval or a medical/policy recommendation. The original SCM inference engine retains its existing floating-point/log-space semantics; this separate functional uses fractions and does not change the original API's arithmetic contract.

## Fixed control and figure

The standard latent-confounding motif U->X, U->Y, X->M->Y is credited, not claimed as a new causal graph. Additional fixed edge controls show all three conditions separately, including multiple simultaneous failures. The final control adds X->V->M and V->Y, which also creates an X->V->Y bypass.

The original numerical fixture uses independent U~Bernoulli(1/3), E1/E2~Bernoulli(1/4), E3~Bernoulli(1/8), with X=U xor E1, M=X xor E2, Y=M xor U xor E3. Its supplied positive joint produces observational P(Y=1|X=0)=41/112 and P(Y=1|X=1)=37/80. The frontdoor functional gives 7/16 and 9/16, independently matched to intervention world sums. These are synthetic arithmetic examples, not estimated effects.

The sixth fixed SVG retains the complete unchanged graph/functional demos as metadata. All 12 condition cells, four graph arrows/nodes and four shared-scale exact bars are checked. Regeneration is byte-identical, and changing an undisplayed inner contribution or provenance is rejected. The original five SVGs are unchanged.

## Checks and reproduction

All 164 native test functions pass locally on Python 3.10, 3.11, 3.12 and 3.13. The 26 additions include:

- All64 fixed-order four-node DAGs, all12 ordered treatment/outcome pairs and all four remaining-node subsets:3,072 criterion checks in768 complete searches. An independent original-DAG simple-path oracle uses explicit local collider/descendant rules, no surgery, moralization or production calls.
- Another720 full mediator-subset cases on60 seeded five/six-node DAGs, plus full1,024-subset candidate-boundary checks.
- Exact rational observational and separate intervention world sums for81 prescribed Boolean SCM parameter settings; every functional contribution, joint-mediator and nonbinary cases, empty mediators, input permutations, support and schema guards. These examples are not a proof for arbitrary causal models.
- Full4,096-cell/1,024-positive-support-cell boundary, denominator LCM rejection, undefined support, CLI duplicate/oversize errors, input/output isolation and four new figure tests.

```
python -m unittest discover -s tests -v
python -m causallab frontdoor examples/frontdoor_graph.json --treatment X --outcome Y --mediators M
python -m causallab frontdoor-sets examples/frontdoor_graph.json --treatment X --outcome Y --available M
python -m causallab frontdoor-effect examples/frontdoor_graph.json --treatment X --outcome Y --mediators M --joint examples/frontdoor_joint.json
python -m causallab frontdoor-demo
python -m causallab frontdoor-effect-demo
bash tools/check_wheel.sh
```

A separate optional external check runs ciflypy 0.1.3 with the CIfly operator's published rule tables and frontdoor function, unchanged except loading the table strings locally. Setting required and allowed mediators to the same candidate set tests that exact set, rather than asking for an unconstrained maximum. All 3,792 fixed-set comparisons agree across the 64 four-node DAGs and another 60 seeded larger graphs. The Rust-backed reference is not a runtime dependency, no source is vendored, and finite agreement is not a proof. Reference API: https://cifly.dev/docs/ciflypy/ .

The wheel check builds and installs the package into a fresh environment, changes outside the checkout, verifies every installed module's origin and reproduces seven whole demo reports through both module and console entrypoints, plus both exact functional CLI queries. No source directory is injected into imports.

## Inspected sources

- Pearl, author-hosted corrected Causality chapter3, section3.3.2, Definition3.3.3/Theorem3.3.4/equation3.29: https://web.cs.ucla.edu/~kaoru/CAUSALITY-REPRINTED-W-CORRECTIONS-2013/causality-ch3-june2013.pdf
- CIfly operator documentation, theorem1/proper paths/explicit candidate constraints and maximum-set distinction: https://cifly.dev/applications/frontdoor/
- University lecture, frontdoor conditions and two-stage derivation: https://www.ida.liu.se/~jospe50/PhDCausality/L2.pdf
- YLearn operator documentation, positivity and distinction from general identification: https://ylearn.readthedocs.io/en/latest/sub/causal_model/causal_model.html
- pgmpy operator API, standard latent-confounding example and validate/identify interface: https://pgmpy.org/api/generated/causal_inference/pgmpy.identification.frontdoor.Frontdoor.html
- Malina's independent derivation and incoming/outgoing-cut notation: https://stephenmalina.com/post/2020-03-09-front-door-do-calc-derivation/
- Gupta/Lipton/Childers, UAI2021 introduction on distinct identifying functionals versus statistical efficiency: https://proceedings.mlr.press/v161/gupta21b/gupta21b.pdf . No statistical efficiency result is claimed for this exact arithmetic inspector.

No source implementation, source text, empirical data or published figure is copied. No general do-calculus engine, causal discovery, statistical calibration, scalability, new theorem or new causal criterion is claimed.
