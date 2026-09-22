# Current scope

The current project asks one question:

> In real-language modeling, with producer and reader trained together, can
> `B: N/2 x 2d` match `A: N x d`?

The next experiment must directly produce the real-language A qualification
curve and, if A passes, the matched A/B comparison. A and B use the same data,
token order, optimizer family, training budget, evaluation documents, and
reader family. Preserving native-Qwen computation is not a goal.

Until that comparison exists, do not add new synthetic tasks, probes, frozen
oracles, migration/homotopy work, layerwise explanations, kernels, adaptive
interfaces, or `(M,D)` sweeps. Existing branches remain frozen evidence.

If A cannot learn language, use remote context, and depend on its global
interface within the small qualification budget, stop as
`INVALID_REAL_LANGUAGE_SUBSTRATE` rather than adding another prerequisite.
