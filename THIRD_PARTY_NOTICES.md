# Third-party notices

The Inkling backport in `patches/` combines work by the oMLX contributors,
including the [`#5557` fusion change](https://github.com/jundot/omlx/pull/5557)
and the [`#5563` threadgroup fallback](https://github.com/jundot/omlx/pull/5563).
It was adapted to the `v0.8.3` source by omitting an unrelated import from
the fallback patch's context. The oMLX source is licensed under Apache-2.0.

Setup downloads oMLX from its [upstream release](https://github.com/jundot/omlx/releases/tag/v0.8.3)
and model files from the [MLX Community model card](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp).
The model is derived from [Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
See upstream for the licenses covering those downloads and their dependencies.
