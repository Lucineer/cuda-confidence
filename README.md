# cuda-confidence

**Uncertainty propagation as a first-class primitive.**

> Every value an agent touches carries 0-1 certainty.
> This certainty propagates through computation like electricity through a circuit.

## The Idea

Most AI systems treat confidence as an afterthought. In the Lucineer fleet, **confidence is a first-class type** that flows through every computation.

## How It Works

Confidence fuses via **harmonic mean**: `fused = 1/(1/a + 1/b)`

- Two high confidences produce high fusion (0.9 and 0.9 = 0.818)
- One high, one low is weighted toward low (0.9 and 0.1 = 0.09)
- Two moderate produces slightly lower (0.5 and 0.5 = 0.25)

This is **not** arithmetic mean. The harmonic mean penalizes uncertainty. If one source is unsure, the fusion is unsure.

## Ecosystem Integration

Confidence appears in **every cognitive crate**:
- `cuda-deliberation` - proposals carry confidence, consensus requires threshold
- `cuda-fusion` - multi-source sensor fusion with confidence weighting
- `cuda-sensor-agent` - Bayesian fusion of sensor readings
- `cuda-trust` - trust is a slowly-changing confidence
- `cuda-learning` - lesson confidence determines when to apply it
- `cuda-goal` - goal motivation modulated by confidence
- `cuda-emotion` - emotional state affects confidence propagation
- `cuda-attention` - saliency scores carry confidence

## Biological Parallel

Dopamine IS confidence. When a prediction is confirmed, dopamine signals strengthen the confidence. When it fails, confidence drops. The harmonic mean fusion mirrors how multiple neural circuits converge.

## See Also

- [cuda-equipment](https://github.com/Lucineer/cuda-equipment) - Shared foundation type
- [cuda-confidence-cascade](https://github.com/Lucineer/cuda-confidence-cascade) - Cascaded confidence
- [cuda-fusion](https://github.com/Lucineer/cuda-fusion) - Multi-source fusion
- [cuda-deliberation](https://github.com/Lucineer/cuda-deliberation) - Decision making

## License

MIT OR Apache-2.0