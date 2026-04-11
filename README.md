# cuda-confidence

Confidence primitive — foundational type where every value carries uncertainty and propagates it (Rust)

Part of the Cocapn fleet — a Lucineer vessel component.

## What It Does

### Key Types

- `Conf` — core data structure
- `ConfidenceDist` — core data structure

## Quick Start

```bash
# Clone
git clone https://github.com/Lucineer/cuda-confidence.git
cd cuda-confidence

# Build
cargo build

# Run tests
cargo test
```

## Usage

```rust
use cuda_confidence::*;

// See src/lib.rs for full API
// 18 unit tests included
```

### Available Implementations

- `Conf` — see source for methods
- `Add for Conf` — see source for methods
- `Sub for Conf` — see source for methods
- `Mul for Conf` — see source for methods
- `Div for Conf` — see source for methods
- `PartialEq for Conf` — see source for methods

## Testing

```bash
cargo test
```

18 unit tests covering core functionality.

## Architecture

This crate is part of the **Cocapn Fleet** — a git-native multi-agent ecosystem.

- **Category**: other
- **Language**: Rust
- **Dependencies**: See `Cargo.toml`
- **Status**: Active development

## Related Crates


## Fleet Position

```
Casey (Captain)
├── JetsonClaw1 (Lucineer realm — hardware, low-level systems, fleet infrastructure)
├── Oracle1 (SuperInstance — lighthouse, architecture, consensus)
└── Babel (SuperInstance — multilingual scout)
```

## Contributing

This is a fleet vessel component. Fork it, improve it, push a bottle to `message-in-a-bottle/for-jetsonclaw1/`.

## License

MIT

---

*Built by JetsonClaw1 — part of the Cocapn fleet*
*See [cocapn-fleet-readme](https://github.com/Lucineer/cocapn-fleet-readme) for the full fleet roadmap*
