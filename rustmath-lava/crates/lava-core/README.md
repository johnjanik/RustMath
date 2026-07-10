# lava-core

Library powering the `lava` CLI. Provides Magma source code formatting via [topiary-core](https://crates.io/crates/topiary-core) and the [tree-sitter-magma](https://github.com/edgarcosta/tree-sitter-magma/) grammar.

## API

```rust
use lava_core::{format_str, FormatOptions};

let source = "function f() return 1; end function;";
let (formatted, query_source) = format_str(source, &FormatOptions::default())?;
```

## Query resolution

The formatter looks for `magma.scm` topiary queries in this order:

1. `--query` override (if provided)
2. `./.lava/magma.scm` (walking up to filesystem root)
3. `$XDG_CONFIG_HOME/lava/magma.scm`
4. Embedded fallback (the canonical query bundled with this crate)

## Testing

```bash
cargo test -p lava-core
```

## License

MIT
