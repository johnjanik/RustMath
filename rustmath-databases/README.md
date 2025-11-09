# RustMath Databases

Mathematical database interfaces for RustMath, providing access to external mathematical databases and resources.

## Features

### OEIS (Online Encyclopedia of Integer Sequences)

Interface to the [OEIS](https://oeis.org/) database through its JSON API.

#### Basic Usage

```rust
use rustmath_databases::oeis::OEISClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OEISClient::new();

    // Look up a sequence by A-number
    if let Some(seq) = client.lookup("A000045")? {
        println!("Sequence: {}", seq.name);
        println!("First 10 terms: {:?}", &seq.data[..10]);

        // Access metadata
        for formula in &seq.formula {
            println!("Formula: {}", formula);
        }
    }

    Ok(())
}
```

#### Search by Terms

```rust
use rustmath_databases::oeis::OEISClient;

let client = OEISClient::new();

// Find sequences containing these terms
let results = client.search_by_terms(&[1, 1, 2, 3, 5, 8, 13, 21])?;

for seq in results.iter().take(5) {
    println!("{}: {}", seq.number, seq.name);
}
```

#### Text Search

```rust
use rustmath_databases::oeis::OEISClient;

let client = OEISClient::new();

// Search by name or description
let results = client.search("catalan numbers")?;

for seq in results {
    println!("{}: {}", seq.number, seq.name);
}
```

#### OEISSequence Structure

Each sequence contains:
- `number`: A-number (e.g., "A000045")
- `name`: Sequence name/description
- `data`: First several terms of the sequence
- `keyword`: Keywords describing the sequence
- `offset`: Index of first term
- `comment`: Comments about the sequence
- `reference`: Academic references
- `link`: Links to related resources
- `formula`: Mathematical formulas
- `example`: Example calculations
- `author`: Author information

## Testing

The crate includes both unit tests and integration tests. Some tests require network access to OEIS and are marked with `#[ignore]`.

```bash
# Run all tests except network tests
cargo test -p rustmath-databases

# Run all tests including network tests
cargo test -p rustmath-databases -- --include-ignored
```

## Dependencies

- `reqwest` - HTTP client for API requests
- `serde` / `serde_json` - JSON deserialization
- `urlencoding` - URL encoding for search queries

## Future Enhancements

Planned database interfaces:
- **Cunningham Tables**: Factorizations of numbers
- **Cremona Database**: Elliptic curves over rational numbers
- **Async API**: Non-blocking async/await interface using `reqwest::Client`

## License

GPL-2.0-or-later

## References

- OEIS API Documentation: https://oeis.org/wiki/JSON_Format
- OEIS Main Site: https://oeis.org/
