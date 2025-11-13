# SageMath to RustMath Tracker - Split Files

## Overview

The original tracker file `sagemath_to_rustmath_tracker_20251110.csv` contained **13,853 lines** (1 header + 13,852 data entries) documenting all SageMath functions, classes, and modules that need to be converted to Rust.

Due to its large size, the file has been split into **14 manageable chunks** for easier processing.

## Split Files

The tracker has been divided into the following files:

| File | Lines (with header) | Data Entries |
|------|---------------------|--------------|
| `sagemath_to_rustmath_tracker_part_01.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_02.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_03.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_04.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_05.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_06.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_07.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_08.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_09.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_10.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_11.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_12.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_13.csv` | 1,001 | 1,000 |
| `sagemath_to_rustmath_tracker_part_14.csv` | 853 | 852 |
| **Total** | **13,866** | **13,852** |

## CSV Structure

Each file contains the same header:

```csv
Status,full_name,module,entity_name,type,bases,source
```

### Column Descriptions

- **Status**: Conversion status (empty = not started, or other status indicators)
- **full_name**: Full qualified name in SageMath (e.g., `sage.algebras.clifford_algebra.CliffordAlgebra`)
- **module**: Parent module path
- **entity_name**: Name of the function/class/module
- **type**: Entity type (`module`, `class`, `function`)
- **bases**: Base classes (for classes) or other inheritance information
- **source**: GitHub URL to the source code

## How to Use These Files

1. **Working with individual chunks**: Each file is small enough to open in spreadsheet software (Excel, Google Sheets, LibreOffice Calc)
2. **Processing programmatically**: Load individual files or iterate through all 14 files
3. **Tracking progress**: Update the `Status` column as conversions are completed
4. **Merging back**: If needed, the files can be merged back using `split_tracker.py` or similar tooling

## Regenerating the Split

To regenerate the split files from the original tracker:

```bash
python3 split_tracker.py
```

The script splits the CSV into chunks of 1,000 data lines each (plus the header).

## Content Coverage

The tracker covers all SageMath modules, including:

- Algebras (Clifford, cluster, commutative DGA, etc.)
- Number theory (integers, rationals, p-adics, finite fields)
- Geometry (schemes, Riemann surfaces, polytopes)
- Combinatorics (permutations, partitions, tableaux)
- Graph theory
- Matrices and linear algebra
- Symbolic computation
- And many more...

Each entry represents a conversion task from SageMath (Python) to RustMath (Rust).
