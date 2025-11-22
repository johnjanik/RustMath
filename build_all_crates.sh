#!/bin/bash

# Build all crates individually and log output
# This script builds each crate in the workspace and captures errors/warnings

LOG_DIR="build_logs"
mkdir -p "$LOG_DIR"

# List of all crates from Cargo.toml (removing duplicates)
CRATES=(
    "rustmath-core"
    "rustmath-features"
    "rustmath-typesetting"
    "rustmath-integers"
    "rustmath-rationals"
    "rustmath-reals"
    "rustmath-complex"
    "rustmath-polynomials"
    "rustmath-powerseries"
    "rustmath-finitefields"
    "rustmath-padics"
    "rustmath-algebraic"
    "rustmath-matrix"
    "rustmath-calculus"
    "rustmath-numbertheory"
    "rustmath-combinatorics"
    "rustmath-constants"
    "rustmath-crystals"
    "rustmath-geometry"
    "rustmath-graphs"
    "rustmath-symbolic"
    "rustmath-symmetricfunctions"
    "rustmath-functions"
    "rustmath-special-functions"
    "rustmath-crypto"
    "rustmath-groups"
    "rustmath-homology"
    "rustmath-category"
    "rustmath-stats"
    "rustmath-numerical"
    "rustmath-logic"
    "rustmath-dynamics"
    "rustmath-coding"
    "rustmath-databases"
    "rustmath-ellipticcurves"
    "rustmath-quadraticforms"
    "rustmath-numberfields"
    "rustmath-manifolds"
    "rustmath-modular"
    "rustmath-monoids"
    "rustmath-modules"
    "rustmath-misc"
    "rustmath-algebras"
    "rustmath-quantumgroups"
    "rustmath-liealgebras"
    "rustmath-lieconformal"
    "rustmath-quivers"
    "rustmath-colors"
    "rustmath-plot-core"
    "rustmath-plot"
    "rustmath-plot3d"
    "rustmath-rings"
    "rustmath-topology"
    "rustmath-sets"
    "rustmath-interfaces"
    "rustmath-affineschemes"
    "rustmath-schemes"
    "rustmath-trees"
    "rustmath-automata"
)

# Summary file
SUMMARY_FILE="$LOG_DIR/build_summary.txt"
echo "Build Summary - $(date)" > "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

SUCCESS=0
FAILED=0
WARNINGS=0

for crate in "${CRATES[@]}"; do
    echo "Building $crate..."
    LOG_FILE="$LOG_DIR/${crate}.log"

    # Build the crate and capture output
    cargo build -p "$crate" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    # Check for errors
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ FAILED: $crate" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))

        # Extract error messages
        echo "  Errors:" >> "$SUMMARY_FILE"
        grep -A 3 "^error" "$LOG_FILE" | sed 's/^/    /' >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    else
        # Check for warnings
        if grep -q "^warning" "$LOG_FILE"; then
            echo "⚠️  SUCCESS with warnings: $crate" | tee -a "$SUMMARY_FILE"
            WARNINGS=$((WARNINGS + 1))

            # Count warnings
            WARNING_COUNT=$(grep -c "^warning" "$LOG_FILE")
            echo "  Warnings: $WARNING_COUNT" >> "$SUMMARY_FILE"
            grep "^warning" "$LOG_FILE" | head -5 | sed 's/^/    /' >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
        else
            echo "✅ SUCCESS: $crate" | tee -a "$SUMMARY_FILE"
            SUCCESS=$((SUCCESS + 1))
        fi
    fi
done

echo "" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "Summary:" >> "$SUMMARY_FILE"
echo "  Total crates: ${#CRATES[@]}" >> "$SUMMARY_FILE"
echo "  Successful: $SUCCESS" >> "$SUMMARY_FILE"
echo "  With warnings: $WARNINGS" >> "$SUMMARY_FILE"
echo "  Failed: $FAILED" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"
