#!/bin/bash
cd /home/user/RustMath
for dir in rustmath-*/src; do
    crate=$(dirname "$dir")
    crate=$(basename "$crate")
    lines=$(find "$dir" -name "*.rs" -exec cat {} \; 2>/dev/null | wc -l)
    echo "$crate $lines"
done | sort -k2 -n
