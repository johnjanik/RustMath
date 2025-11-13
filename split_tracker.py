#!/usr/bin/env python3
"""
Split the large SageMath to RustMath tracker CSV file into manageable chunks.
Each chunk will contain 1000 data lines plus the header.
"""

import csv
import os

INPUT_FILE = "sagemath_to_rustmath_tracker_20251110.csv"
OUTPUT_PREFIX = "sagemath_to_rustmath_tracker_part"
CHUNK_SIZE = 1000  # Lines per chunk (excluding header)

def split_csv(input_file, output_prefix, chunk_size):
    """Split a large CSV file into smaller chunks."""

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header

        chunk_num = 1
        line_count = 0
        current_chunk = []

        for row in reader:
            current_chunk.append(row)
            line_count += 1

            if line_count >= chunk_size:
                # Write current chunk
                output_file = f"{output_prefix}_{chunk_num:02d}.csv"
                write_chunk(output_file, header, current_chunk)
                print(f"Created {output_file} with {len(current_chunk)} lines")

                # Reset for next chunk
                chunk_num += 1
                line_count = 0
                current_chunk = []

        # Write remaining lines if any
        if current_chunk:
            output_file = f"{output_prefix}_{chunk_num:02d}.csv"
            write_chunk(output_file, header, current_chunk)
            print(f"Created {output_file} with {len(current_chunk)} lines")

    print(f"\nTotal chunks created: {chunk_num}")
    print(f"Original file had {sum(1 for _ in open(input_file, 'r', encoding='utf-8'))} lines (including header)")

def write_chunk(filename, header, rows):
    """Write a chunk to a CSV file."""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

if __name__ == "__main__":
    split_csv(INPUT_FILE, OUTPUT_PREFIX, CHUNK_SIZE)
