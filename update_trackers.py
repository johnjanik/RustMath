#!/usr/bin/env python3
"""
Update all tracker files to add Implementation Type column (STUB or FULL)
"""

import csv
import os
from pathlib import Path

def load_implementation_mapping():
    """Load the implementation type mapping into a dictionary."""
    mapping = {}
    with open('/home/user/RustMath/IMPLEMENTATION_TYPE_MAPPING.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Key by full_name for lookup
            mapping[row['full_name']] = row['implementation_type']
    return mapping

def update_tracker_file(tracker_path, mapping):
    """Update a single tracker file with Implementation Type column."""
    print(f"Processing {tracker_path}...")

    # Read the tracker file
    rows = []
    with open(tracker_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Check if Implementation Type already exists
        if 'Implementation Type' in fieldnames:
            print(f"  - Already has Implementation Type column, updating values...")
            new_fieldnames = fieldnames
        else:
            # Insert Implementation Type after Status
            new_fieldnames = []
            for field in fieldnames:
                new_fieldnames.append(field)
                if field == 'Status':
                    new_fieldnames.append('Implementation Type')

        for row in reader:
            # Look up implementation type
            full_name = row['full_name']
            status = row['Status']

            if status.strip().lower() in ['implemented', 'implementation']:
                # Look up in mapping
                impl_type = mapping.get(full_name, '')
                if not impl_type:
                    # Try with entity_name if full lookup fails
                    print(f"  - Warning: No mapping found for {full_name}")
                    impl_type = ''
            else:
                # Not implemented, leave blank
                impl_type = ''

            row['Implementation Type'] = impl_type
            rows.append(row)

    # Write updated tracker file
    with open(tracker_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Count updates
    stub_count = sum(1 for r in rows if r.get('Implementation Type') == 'STUB')
    full_count = sum(1 for r in rows if r.get('Implementation Type') == 'FULL')
    print(f"  - Updated: {full_count} FULL, {stub_count} STUB")

def main():
    """Main function to update all tracker files."""
    print("Loading implementation type mapping...")
    mapping = load_implementation_mapping()
    print(f"Loaded {len(mapping)} mappings")

    # Find all tracker files
    tracker_files = sorted(Path('/home/user/RustMath').glob('sagemath_to_rustmath_tracker_part_*.csv'))

    print(f"\nFound {len(tracker_files)} tracker files")

    total_full = 0
    total_stub = 0

    for tracker_file in tracker_files:
        update_tracker_file(str(tracker_file), mapping)

    print("\nAll tracker files updated successfully!")

if __name__ == '__main__':
    main()
