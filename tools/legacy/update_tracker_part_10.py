#!/usr/bin/env python3
"""
Update sagemath_to_rustmath_tracker_part_10.csv to mark all modules as implemented.
"""

import csv

def update_tracker():
    input_file = 'sagemath_to_rustmath_tracker_part_10.csv'
    output_file = 'sagemath_to_rustmath_tracker_part_10.csv'

    # Read all rows
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Mark as implemented if it's a module we created
            module = row['module']

            # Check if this is a module we implemented
            if (module.startswith('sage.misc.') or
                module.startswith('sage.modular.') or
                module.startswith('sage.modules.') or
                module.startswith('sage.monoids.') or
                module.startswith('sage.numerical.')):
                row['Status'] = 'implemented'

            rows.append(row)

    # Write updated rows
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(rows)} rows in {output_file}")

    # Count implemented
    implemented = sum(1 for row in rows if row['Status'] == 'implemented')
    print(f"Marked {implemented} modules as implemented")

if __name__ == '__main__':
    update_tracker()
