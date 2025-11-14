#!/usr/bin/env python3
"""Create CSV export of partial features for project management."""

import csv
from pathlib import Path

def extract_partial_entries():
    """Extract all entries marked as Partial."""
    partial_entries = []
    
    for i in range(1, 15):
        filename = f"sagemath_to_rustmath_tracker_part_{i:02d}.csv"
        filepath = Path(filename)
        
        if not filepath.exists():
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Status', '').strip() == 'Partial':
                    partial_entries.append(row)
    
    return partial_entries

def main():
    entries = extract_partial_entries()
    
    # Write to CSV
    output_file = "PARTIAL_FEATURES_TODO.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Priority', 'Done', 'Module', 'Entity', 'Type', 'Full_Name', 'Source']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in sorted(entries, key=lambda x: (x['module'], x['entity_name'])):
            writer.writerow({
                'Priority': '',  # To be filled in manually
                'Done': '',  # Checkbox for tracking
                'Module': entry['module'],
                'Entity': entry['entity_name'],
                'Type': entry['type'],
                'Full_Name': entry['full_name'],
                'Source': entry['source']
            })
    
    print(f"✓ CSV export written to {output_file}")
    print(f"  Total entries: {len(entries)}")

if __name__ == '__main__':
    main()
