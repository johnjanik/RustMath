#!/usr/bin/env python3
"""Extract all Partial entries from tracker files to create TODO list."""

import csv
from collections import defaultdict
from pathlib import Path

def extract_partial_entries():
    """Extract all entries marked as Partial."""
    partial_entries = []
    
    # Process all 14 tracker files
    for i in range(1, 15):
        filename = f"sagemath_to_rustmath_tracker_part_{i:02d}.csv"
        filepath = Path(filename)
        
        if not filepath.exists():
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Status', '').strip() == 'Partial':
                    partial_entries.append({
                        'full_name': row['full_name'],
                        'module': row['module'],
                        'entity_name': row['entity_name'],
                        'type': row['type'],
                        'bases': row['bases'],
                        'source': row['source']
                    })
    
    return partial_entries

def organize_by_module(entries):
    """Organize entries by top-level SageMath module."""
    by_module = defaultdict(list)
    
    for entry in entries:
        # Extract top-level module (e.g., sage.rings, sage.symbolic, etc.)
        full_name = entry['full_name']
        if full_name.startswith('sage.'):
            parts = full_name.split('.')
            if len(parts) >= 2:
                top_module = f"{parts[0]}.{parts[1]}"
            else:
                top_module = parts[0]
        else:
            top_module = "other"
        
        by_module[top_module].append(entry)
    
    return by_module

def format_entry(entry):
    """Format a single entry for the TODO list."""
    name = entry['full_name']
    entity_type = entry['type']
    bases = entry['bases']
    source = entry['source']
    
    # Create a concise description
    desc = f"  - [ ] `{name}`"
    if entity_type:
        desc += f" ({entity_type})"
    if bases:
        desc += f" extends {bases}"
    if source:
        desc += f"\n        Source: {source}"
    
    return desc

def main():
    print("Extracting Partial entries from tracker files...")
    partial_entries = extract_partial_entries()
    
    print(f"Found {len(partial_entries)} Partial entries")
    
    # Organize by module
    by_module = organize_by_module(partial_entries)
    
    # Write to markdown file
    output_file = "PARTIAL_FEATURES_TODO.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# SageMath Features - Partial Implementation TODO List\n\n")
        f.write(f"**Total Partial Features:** {len(partial_entries)}\n\n")
        f.write("This list contains all SageMath features marked as 'Partial' in the tracker files.\n")
        f.write("These represent features where some related functionality exists in RustMath, but the implementation is incomplete.\n\n")
        f.write("---\n\n")
        
        # Sort modules alphabetically
        for module in sorted(by_module.keys()):
            entries = by_module[module]
            f.write(f"## {module} ({len(entries)} features)\n\n")
            
            for entry in sorted(entries, key=lambda x: x['full_name']):
                f.write(format_entry(entry) + "\n")
            
            f.write("\n")
        
        # Write summary at the end
        f.write("---\n\n")
        f.write("## Summary by Module\n\n")
        f.write("| Module | Partial Features |\n")
        f.write("|--------|------------------|\n")
        for module in sorted(by_module.keys()):
            f.write(f"| {module} | {len(by_module[module])} |\n")
        f.write(f"| **TOTAL** | **{len(partial_entries)}** |\n")
    
    print(f"✓ TODO list written to {output_file}")
    
    # Also create a simple text version
    text_file = "PARTIAL_FEATURES_TODO.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("SageMath Features - Partial Implementation TODO List\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Partial Features: {len(partial_entries)}\n\n")
        
        for module in sorted(by_module.keys()):
            entries = by_module[module]
            f.write(f"\n{module} ({len(entries)} features)\n")
            f.write("-" * 60 + "\n")
            
            for entry in sorted(entries, key=lambda x: x['full_name']):
                f.write(f"  [ ] {entry['full_name']}")
                if entry['type']:
                    f.write(f" ({entry['type']})")
                f.write("\n")
    
    print(f"✓ Text version written to {text_file}")

if __name__ == '__main__':
    main()
