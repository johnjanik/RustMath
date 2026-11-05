#!/usr/bin/env python3
"""Create prioritized TODO list from partial entries."""

import csv
from collections import defaultdict
from pathlib import Path

# Priority classifications
PRIORITY_MAP = {
    # High priority - fundamental operations frequently used
    'HIGH': {
        'sage.arith': ['gcd', 'lcm', 'factorial', 'binomial', 'divisors', 'factor', 'euler_phi', 'carmichael'],
        'sage.rings.polynomial': ['factor', 'gcd', 'resultant', 'discriminant', 'roots'],
        'sage.rings.number_field': ['NumberField', 'number_field'],
        'sage.symbolic': ['simplify', 'expand', 'factor', 'collect', 'solve', 'integrate', 'diff'],
        'sage.calculus': ['limit', 'derivative', 'integral', 'taylor'],
        'sage.functions': ['sin', 'cos', 'exp', 'log', 'gamma', 'bessel'],
    },
    # Medium priority - useful but not critical
    'MEDIUM': {
        'sage.rings': ['ideal', 'quotient', 'extension'],
        'sage.categories': ['Category', 'Functor'],
        'sage.arith': ['bernoulli', 'harmonic', 'zeta'],
    },
    # Low priority - specialized or advanced features
    'LOW': {
        'sage.symbolic.units': ['*'],  # Unit conversion
        'sage.symbolic.assumptions': ['*'],  # Advanced assumptions
    }
}

def get_priority(entry):
    """Determine priority of an entry."""
    full_name = entry['full_name'].lower()
    entity_name = entry['entity_name'].lower()
    
    # Check high priority
    for module, keywords in PRIORITY_MAP['HIGH'].items():
        if module in full_name:
            if any(kw in entity_name or kw in full_name for kw in keywords):
                return 'HIGH'
    
    # Check medium priority
    for module, keywords in PRIORITY_MAP['MEDIUM'].items():
        if module in full_name:
            if any(kw in entity_name or kw in full_name for kw in keywords):
                return 'MEDIUM'
    
    # Check low priority
    for module, keywords in PRIORITY_MAP['LOW'].items():
        if module in full_name:
            if '*' in keywords or any(kw in entity_name or kw in full_name for kw in keywords):
                return 'LOW'
    
    # Default to medium
    return 'MEDIUM'

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
                    partial_entries.append({
                        'full_name': row['full_name'],
                        'module': row['module'],
                        'entity_name': row['entity_name'],
                        'type': row['type'],
                        'bases': row['bases'],
                        'source': row['source']
                    })
    
    return partial_entries

def main():
    entries = extract_partial_entries()
    
    # Organize by priority
    by_priority = defaultdict(list)
    for entry in entries:
        priority = get_priority(entry)
        by_priority[priority].append(entry)
    
    # Organize each priority by module
    organized = {}
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        by_module = defaultdict(list)
        for entry in by_priority[priority]:
            module = entry['module']
            by_module[module].append(entry)
        organized[priority] = by_module
    
    # Write prioritized TODO list
    output_file = "PRIORITIZED_TODO.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# RustMath Development TODO - Prioritized Partial Features\n\n")
        f.write(f"**Total Features:** {len(entries)}\n\n")
        f.write("This list organizes the 970 partial SageMath features by implementation priority.\n\n")
        
        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            count = len(by_priority[priority])
            f.write(f"## Priority: {priority} ({count} features)\n\n")
            
            if priority == 'HIGH':
                f.write("These are fundamental operations used frequently. Implementing these will have the highest impact.\n\n")
            elif priority == 'MEDIUM':
                f.write("Useful features that enhance functionality but are not critical.\n\n")
            else:
                f.write("Specialized or advanced features for specific use cases.\n\n")
            
            for module in sorted(organized[priority].keys()):
                entries_list = organized[priority][module]
                f.write(f"### {module} ({len(entries_list)} features)\n\n")
                
                for entry in sorted(entries_list, key=lambda x: x['entity_name']):
                    f.write(f"- [ ] **{entry['entity_name']}**")
                    if entry['type']:
                        f.write(f" `({entry['type']})`")
                    f.write(f"\n  - Full name: `{entry['full_name']}`\n")
                    if entry['source']:
                        f.write(f"  - [Source]({entry['source']})\n")
                    f.write("\n")
            
            f.write("\n---\n\n")
    
    print(f"✓ Prioritized TODO written to {output_file}")
    print(f"  - HIGH priority: {len(by_priority['HIGH'])} features")
    print(f"  - MEDIUM priority: {len(by_priority['MEDIUM'])} features")
    print(f"  - LOW priority: {len(by_priority['LOW'])} features")

if __name__ == '__main__':
    main()
