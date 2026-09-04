#!/usr/bin/env python3
"""
replace_terms.py - Replace Star Wars terms with fictional equivalents in knowledge base documents.

This script serves as documentation of the replacement process for the fictional universe
knowledge base used in the RAG bot project. It loads a mapping of original Star Wars terms
to their fictional replacements, scans all .txt files in the knowledge_base/ directory,
and applies any remaining replacements.
"""

import json
import os
from pathlib import Path
from typing import Dict


def load_terms_map(json_path: Path) -> Dict[str, str]:
    """Load the terms mapping from JSON file.
    
    Args:
        json_path: Path to terms_map.json
        
    Returns:
        Dictionary mapping original terms to fictional replacements.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def replace_terms_in_text(text: str, terms_map: Dict[str, str]) -> tuple[str, int]:
    """Replace original terms with fictional equivalents in text.
    
    Processes terms from longest to shortest to avoid partial replacements
    (e.g., replacing "Darth Vader" before "Vader").
    
    Args:
        text: The source text to process.
        terms_map: Dictionary of term replacements.
        
    Returns:
        Tuple of (modified text, number of replacements made).
    """
    replacements_count = 0
    
    # Sort by length descending so longer phrases match first
    sorted_terms = sorted(terms_map.keys(), key=len, reverse=True)
    
    for original_term in sorted_terms:
        replacement = terms_map[original_term]
        # Case-insensitive replacement with case preservation
        lower_text = text.lower()
        lower_term = original_term.lower()
        
        while lower_term in lower_text:
            idx = lower_text.index(lower_term)
            text = text[:idx] + replacement + text[idx + len(original_term):]
            lower_text = text.lower()
            replacements_count += 1
    
    return text, replacements_count


def process_knowledge_base(
    knowledge_base_dir: Path, terms_map: Dict[str, str], dry_run: bool = False
) -> Dict[str, int]:
    """Process all .txt files in the knowledge base directory.
    
    Args:
        knowledge_base_dir: Path to the knowledge_base/ directory.
        terms_map: Dictionary of term replacements.
        dry_run: If True, report changes without writing files.
        
    Returns:
        Dictionary mapping filenames to their replacement counts.
    """
    results = {}
    txt_files = sorted(knowledge_base_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"  No .txt files found in {knowledge_base_dir}")
        return results
    
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            original_text = f.read()
        
        new_text, count = replace_terms_in_text(original_text, terms_map)
        results[txt_file.name] = count
        
        if count > 0:
            status = "[DRY RUN] Would replace" if dry_run else "Replaced"
            print(f"  {status} {count} term(s) in {txt_file.name}")
            if not dry_run:
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(new_text)
        else:
            print(f"  No changes needed for {txt_file.name}")
    
    return results


def main() -> None:
    """Main entry point for the term replacement script."""
    # Resolve paths relative to the project root
    project_root = Path(__file__).parent.parent
    terms_map_path = project_root / "terms_map.json"
    knowledge_base_dir = project_root / "knowledge_base"
    
    print("=" * 60)
    print("Fictional Universe Term Replacement Script")
    print("=" * 60)
    
    # Validate paths
    if not terms_map_path.exists():
        print(f"Error: terms_map.json not found at {terms_map_path}")
        return
    
    if not knowledge_base_dir.exists():
        print(f"Error: knowledge_base/ directory not found at {knowledge_base_dir}")
        return
    
    # Load terms map
    terms_map = load_terms_map(terms_map_path)
    print(f"\nLoaded {len(terms_map)} term mappings from {terms_map_path.name}")
    
    # Process knowledge base
    print(f"\nProcessing files in {knowledge_base_dir.name}/:")
    results = process_knowledge_base(knowledge_base_dir, terms_map)
    
    # Summary
    total_replacements = sum(results.values())
    files_modified = sum(1 for count in results.values() if count > 0)
    
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Files processed: {len(results)}")
    print(f"  Files modified:  {files_modified}")
    print(f"  Total replacements: {total_replacements}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
