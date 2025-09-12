#!/usr/bin/env python3
"""
Sync modified CSS themes back to Rust source

Usage:
    python sync_to_rust.py [theme_name]

If theme_name is provided, only sync that theme.
Otherwise, sync all themes.
"""

import sys
from pathlib import Path
import shutil

def sync_themes_to_rust(specific_theme=None):
    """Sync CSS themes from template_prototype back to Rust source"""
    
    template_dir = Path("template_prototype")
    rust_themes_dir = Path("../../src/core/display/themes")
    
    if not template_dir.exists():
        print("❌ template_prototype directory not found!")
        return False
        
    if not rust_themes_dir.exists():
        print("❌ Rust themes directory not found!")
        return False
    
    themes = ["light.css", "dark.css", "minimal.css", "publication.css"]
    
    if specific_theme:
        if not specific_theme.endswith('.css'):
            specific_theme += '.css'
        if specific_theme not in themes:
            print(f"❌ Unknown theme: {specific_theme}")
            print(f"   Available: {', '.join(themes)}")
            return False
        themes = [specific_theme]
    
    print("🔄 Syncing CSS themes to Rust source...")
    
    synced = 0
    for theme in themes:
        src_file = template_dir / "themes" / theme
        dest_file = rust_themes_dir / theme
        
        if src_file.exists():
            # Read and compare content
            src_content = src_file.read_text()
            
            if dest_file.exists():
                dest_content = dest_file.read_text()
                
                if src_content != dest_content:
                    # Backup original
                    backup_file = dest_file.with_suffix(f'.css.backup')
                    shutil.copy2(dest_file, backup_file)
                    print(f"   📋 Backed up {theme} to {backup_file.name}")
                    
                    # Copy new content
                    shutil.copy2(src_file, dest_file)
                    print(f"   ✅ Updated {theme}")
                    synced += 1
                else:
                    print(f"   ✨ {theme} - no changes")
            else:
                shutil.copy2(src_file, dest_file)
                print(f"   ➕ Created {theme}")
                synced += 1
        else:
            print(f"   ⚠️ {theme} not found in template_prototype")
    
    if synced > 0:
        print(f"\n✅ Synced {synced} theme(s) to Rust source")
        print("   💡 Remember to:")
        print("      1. Test with Rust: cargo test")
        print("      2. Verify display: cargo run --example display_test")
        print("      3. Commit changes to both template and Rust files")
    else:
        print("\n✨ All themes are up to date")
    
    return True

def main():
    """Main function"""
    specific_theme = sys.argv[1] if len(sys.argv) > 1 else None
    
    if specific_theme and specific_theme in ['-h', '--help']:
        print(__doc__)
        return
    
    sync_themes_to_rust(specific_theme)

if __name__ == "__main__":
    main()
