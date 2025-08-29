"""
Auto-generator for Groggy API Python test harness.
Parses pymethods_report.txt and generates a test script that attempts to instantiate every class and call every method with placeholder arguments.
"""
import re

# Read the pymethods_report.txt
with open('scripts/pymethods_report.txt', 'r') as f:
    lines = f.readlines()

# Parse classes and methods
tests = []
current_class = None
for line in lines:
    # Match class impl
    m = re.match(r'impl ([A-Za-z0-9_]+) {', line)
    if m:
        current_class = m.group(1)
        continue
    # Match method
    m = re.match(r'\s*fn ([a-zA-Z0-9_]+)\(([^)]*)\)', line)
    if m and current_class:
        method = m.group(1)
        args = m.group(2)
        tests.append((current_class, method, args))
    # Match pub fn (for free functions)
    m = re.match(r'\s*pub fn ([a-zA-Z0-9_]+)\(([^)]*)\)', line)
    if m:
        method = m.group(1)
        args = m.group(2)
        tests.append(('__free__', method, args))

# Generate test script
with open('scripts/groggy_api_full_test.py', 'w') as out:
    out.write('''"""
Groggy FFI Python API Full Coverage Test Suite (auto-generated)
This script attempts to instantiate every class and call every method (from Rust #[pymethods]) with placeholder arguments.
"""
import groggy as gr
import traceback

results = []
def log_result(class_name, method, success, error=None):
    results.append({
        'class': class_name,
        'method': method,
        'success': success,
        'error': error
    })
    status = '✅' if success else '❌'
    print(f"{status} {class_name}.{method} {'' if success else error}")

def try_call(obj, method, *args, **kwargs):
    try:
        getattr(obj, method)(*args, **kwargs)
        return True, None
    except Exception as e:
        return False, str(e)

def try_instantiate(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] Could not instantiate {cls.__name__}: {e}")
        return None
''')

    # Instantiate objects for each class
    out.write('\n# --- Instantiate objects for each class ---\n')
    out.write('instances = {}\n')
    out.write('try:\n    instances["Graph"] = gr.Graph()\nexcept Exception as e:\n    print("[WARN] Could not instantiate Graph:", e)\n')
    # Add more instantiations as needed

    # Generate test calls
    out.write('\n# --- Test all methods ---\n')
    for cls, method, args in tests:
        if cls == '__free__':
            out.write(f'# Free function: {method}\n')
            out.write(f'try:\n    success, error = try_call(gr, "{method}")\n    log_result("__free__", "{method}", success, error)\nexcept Exception as e:\n    print("[WARN] Could not call free function {method}:", e)\n')
        else:
            out.write(f'# {cls}.{method}\n')
            out.write(f'if "{cls}" in instances:\n')
            out.write(f'    try:\n        success, error = try_call(instances["{cls}"], "{method}")\n        log_result("{cls}", "{method}", success, error)\n    except Exception as e:\n        print("[WARN] Could not call {cls}.{method}:", e)\n')

    # Report summary
    out.write('\nprint("\\n--- Test Summary ---")\n')
    out.write('for r in results:\n    print(f"{r[\\'class\\']}.{r[\\'method\\']}: {{'PASS' if r['success'] else 'FAIL'}}")\n    if not r[\\'success\\']:\n        print(f"    Error: {{r['error']}}")\n')
