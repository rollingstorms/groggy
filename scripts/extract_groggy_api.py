import inspect
import groggy as gr
import types

def is_public(name):
    return not name.startswith('_')

def print_class_methods(cls, seen):
    if cls in seen:
        return
    seen.add(cls)
    print(f"\n## {cls.__name__}")
    for name, member in inspect.getmembers(cls):
        if is_public(name):
            if inspect.isfunction(member) or inspect.ismethod(member):
                try:
                    sig = str(inspect.signature(member))
                except Exception:
                    sig = "(?)"
                print(f"- {name}{sig}")
            elif isinstance(member, property):
                print(f"- [property] {name}")

def walk_module(mod, seen_classes, seen_mods):
    if mod in seen_mods:
        return
    seen_mods.add(mod)
    for name in dir(mod):
        if is_public(name):
            obj = getattr(mod, name)
            if inspect.isclass(obj):
                print_class_methods(obj, seen_classes)
            elif isinstance(obj, types.ModuleType) and obj.__name__.startswith('groggy'):
                walk_module(obj, seen_classes, seen_mods)

print("# Groggy API Method Extraction\n")
seen_classes = set()
seen_mods = set()
walk_module(gr, seen_classes, seen_mods)
