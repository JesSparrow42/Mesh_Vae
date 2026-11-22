"""Build the single upload file from independently tested, readable components."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    header = '''#!/usr/bin/env python3
"""Standalone Gefion comparison/figure workflow; upload this file only.

Bundled readable source is executed in isolated module namespaces. No files
are unpacked or installed, and existing training entry points are unchanged.
"""
import sys as _sys
import types as _types

def _component(name, source):
    module = _types.ModuleType(name)
    module.__file__ = __file__
    _sys.modules[name] = module
    exec(compile(source, __file__+":"+name, "exec"), module.__dict__)
    return module

'''
    blocks = []
    for variable, filename in (("COMPARISON", "clean_gefion_comparison.py"),
                               ("LONG", "clean_gefion_long.py"),
                               ("VOLUME", "clean_gefion_volume_aux.py"),
                               ("FIGURES", "clean_gefion_figures.py"),
                               ("LOSSES", "plot_gefion_loss_comparison.py")):
        source = (HERE/filename).read_text(encoding="utf-8")
        # Keep the single download readable when pasted into a terminal editor.
        if "'''" not in source:
            literal = "r'''"+source+"'''"
        elif '\"\"\"' not in source:
            literal = 'r\"\"\"'+source+'\"\"\"'
        else:
            literal = repr(source)
        blocks.append(f"{variable} = _component('_gefion_{variable.lower()}', {literal})\n")
    entry = (HERE/"clean_gefion_entrypoint.py").read_text(encoding="utf-8")
    out = HERE/"gefion_clean_results.py"
    out.write_text(header+"\n".join(blocks)+"\n"+entry, encoding="utf-8", newline="\n")
    compile(out.read_text(encoding="utf-8"), str(out), "exec")
    print(out)
    return out


if __name__ == "__main__":
    build()
