import sys, pathlib, site, subprocess

path = '/root/newgen/app.py'
content = open(path).read()

# The broken block to find and replace
OLD = (
    'os.makedirs("/dev/shm/newgen", exist_ok=True)\n'
    'os.makedirs("/root/newgen/tmp/gradio", exist_ok=True)\n'
    'try:\n'
    '    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _aesgcm_check\n'
    '    del _aesgcm_check\n'
    'except ImportError:\n'
    '    import subprocess as _sp\n'
    '    _sp.run([sys.executable, "-m", "pip", "install", "--quiet", "cryptography==42.0.8"], check=True)'
)

NEW = (
    'os.makedirs("/dev/shm/newgen", exist_ok=True)\n'
    'os.makedirs("/root/newgen/tmp/gradio", exist_ok=True)\n'
    'try:\n'
    '    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _aesgcm_check\n'
    '    del _aesgcm_check\n'
    'except ImportError:\n'
    '    import subprocess as _sp\n'
    '    _sp.run([sys.executable, "-m", "pip", "install", "--quiet", "cryptography==42.0.8"], check=True)\n'
    '\n'
    '# One-shot dep heal: use metadata (no imports) to check, one pip call, patch oauth.py\n'
    'import importlib.metadata as _imeta\n'
    'def _pkg_ver(pkg):\n'
    '    try: return tuple(int(x) for x in _imeta.version(pkg).split(".")[:3])\n'
    '    except: return (0,)\n'
    '_to_install = []\n'
    'if _pkg_ver("huggingface_hub") >= (1, 0):\n'
    '    _to_install += ["huggingface-hub>=0.34.0,<1.0"]\n'
    'if _pkg_ver("gradio")[:2] != (4, 43):\n'
    '    _to_install += ["gradio==4.43.0", "gradio-client==1.3.0"]\n'
    'if _pkg_ver("pydantic") >= (2, 10):\n'
    '    _to_install += ["pydantic>=2.0,<2.10"]\n'
    'if _to_install:\n'
    '    print(f"[startup] Fixing: {_to_install}")\n'
    '    import subprocess as _spx\n'
    '    _spx.run([sys.executable, "-m", "pip", "install", "--quiet",\n'
    '              "--force-reinstall", "--no-cache-dir"] + _to_install, check=False)\n'
    '    print("[startup] Done.")\n'
    'try:\n'
    '    import pathlib as _pl, site as _st\n'
    '    _dirs = _st.getsitepackages()\n'
    '    try: _dirs += [_st.getusersitepackages()]\n'
    '    except: pass\n'
    '    _op = next((p for d in _dirs for p in [_pl.Path(d, "gradio", "oauth.py")] if p.exists()), None)\n'
    '    if _op:\n'
    '        _ot = _op.read_text()\n'
    '        _OOLD = "from huggingface_hub import HfFolder, whoami"\n'
    '        _ONEW = (\n'
    '            "try:\\n"\n'
    '            "    from huggingface_hub import HfFolder, whoami\\n"\n'
    '            "except ImportError:\\n"\n'
    '            "    from huggingface_hub import whoami\\n"\n'
    '            "    try:\\n"\n'
    '            "        from huggingface_hub import get_token as _hf_gt\\n"\n'
    '            "    except ImportError:\\n"\n'
    '            "        _hf_gt = lambda: None\\n"\n'
    '            "    class HfFolder:\\n"\n'
    '            "        @staticmethod\\n"\n'
    '            "        def get_token(): return _hf_gt()"\n'
    '        )\n'
    '        if _OOLD in _ot:\n'
    '            _op.write_text(_ot.replace(_OOLD, _ONEW, 1))\n'
    '            print("[startup] Patched gradio/oauth.py.")\n'
    'except Exception as _oe:\n'
    '    print(f"[startup] oauth patch skipped: {_oe}")'
)

# Also remove any previous broken self-heal block if present
BROKEN_BLOCK = (
    '\n# ── Dependency self-heal (runs before any gradio/pydantic import) ──────────────'
)
if BROKEN_BLOCK in content:
    # Find start and end of the broken block
    start = content.find(BROKEN_BLOCK)
    end_marker = '# ──────────────────────────────────────────────────────────────────────────────'
    end = content.find(end_marker, start)
    if end != -1:
        end = end + len(end_marker)
        content = content[:start] + content[end:]
        print("Removed old broken self-heal block.")

# Also remove the newer broken variant
BROKEN_BLOCK2 = '\n# ── One-shot dependency heal (no restart, no loop)'
if BROKEN_BLOCK2 in content:
    start = content.find(BROKEN_BLOCK2)
    end_marker2 = '# ─────────────────────────────────────────────────────────────────────────────'
    end = content.find(end_marker2, start)
    if end != -1:
        end = end + len(end_marker2)
        content = content[:start] + content[end:]
        print("Removed second broken self-heal block.")

# Remove yet another variant
BROKEN_BLOCK3 = '\n# One-shot dep heal:'
if BROKEN_BLOCK3 in content:
    start = content.find(BROKEN_BLOCK3)
    # find the next top-level statement after the block
    import re
    m = re.search(r'\nos\.environ\.update', content[start:])
    if m:
        end = start + m.start()
        content = content[:start] + content[end:]
        print("Removed third broken self-heal block.")

if OLD in content:
    content = content.replace(OLD, NEW, 1)
    open(path, 'w').write(content)
    # Syntax check
    import ast
    try:
        ast.parse(content)
        print("SUCCESS: app.py patched and syntax is valid.")
        print("Run: python3 /root/newgen/app.py")
    except SyntaxError as e:
        print(f"SYNTAX ERROR after patch: {e}")
else:
    print("ERROR: Could not find the target block. Showing what's near line 48:")
    lines = content.split('\n')
    for i, l in enumerate(lines[45:60], 46):
        print(f"{i}: {l}")
