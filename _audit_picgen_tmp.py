import sys, types, os, traceback

os.environ["NEWGEN_SKIP_MODEL_LOAD"] = "1"

# ---- torch mock ----
torch = types.ModuleType("torch")
torch.__version__ = "2.8.0.dev20250526+cu128"
torch.version = types.SimpleNamespace(cuda="12.8")
torch.cuda = types.SimpleNamespace(
    is_available=lambda: False, device_count=lambda: 0,
    get_device_properties=lambda i: (_ for _ in ()).throw(RuntimeError("no gpu")),
    empty_cache=lambda: None, synchronize=lambda: None, set_device=lambda d: None,
    current_device=lambda: 0,
)
class _T:
    float8_e4m3fn = "fp8"
    bfloat16 = "bf16"
    float16 = "fp16"
    device = type("device", (), {"__init__": lambda self, s: setattr(self, "type", s)})
    class backends:
        cuda = types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))
        cudnn = types.SimpleNamespace(benchmark=False)
    class nn:
        class Module:
            def __init__(self, *a, **k): pass
torch._dynamo = types.SimpleNamespace(config=types.SimpleNamespace(suppress_errors=True))
_nn = types.ModuleType("torch.nn")
class _NNModule:
    def __init__(self, *a, **k): pass
_nn.Module = _NNModule
torch.nn = _nn
sys.modules["torch"] = torch
sys.modules["torch.nn"] = _nn

# ---- numpy / cv2 / PIL stubs (numpy may be real) ----
try:
    import numpy  # noqa
except Exception:
    np = types.ModuleType("numpy"); sys.modules["numpy"] = np

# ---- gradio mock (records attribute access) ----
gr = types.ModuleType("gradio")
class _C:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self
    def __getattr__(self, n): return _C()
class _Block(_C):
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def click(self, *a, **k): pass
    def change(self, *a, **k): pass
    def input(self, *a, **k): pass
    def select(self, *a, **k): pass
    def upload(self, *a, **k): pass
    def load(self, *a, **k): pass
for name in ["Blocks", "Tab", "Tabs", "Row", "Column", "Accordion", "Group"]:
    setattr(gr, name, type(name, (_Block,), {}))
for name in ["Button", "Textbox", "Checkbox", "Dropdown", "Radio", "Slider",
             "Gallery", "Image", "Video", "Audio", "File", "Markdown", "HTML",
             "Chatbot", "CheckboxGroup", "Number", "TabItem"]:
    setattr(gr, name, type(name, (_C,), {}))
gr.on = lambda *a, **k: None
gr.utils = types.ModuleType("gradio.utils")
gr.utils.get_space = lambda: None
sys.modules["gradio"] = gr
sys.modules["gradio.utils"] = gr.utils

# ---- diffusers / transformers / accelerate mocks ----
class _Stub:
    def __getattr__(self, n): return _Stub()
    def __call__(self, *a, **k): return _Stub()
diffusers = types.ModuleType("diffusers"); diffusers.__version__ = "0.37.1"
diffusers.QwenImageEditPlusPipeline = _Stub()
diffusers.__getattr__ = lambda n: _Stub()
sys.modules["diffusers"] = diffusers
transformers = types.ModuleType("transformers"); transformers.__version__ = "4.55.4"
transformers.__getattr__ = lambda n: _Stub()
sys.modules["transformers"] = transformers
accel = types.ModuleType("accelerate"); accel.__getattr__ = lambda n: _Stub()
sys.modules["accelerate"] = accel
hf = types.ModuleType("huggingface_hub")
hf.hf_hub_download = lambda *a, **k: ""
hf.snapshot_download = lambda *a, **k: ""
hf.__getattr__ = lambda n: _Stub()
sys.modules["huggingface_hub"] = hf
safet = types.ModuleType("safetensors"); safetorch = types.ModuleType("safetensors.torch")
safetorch.load_file = lambda *a, **k: {}
sys.modules["safetensors"] = safet
sys.modules["safetensors.torch"] = safetorch

# ---- run the app module ----
os.chdir("picgen")
sys.path.insert(0, os.getcwd())
g = {"__name__": "__not_main__", "__file__": os.path.abspath("app.py")}
try:
    exec(compile(open("app.py", encoding="utf-8").read(), "app.py", "exec"), g)
    print("APP MODULE RAN CLEAN (module-level, no model load)")
except SystemExit as e:
    print(f"APP MODULE SystemExit({e})")
except Exception as e:
    traceback.print_exc()
    print(f"APP MODULE FAILED: {type(e).__name__}: {e}")
