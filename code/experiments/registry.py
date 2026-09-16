"""Pre-registered single-factor model definitions; no per-direction switches."""
from copy import deepcopy
from data.cache.store import digest

BASE = dict(branches="both", scorer="attention", soft_threshold=True, inter_cluster=True,
            symmetric_selection=False, shared_theta=False, anchors="weighted",
            similarity="structure", query_residual=True, memory_warmup=True, consistency=True)
CHANGES = {
 "A1": ("scorer", "mlp"), "A2": ("soft_threshold", False),
 "A3": ("inter_cluster", False), "A4": ("symmetric_selection", True),
 "A5": ("shared_theta", True), "M1": ("anchors", "uniform"),
 "M2": ("anchors", "degree_quantile"), "M3": ("similarity", "has"),
 "M4": ("query_residual", False), "M5": ("memory_warmup", False),
 "M6": ("consistency", False)}
CORE = tuple(f"B{i}" for i in range(7))
ABLATIONS = tuple(CHANGES)
FINAL = dict(zip(CORE, ["encoder_best"] + ["da_best"]*4 + ["source_pl_best", "full_best"]))
FINAL.update({v: "full_best" for v in ABLATIONS})

def variant(name):
    if name not in FINAL: raise ValueError("Unregistered variant")
    options = deepcopy(BASE)
    if name in CHANGES: options[CHANGES[name][0]] = CHANGES[name][1]
    if name in ("B0", "B1", "B2", "B3"):
        options["branches"] = {"B0":"none", "B1":"none", "B2":"attribute", "B3":"structure"}[name]
    return dict(id=name, parent="B6" if name in CHANGES else None,
                changed_component=CHANGES[name][0] if name in CHANGES else None,
                options=options, final=FINAL[name],
                stages=1 if name=="B0" else 2 if name in CORE[1:5] else 4 if name=="B5" else 5)

def registry_hash(): return digest({v:variant(v) for v in FINAL})

def verify_single_factors():
    for name, (key, value) in CHANGES.items():
        actual=variant(name)["options"]
        if [k for k in BASE if BASE[k]!=actual[k]] != [key]:
            raise ValueError("Ablations must change exactly one factor")
    return registry_hash()
