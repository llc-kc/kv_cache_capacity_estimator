"""Profile current per-stage cost with evalscope-style 64k content."""
import json
import sys
import time

sys.path.insert(0, "/opt/sglang/lib/python3.12/site-packages")

import numpy as np
from kv_capacity_estimator import (
    ReplaySimulator,
    SimulationConfig,
    chained_page_hashes,
    simulation_config_from_dict,
)
from transformers import AutoTokenizer

MODEL = "/data1/models/GLM-5.2-FP8"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
import re

pat = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
prohibited = set(tok.all_special_ids)
for s, i in tok.get_vocab().items():
    if pat.match(s):
        prohibited.add(i)
allowed = np.array(sorted(set(range(len(tok))) - prohibited))


def fresh_ids(n=64000, seed=0):
    sampled = allowed[np.random.RandomState(seed).randint(0, len(allowed), size=n)]
    text = tok.decode(sampled)
    return tok.encode(text, add_special_tokens=False)


# Build real config like server
cfg_dict = {
    "kv_bytes_per_token": 61505,
    "target_hit_rate_ratio": 0.8,
    "warm_up_ratio": 0.0,
}
sim_cfg = simulation_config_from_dict(cfg_dict)
sim = ReplaySimulator(sim_cfg)

# warm history
for k in range(200):
    ids = fresh_ids(seed=k)
    keys = chained_page_hashes(ids, page_size=64)
    sim.process_page_keys(keys)

N = 300
t_page = t_analyze = 0.0
for k in range(200, 200 + N):
    ids = fresh_ids(seed=k)
    t0 = time.perf_counter()
    keys = chained_page_hashes(ids, page_size=64)
    t_page += time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.process_page_keys(keys)
    t_analyze += time.perf_counter() - t0

print(f"page_hash (in worker, parallelizable): {t_page/N*1000:.2f} ms")
print(f"analyzer  (serial main thread)       : {t_analyze/N*1000:.2f} ms")
an = sim._analyzer
print(f"   reuse-distance                   : {an.reuse_distance_seconds/N*1000:.2f} ms")
print(f"   fixed-capacity loop              : {an.fixed_capacity_seconds/N*1000:.2f} ms")
print(f"   capacity-requirement             : {an.capacity_requirement_seconds/N*1000:.2f} ms")

# Now measure tokenize cost itself (the dominant worker cost)
t_enc = 0.0
texts = []
for k in range(200, 200 + N):
    sampled = allowed[np.random.RandomState(k).randint(0, len(allowed), size=64000)]
    texts.append(tok.decode(sampled))
for text in texts:
    t0 = time.perf_counter()
    tok.encode(text, add_special_tokens=False)
    t_enc += time.perf_counter() - t0
print(f"\ntokenizer.encode (worker dominant)  : {t_enc/N*1000:.2f} ms ({64000/(t_enc/N)/1e6:.2f} M tok/s)")
print(f"worker total ~ encode + page_hash   : {(t_enc+t_page)/N*1000:.2f} ms")
