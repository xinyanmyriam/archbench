import json, random
random.seed(123)

projects = [
    ("kubernetes", "../kubernetes", "Kubernetes", "container orchestration"),
    ("tensorflow", "../tensorflow", "TensorFlow", "machine learning framework"),
    ("codeaster", "../codeaster", "Code_Aster", "finite element analysis"),
    ("linux", "../linux", "Linux Kernel", "operating system kernel"),
]

print("=== CI Easy Samples ===")
for name, path, system, domain in projects:
    with open(path + "/component_identification.jsonl") as f:
        data = [json.loads(l) for l in f]
    s = random.choice(data)
    print(f"\n[{name}]")
    print(f"  file: {s['file']}")
    print(f"  candidates: {s['candidates']}")
    print(f"  label: {s['label']}")

print("\n\n=== CI Hard Samples ===")
for name, path, system, domain in projects:
    with open(path + "/component_identification_hard.jsonl") as f:
        data = [json.loads(l) for l in f]
    s = random.choice(data)
    print(f"\n[{name}]")
    print(f"  file: {s['file']}")
    print(f"  candidates: {s['candidates']}")
    print(f"  label: {s['label']}")

print("\n\n=== DP Samples ===")
for name, path, system, domain in projects:
    with open(path + "/dependency_prediction.jsonl") as f:
        data = [json.loads(l) for l in f]
    pos = [d for d in data if d["label"] == 1]
    neg = [d for d in data if d["label"] == 0]
    sp = random.choice(pos)
    sn = random.choice(neg)
    print(f"\n[{name}] POS: {sp['module_A']} -> {sp['module_B']}")
    print(f"[{name}] NEG: {sn['module_A']} -> {sn['module_B']}")

print("\n\n=== AR Samples ===")
for name, path, system, domain in projects:
    with open(path + "/architecture_reconstruction.jsonl") as f:
        data = [json.loads(l) for l in f]
    s = random.choice(data)
    print(f"\n[{name}] source: {s['source_module']}  target: {s['target_module']}")
