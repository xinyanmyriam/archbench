"""
只跑 qwen3-14b 的 probing，跳过已有结果
"""
import os
import subprocess

PROJECTS = [
    'codeaster', 'hadoop', 'kubernetes', 'linux',
    'nastran', 'openfoam', 'openmodelica', 'paraview',
    'pytorch', 'salome', 'spring', 'tensorflow'
]

TASKS = ['ci', 'ci_hard', 'ci_obfuscated', 'ci_content', 'dp']

MODEL = 'qwen3-14b'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probing', 'results')

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    total = len(PROJECTS) * len(TASKS)
    skipped = 0
    done = 0
    failed = 0

    for project in PROJECTS:
        for task in TASKS:
            filename = f"probing_{MODEL}_{project}_{task}.json"
            filepath = os.path.join(RESULTS_DIR, filename)

            if os.path.exists(filepath):
                print(f"[SKIP] {project}/{task}")
                skipped += 1
                continue

            print(f"\n[RUN] {project}/{task}")
            cmd = f"python probing/probe_layers.py --model {MODEL} --project {project} --task {task}"
            result = subprocess.run(cmd, shell=True)

            if result.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"[FAIL] {project}/{task}")

    print(f"\nDone. Completed: {done}, Skipped: {skipped}, Failed: {failed}")

if __name__ == '__main__':
    main()
