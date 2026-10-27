"""临时诊断脚本：统计各模型结果的失败模式（空响应 / 解析失败 / API错误）"""
import json
import glob
import os
import re
from collections import defaultdict

PAT = re.compile(
    r'^(?P<model>.+?)_(?P<proj>openfoam|nastran|salome|paraview|kubernetes|linux|hadoop|spring|codeaster|openmodelica|tensorflow|pytorch)_(?P<task>ci_hard|ci_obfuscated|ci|dp|ar)\.json$'
)

stats = defaultdict(lambda: {'total': 0, 'empty': 0, 'parse_err': 0, 'error_text': 0, 'ok': 0})
by_task = defaultdict(lambda: [0, 0, 0])  # (model, task) -> [total, empty, parse_err]

for path in glob.glob(os.path.join(os.path.dirname(__file__), 'results', '*.json')):
    m = PAT.match(os.path.basename(path))
    if not m:
        continue
    model, task = m.group('model'), m.group('task')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print('LOAD_ERR', path, e)
        continue
    results = data.get('results', [])
    s = stats[model]
    t = by_task[(model, task)]
    for r in results:
        s['total'] += 1
        t[0] += 1
        rr = r.get('raw_response') or ''
        if rr.startswith('ERROR:'):
            s['error_text'] += 1
        elif rr == '':
            s['empty'] += 1
            t[1] += 1
        elif task in ('ci', 'ci_hard'):
            if r.get('predicted_module') == 'PARSE_ERROR' or r.get('predicted_letter') is None:
                s['parse_err'] += 1
                t[2] += 1
            else:
                s['ok'] += 1
        elif task == 'dp':
            if r.get('predicted') not in (0, 1):
                s['parse_err'] += 1
                t[2] += 1
            else:
                s['ok'] += 1
        elif task == 'ar':
            pred = r.get('predicted_modules') or r.get('predicted') or r.get('predictions')
            if not pred:
                s['parse_err'] += 1
                t[2] += 1
            else:
                s['ok'] += 1

print(f"{'model':22s} {'total':>6s} {'ok':>6s} {'empty':>6s} {'parse_err':>9s} {'api_err':>8s} {'fail%':>7s}")
for model, s in sorted(stats.items(),
                       key=lambda x: -(x[1]['empty'] + x[1]['parse_err'] + x[1]['error_text']) / max(x[1]['total'], 1)):
    t, ok, emp, pe, et = s['total'], s['ok'], s['empty'], s['parse_err'], s['error_text']
    fail = t - ok
    print(f"{model:22s} {t:6d} {ok:6d} {emp:6d} {pe:9d} {et:8d} {fail / t * 100:6.1f}%")

print()
print('=== 受影响模型的分任务明细 (total / empty / parse_err) ===')
for (model, task), (tot, emp, pe) in sorted(by_task.items()):
    if model in ('gpt-5', 'kimi-k2.5', 'gpt-oss-120b', 'deepseek-v4-flash') and (emp or pe):
        print(f"{model:22s} {task:14s} {tot:5d} {emp:5d} {pe:5d}")
