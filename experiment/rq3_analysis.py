#!/usr/bin/env python3
"""
RQ3: Externalization - 统计 AR (Architecture Recovery) 结果

分析各模型在各项目上的架构知识外化表现
"""
import os
import json
import glob

# 配置
RESULTS_DIR = "results"
OUTPUT_DIR = "../paperB/rq3_externalization"

def analyze_ar_results():
    """统计 AR 结果"""
    results = []
    
    # 找到所有 ar.json 文件
    ar_files = glob.glob(os.path.join(RESULTS_DIR, "*_ar.json"))
    
    for filepath in ar_files:
        filename = os.path.basename(filepath)
        
        # 解析模型名和项目名
        # 格式: model_project_ar.json
        parts = filename.replace("_ar.json", "").split("_")
        
        if len(parts) < 2:
            continue
        
        project = parts[-1]
        model = "_".join(parts[:-1])
        
        # 读取结果
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results.append({
            'model': model,
            'project': project,
            'precision_at_5': data.get('precision_at_5', 0),
            'recall_at_5': data.get('recall_at_5', 0),
            'total': data.get('total', 0),
        })
    
    return results

def aggregate_by_model(results):
    """按模型汇总"""
    model_stats = {}
    
    for r in results:
        model = r['model']
        if model not in model_stats:
            model_stats[model] = {
                'precision': [],
                'recall': [],
                'count': 0
            }
        model_stats[model]['precision'].append(r['precision_at_5'])
        model_stats[model]['recall'].append(r['recall_at_5'])
        model_stats[model]['count'] += 1
    
    # 计算平均值
    summary = []
    for model, stats in sorted(model_stats.items()):
        avg_precision = sum(stats['precision']) / len(stats['precision'])
        avg_recall = sum(stats['recall']) / len(stats['recall'])
        summary.append({
            'model': model,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'num_projects': stats['count']
        })
    
    return summary

def aggregate_by_project(results):
    """按项目汇总"""
    project_stats = {}
    
    for r in results:
        project = r['project']
        if project not in project_stats:
            project_stats[project] = {
                'precision': [],
                'recall': [],
                'count': 0
            }
        project_stats[project]['precision'].append(r['precision_at_5'])
        project_stats[project]['recall'].append(r['recall_at_5'])
        project_stats[project]['count'] += 1
    
    # 计算平均值
    summary = []
    for project, stats in sorted(project_stats.items()):
        avg_precision = sum(stats['precision']) / len(stats['precision'])
        avg_recall = sum(stats['recall']) / len(stats['recall'])
        summary.append({
            'project': project,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'num_models': stats['count']
        })
    
    return summary

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("RQ3: Externalization Analysis")
    print("=" * 60)
    
    # 1. 统计所有 AR 结果
    results = analyze_ar_results()
    print(f"\n总结果数: {len(results)}")
    
    # 2. 按模型汇总
    model_summary = aggregate_by_model(results)
    print("\n" + "=" * 60)
    print("按模型汇总 (Average Precision@5 & Recall@5)")
    print("=" * 60)
    print(f"{'Model':<30} {'Precision@5':>12} {'Recall@5':>12} {'#Projects':>10}")
    print("-" * 60)
    
    model_data = []
    for s in model_summary:
        print(f"{s['model']:<30} {s['avg_precision']:>12.3f} {s['avg_recall']:>12.3f} {s['num_projects']:>10}")
        model_data.append(s)
    
    # 保存模型汇总
    with open(os.path.join(OUTPUT_DIR, "ar_by_model.json"), 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # 3. 按项目汇总
    project_summary = aggregate_by_project(results)
    print("\n" + "=" * 60)
    print("按项目汇总")
    print("=" * 60)
    print(f"{'Project':<20} {'Precision@5':>12} {'Recall@5':>12} {'#Models':>10}")
    print("-" * 60)
    
    project_data = []
    for s in project_summary:
        print(f"{s['project']:<20} {s['avg_precision']:>12.3f} {s['avg_recall']:>12.3f} {s['num_models']:>10}")
        project_data.append(s)
    
    # 保存项目汇总
    with open(os.path.join(OUTPUT_DIR, "ar_by_project.json"), 'w') as f:
        json.dump(project_data, f, indent=2)
    
    # 4. 保存详细结果
    with open(os.path.join(OUTPUT_DIR, "ar_all_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {OUTPUT_DIR}/")
    print("  - ar_by_model.json")
    print("  - ar_by_project.json")
    print("  - ar_all_results.json")

if __name__ == "__main__":
    main()