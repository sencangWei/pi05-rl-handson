"""Check complete matched rollouts and build this experiment's report/videos."""
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/20261028-taskphase"
REPORT = Path(__file__).resolve().parent
status = json.loads((ROOT / "evidence/manual6_taskphase_2000_v1-status.json").read_text())
assert status["status"] == "COMPLETE" and status["actual_updates"] == 2000
assert status["sampling_preview_exact_match"] is True
log = (ROOT / "logs/manual6_taskphase_2000_v1-train.log").read_text()
metrics = re.findall(r"Step (\d+): grad_norm=([\d.e+-]+), loss=([\d.e+-]+), param_norm=([\d.e+-]+)", log)
assert len(metrics) == 200 and [int(r[0]) for r in metrics] == list(range(0, 2000, 10))
assert all(math.isfinite(float(v)) for r in metrics for v in r[1:])
directories = {
    "parent": ROOT / "artifacts/20261028-diagnostics/physicaldev-before",
    "old_mix": ROOT / "artifacts/20261028-diagnostics/physicaldev-after",
    "sampling025": ROOT / "artifacts/20261028-sampling025/candidate",
    "candidate": OUT / "candidate",
}
results = {k: {r["id"]: r for r in json.loads((p / "summary.json").read_text())}
           for k, p in directories.items()}
assert all(len(rows) == 12 and rows.keys() == results["parent"].keys() for rows in results.values())
for name in results["parent"]:
    assert len({rows[name]["state_sha256"] for rows in results.values()}) == 1
    assert all(rows[name]["exception"] is None for rows in results.values())

paired = {}
for baseline in ("parent", "old_mix", "sampling025"):
    destination = OUT / (baseline + "-comparison")
    if not destination.exists():
        subprocess.run([str(ROOT / ".venv-train/bin/python"), "scripts/package_recovery_transfer.py",
                        "--before", str(directories[baseline]), "--after", str(directories["candidate"]),
                        "--output", str(destination)], cwd=ROOT, check=True)
    comparison = json.loads((destination / "comparison.json").read_text())
    assert comparison["status"] == "PASS" and comparison["matched_states"] == 12
    shared = [name for name in results[baseline]
              if results[baseline][name]["success"] and results["candidate"][name]["success"]]
    controls = [(results[baseline][n]["steps"], results["candidate"][n]["steps"]) for n in shared]
    paired[baseline] = {
        "shared_success_ids": shared,
        "baseline_median_controls": statistics.median(a for a, b in controls) if controls else None,
        "candidate_median_controls": statistics.median(b for a, b in controls) if controls else None,
        "median_paired_delta_controls": statistics.median(b - a for a, b in controls) if controls else None,
        "faster": sum(b < a for a, b in controls), "slower": sum(b > a for a, b in controls),
        "equal": sum(b == a for a, b in controls),
    }

categories = sorted({r["category"] for r in results["parent"].values()})
totals = {label: {
    "successes": sum(r["success"] for r in rows.values()), "trials": 12,
    "categories": {c: sum(r["success"] for r in rows.values() if r["category"] == c) for c in categories},
    "total_controls_including_timeouts": sum(r["steps"] for r in rows.values()),
    "wall_seconds": sum(r["wall_seconds"] for r in rows.values()),
} for label, rows in results.items()}
parent_pair = paired["parent"]
gates = {
    "success_at_least_10_of_12": totals["candidate"]["successes"] >= 10,
    "paired_controls_median_no_worse_than_parent": bool(parent_pair["shared_success_ids"]) and
        parent_pair["candidate_median_controls"] <= parent_pair["baseline_median_controls"],
}
diagnostics = []
for name, result in results["candidate"].items():
    trace = [json.loads(line) for line in (directories["candidate"] / (name + ".jsonl")).read_text().splitlines()]
    assert len(trace) == result["steps"]
    longest = current = 0
    for step in trace:
        current = current + 1 if step["eef_motion_m"] < .0002 and not step["native_success"] else 0
        longest = max(longest, current)
    diagnostics.append({"id": name, "success": result["success"], "steps": result["steps"],
        "native_success_controls": sum(t["native_success"] for t in trace),
        "open_command_controls": sum(t["action"][6] < 0 for t in trace),
        "longest_low_motion_controls": longest, "last_sample": trace[-1]})
evaluation = {
    "status": "COMPLETE", "experiment": status["experiment"], "totals": totals,
    "paired_comparisons": paired, "predeclared_gates": gates,
    "diagnostic_gates_passed": all(gates.values()), "automatic_promotion": False,
    "heldout_evaluated": False,
    "data_scope": "12 reused physically-valid development states; not unseen generalization",
    "intervention": "combined source-weight and command-phase balancing, not a single-variable ablation",
    "training_metrics": {"logged_updates": len(metrics), "all_logged_values_finite": True,
                         "first_loss": float(metrics[0][2]), "last_loss": float(metrics[-1][2])},
    "candidate_trace_diagnostics": diagnostics,
}
(OUT / "result.json").write_text(json.dumps(evaluation, indent=2) + "\n")
for kind in ("train", "server", "evaluation"):
    shutil.copyfile(ROOT / f"logs/manual6_taskphase_2000_v1-{kind}.log", REPORT / f"{kind}.log")

names = {"parent": "人工纠错前父模型", "old_mix": "原25%人工纠错模型",
         "sampling025": "上轮2.5%人工纠错模型", "candidate": "本轮任务＋指令阶段平衡模型"}
intro = (REPORT / "README.md").read_text().split("\n## 实际结果", 1)[0]
intro = intro.replace("状态：训练前协议已固定，结果尚待实际评测。", "状态：2000次更新及12例配对评测完成，结果如下；未替换面板模型。")
lines = [intro.rstrip(), "", "## 实际结果", "",
    "| 模型 | 成功/12 | 抓空/4 | 物体偏移/4 | 放置失败/4 | 总控制步（含超时） |",
    "|---|---:|---:|---:|---:|---:|"]
for label, total in totals.items():
    cat = total["categories"]
    lines.append(f"| {names[label]} | {total['successes']}/12 | {cat['empty_grasp']}/4 | "
                 f"{cat['object_offset']}/4 | {cat['placement_failure']}/4 | {total['total_controls_including_timeouts']} |")
lines += ["", "所有12例均计入成功率。效率仅比较双方都成功的相同场景：", ""]
for label, pair in paired.items():
    lines.append(f"- 相比{names[label]}，共同成功{len(pair['shared_success_ids'])}例；中位控制步 "
                 f"{pair['baseline_median_controls']}→{pair['candidate_median_controls']}；逐场差值中位数"
                 f"{pair['median_paired_delta_controls']}；{pair['faster']}例变快、{pair['slower']}例变慢、{pair['equal']}例相同。")
lines += ["", f"预先冻结诊断门：成功≥10/12={gates['success_at_least_10_of_12']}；"
    f"共同成功子集效率不差于父模型={gates['paired_controls_median_no_worse_than_parent']}。",
    "本轮不推广到面板；40任务旧技能回归未运行、封存集未打开。这些结果不能替代未知场景泛化验证。", "",
    "## 训练与逐步轨迹核验", "",
    f"实际来源计数：{json.dumps(status['consumed_gradient_source_counts'], ensure_ascii=False)}。", "",
    f"实际阶段计数：{json.dumps(status['consumed_gradient_phase_counts'], ensure_ascii=False)}。", "",
    "2000个梯度样本顺序与种子预览逐项相同，额外预取样本未计入梯度。200条已记录loss、梯度范数和参数范数均为有限数。",
    "实际独立起点覆盖见evidence/recovery-taskphase-28-preview-coverage.json；抽到的窗口存在重复，不能把2000次更新当作2000条不同示范。",
    "低运动量定义为末端单步位移<0.2mm且未达到原生目标；这只能提示接触或停滞，不能单独证明失败原因。",
    "动作最多300控制步=15秒仿真时间，推理预测10步执行5步；各模型预算相同。墙钟时间另外保留，不能直接当作面板响应延迟。", "",
    "| 场景 | 结果 | 控制步 | 最长低运动连续步 | 曾满足原生目标步数 | 张开指令步数 |",
    "|---|---|---:|---:|---:|---:|"]
for row in diagnostics:
    lines.append(f"| {row['id']} | {'成功' if row['success'] else '超时'} | {row['steps']} | "
                 f"{row['longest_low_motion_controls']} | {row['native_success_controls']} | {row['open_command_controls']} |")
lines += ["", "## 同场景左右同步录像", "",
    "左侧是对应基线，右侧是本轮新模型，均按20Hz仿真时间播放；先结束的一侧冻结末帧，没有加速慢动作。", "",
    "| 场景 | 父模型 | 原25%模型 | 上轮2.5%模型 | 本轮 | 与父模型对照 | 与原25%对照 | 与上轮对照 |",
    "|---|---:|---:|---:|---:|---|---|---|"]
for name in results["parent"]:
    cells = [("成功 " if results[label][name]["success"] else "超时 ") + str(results[label][name]["steps"])
             for label in names]
    links = [f"[左右录像](../../artifacts/20261028-taskphase/{label}-comparison/comparison-{name}.mp4)"
             for label in ("parent", "old_mix", "sampling025")]
    lines.append("| " + " | ".join([name] + cells + links) + " |")
lines += ["", "## 复现与边界", "",
    "- 本轮只做LoRA监督微调，没有新增PPO更新，也没有手写轨迹替代模型。",
    "- 本轮同时调整来源比例和指令阶段分布，不能把变化单独归因于其中一个因素。指令切换不等同于实际抓住或释放物体。",
    "- 人工1612个有效起点及公开task8的7633个有效起点全部仍可抽到；没有删原视频、删除失败帧或重定时动作。",
    "- 原始视频、逐步动作/末端/物体轨迹、同状态SHA、三组完整配对结果均保留在artifacts/20261028-taskphase。",
    "- 命令见本目录commands.md；冻结输入、实际环境、数据审计及备份证明见evidence/recovery-taskphase-28-*.json。",
    "- 推理LoRA包须结合官方pi05_libero基座精确恢复，优化器状态只保留本机；推理包不等于断点续训包。", ""]
(REPORT / "README.md").write_text("\n".join(lines))
(OUT / "README.md").write_text("# 任务与夹爪指令阶段抽样对照\n\n[完整报告](../../reports/recovery-taskphase-20261028/README.md)；candidate为本轮原始评测，三个comparison目录为逐场左右录像。\n")
print(json.dumps({k: v for k, v in evaluation.items() if k != "candidate_trace_diagnostics"}, indent=2))
