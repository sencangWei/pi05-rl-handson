"""Validate paired learned-policy rollouts and report every success and failure."""
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/20261029-osc-sft"
EXP = "osc_expert_replay2000_v1"


def main():
    status = json.loads((ROOT / f"evidence/{EXP}-status.json").read_text())
    assert status["status"] == "COMPLETE" and status["actual_updates"] == 2000
    assert status["sampling_preview_exact_match"] is True
    metrics = re.findall(r"Step (\d+): grad_norm=([\d.e+-]+), loss=([\d.e+-]+), param_norm=([\d.e+-]+)",
                         (ROOT / f"logs/{EXP}-train.log").read_text())
    assert [int(r[0]) for r in metrics] == list(range(0, 2000, 10))
    assert all(math.isfinite(float(v)) for r in metrics for v in r[1:])
    comparison_dir = OUT / "parent-comparison"
    if not comparison_dir.exists():
        subprocess.run([str(ROOT / ".venv-train/bin/python"), "scripts/package_recovery_transfer.py",
            "--before", str(OUT / "parent"), "--after", str(OUT / "candidate"),
            "--output", str(comparison_dir)], cwd=ROOT, check=True)
    comparison = json.loads((comparison_dir / "comparison.json").read_text())
    assert comparison["status"] == "PASS" and comparison["matched_states"] == 12
    results = {k: {r["id"]: r for r in json.loads((OUT / k / "summary.json").read_text())}
               for k in ("parent", "candidate")}
    categories = ("empty_grasp", "object_offset", "placement_failure")
    totals = {k: dict(successes=sum(r["success"] for r in rows.values()), trials=len(rows),
        categories={c: sum(r["success"] for r in rows.values() if r["category"] == c) for c in categories},
        total_controls_including_timeouts=sum(r["steps"] for r in rows.values()),
        wall_seconds=sum(r["wall_seconds"] for r in rows.values())) for k, rows in results.items()}
    shared = [n for n in results["parent"] if all(rows[n]["success"] for rows in results.values())]
    medians = {k: statistics.median(rows[n]["steps"] for n in shared) if shared else None
               for k, rows in results.items()}
    delta = [results["candidate"][n]["steps"] - results["parent"][n]["steps"] for n in shared]
    gates = dict(success_at_least_10_of_12=totals["candidate"]["successes"] >= 10,
        paired_controls_median_no_worse_than_parent=bool(shared) and medians["candidate"] <= medians["parent"])
    output = dict(status="COMPLETE", experiment=EXP, totals=totals,
        shared_success_ids=shared, paired_success_median_controls=medians,
        median_paired_delta_controls=statistics.median(delta) if delta else None,
        predeclared_gates=gates, diagnostic_gates_passed=all(gates.values()),
        automatic_promotion=False, heldout_evaluated=False, ordinary_40task_retention_evaluated=False,
        data_scope="12 reused development states; not unseen generalization; expert demonstrations are not policy tests",
        actual_source_counts=status["consumed_gradient_source_counts"],
        metrics=dict(logged_updates=len(metrics), all_finite=True,
                     first_loss=float(metrics[0][2]), last_loss=float(metrics[-1][2])))
    (OUT / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    names = {"parent": "人工纠错前40任务父模型", "candidate": "本轮专家补充微调模型"}
    intro = (REPORT / "README.md").read_text().split("\n## 实际结果", 1)[0]
    intro = intro.replace("状态：训练前协议固定，尚未产生模型评测结论。", "状态：2000次更新与父模型/候选各12例仿真评测已完成。")
    lines = [intro.rstrip(), "", "## 实际结果", "",
        "| 模型 | 成功/12 | 抓空/4 | 偏移/4 | 放置失败/4 | 总控制步（含超时） |",
        "|---|---:|---:|---:|---:|---:|"]
    for k, total in totals.items():
        lines.append(f"| {names[k]} | {total['successes']}/12 | " +
            " | ".join(f"{total['categories'][c]}/4" for c in categories) +
            f" | {total['total_controls_including_timeouts']} |")
    lines += ["", f"共同成功{len(shared)}例，控制步中位数{medians['parent']}→{medians['candidate']}；逐场差值中位数{output['median_paired_delta_controls']}。",
        f"预先冻结门槛：{json.dumps(gates, ensure_ascii=False)}。", "",
        "全部失败、超时保留在分母。每次最多300步=15秒仿真时间，墙钟时长分别记录，不等同于交互面板响应延迟。",
        "本轮未评测封存集、普通40任务旧技能；未替换面板。12例是重复使用的开发集，不能宣称未知场景泛化。", "",
        "## 训练证据", "",
        f"实际梯度样本来源：{json.dumps(output['actual_source_counts'])}。",
        "2000个梯度样本与冻结种子预览逐项相同；额外预取不计训练。200条已记录loss/梯度/参数范数均有限。",
        "人工原数据全部保留。两段新增监督来自人类恢复轨迹的专家OSC执行，不能当作模型自己完成任务。",
        "本轮为监督LoRA微调，无新增PPO。新增监督与抽样分布共同变化，不能声称单一因素的因果改善。", "",
        "## 同场景左右录像", "",
        "左=父模型，右=本轮模型，20Hz原仿真时间；先结束的一侧冻结末帧。", "",
        "| 场景 | 父模型 | 本轮 | 最长低运动步（父/新） | 同步录像 |",
        "|---|---|---|---|---|"]
    for row in comparison["rows"]:
        before, after = row["models"]["before"], row["models"]["after"]
        cells = [row["id"], *[("成功" if r["success"] else "超时") + f" {r['steps']}步" for r in (before, after)],
                 f"{before['longest_low_motion_controls']}/{after['longest_low_motion_controls']}",
                 f"[录像](../../artifacts/20261029-osc-sft/parent-comparison/{row['comparison_video']})"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "低运动=末端单步位移小于0.2mm且未满足原生目标，只作停滞提示，不能单独证明原因。",
              "复现命令见 commands.md；输入协议与校验见 evidence/osc-sft29-*；逐步轨迹和完整结果见 artifacts/20261029-osc-sft。", ""]
    (REPORT / "README.md").write_text("\n".join(lines))
    for path in (ROOT / "logs").glob(EXP + "-*.log"):
        shutil.copyfile(path, REPORT / path.name)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
