"""Charts use actual CSV logs, never illustrative or paper-copied curves."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS={"dqn":"#2563eb","ddqn":"#e87924","newdqn":"#059669"}

def smooth(x,window=50):
    return np.array([np.mean(x[max(0,i-window+1):i+1]) for i in range(len(x))])

def make_comparison(output):
    output=Path(output)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    all_summaries=[]
    for alg,color in COLORS.items():
        runs=[]
        for path in sorted((output/alg).glob("seed_*")):
            with (path/"training.csv").open() as f: runs.append(list(csv.DictReader(f)))
            all_summaries.append(json.loads((path/"summary.json").read_text()))
        if not runs: continue
        for ax,key in zip(axes,["success","reward"]):
            values=np.array([smooth([float(r[key]) for r in run]) for run in runs])
            mean=values.mean(0);sd=values.std(0)
            xs=np.arange(1,len(mean)+1)
            ax.plot(xs,mean,label=alg.upper(),color=color)
            ax.fill_between(xs,mean-sd,mean+sd,color=color,alpha=.12)
            ax.set_xlabel("Training episode");ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel("Training success (rolling 50 episodes)");axes[0].set_ylim(0,1)
    axes[1].set_ylabel("Training reward (rolling 50 episodes)")
    fig.suptitle("Measured training results • mean ± population SD across seeds")
    fig.tight_layout();fig.savefig(output/"training_comparison.png",dpi=180);plt.close(fig)
    rows=[]
    for alg in COLORS:
        entries=[s for s in all_summaries if s["algorithm"]==alg]
        if not entries: continue
        for map_name in [v["map"] for v in entries[0]["evaluation"]]:
            metrics=[v for s in entries for v in s["evaluation"] if v["map"]==map_name]
            successes=[m["success_rate"] for m in metrics]
            steps=[m["mean_steps_success"] for m in metrics if m["mean_steps_success"] is not None]
            rows.append({"algorithm":alg,"map":map_name,"training_seeds":len(entries),
                         "eval_episodes_per_seed":metrics[0]["episodes"],
                         "success_mean":float(np.mean(successes)),"success_sd":float(np.std(successes)),
                         "mean_steps_success":float(np.mean(steps)) if steps else None,
                         "mean_training_seconds":float(np.mean([s["training_seconds"] for s in entries]))})
    with (output/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/"comparison.json").write_text(json.dumps(rows,indent=2))
    maps=list(dict.fromkeys(r["map"] for r in rows))
    fig,ax=plt.subplots(figsize=(8,4.8))
    for i,(alg,color) in enumerate(COLORS.items()):
        selected=[r for r in rows if r["algorithm"]==alg]
        ax.bar(np.arange(len(maps))+(i-1)*.24,[r["success_mean"] for r in selected],.24,
               yerr=[r["success_sd"] for r in selected],label=alg.upper(),color=color,capsize=4)
    ax.set_xticks(range(len(maps)),maps);ax.set_ylim(0,1);ax.set_ylabel("Greedy held-out capture rate")
    ax.set_title("Evaluation • mean ± population SD across training seeds")
    ax.legend();fig.tight_layout();fig.savefig(output/"evaluation_comparison.png",dpi=180);plt.close(fig)
    lines=["# Measured results", "", "No ranking is guaranteed. These results are from this implementation, not the paper.", "",
           "| Algorithm | Map | Success mean ± SD | Successful steps* |", "|---|---|---:|---:|"]
    for r in rows:
        step_text=f"{r['mean_steps_success']:.2f}" if r['mean_steps_success'] is not None else "N/A"
        lines.append(f"| {r['algorithm'].upper()} | {r['map']} | {100*r['success_mean']:.1f}% ± {100*r['success_sd']:.1f}% | {step_text} |")
    lines += ["", "*Mean of per-seed successful-episode step means; failures excluded. Compare alongside success rate.",
              "", "Evaluation: 200 held-out scenarios per map per seed by default; greedy actions, moving target, no training updates.",
              "Error bars describe variation across training seeds, not confidence intervals. See config.json for actual episode counts.",
              "", "![Training](training_comparison.png)","![Evaluation](evaluation_comparison.png)"]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
