"""Execute all three algorithms and generate measured comparison artifacts."""
import argparse
import json
from pathlib import Path
from config import load_config
from training import train,AGENTS
from plotting import make_comparison
from reporting import make_training_reports

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/default.json")
    p.add_argument("--episodes",type=int)
    p.add_argument("--eval-episodes",type=int)
    p.add_argument("--seeds",type=int,nargs="+")
    p.add_argument("--output",default="results/local_comparison")
    a=p.parse_args();c=load_config(a.config)
    if a.episodes is not None: c["episodes"]=a.episodes
    if a.eval_episodes is not None: c["eval_episodes"]=a.eval_episodes
    if a.seeds is not None: c["seeds"]=a.seeds
    if c["episodes"]<1 or c["eval_episodes"]<1: p.error("Episode counts must be positive")
    out=Path(a.output)
    # Refuse mixed old/new runs before doing expensive work.
    if out.exists() and any(out.iterdir()): p.error("Output directory must be new or empty")
    out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(c,indent=2))
    for algorithm in AGENTS:
        for seed in c["seeds"]: train(algorithm,c,seed,out)
    make_comparison(out)
    print("Training reports:", make_training_reports(out))
