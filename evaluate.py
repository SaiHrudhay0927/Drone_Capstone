"""Evaluate an existing checkpoint without retraining."""
import argparse
import json
from pathlib import Path
import torch
from training import load_agent
from evaluation import evaluate

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--episodes",type=int,default=200)
    p.add_argument("--output",default="results/local_evaluation")
    a=p.parse_args()
    if a.episodes<1: p.error("episodes must be positive")
    torch.set_num_threads(1)
    agent,data=load_agent(a.checkpoint);c=data["config"];c["eval_episodes"]=a.episodes
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    summary=evaluate(agent,c,out)
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
