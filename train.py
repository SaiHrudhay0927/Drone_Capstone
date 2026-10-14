"""Train just one algorithm, independently."""
import argparse
from config import load_config
from training import train,AGENTS

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm",choices=AGENTS,required=True)
    p.add_argument("--config",default="configs/default.json")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--episodes",type=int)
    p.add_argument("--output",default="results/local_single")
    a=p.parse_args();c=load_config(a.config)
    if a.episodes is not None:
        if a.episodes<1: p.error("episodes must be positive")
        c["episodes"]=a.episodes
    train(a.algorithm,c,a.seed,a.output)
