"""Shared training pipeline; checkpoints include both networks and run configuration."""
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from agents.dqn import DQNAgent
from agents.ddqn import DDQNAgent
from agents.newdqn import NEWDQNAgent
from environment.grid_env import DroneGridEnv
from evaluation import evaluate
from reporting import progress_line, make_individual

AGENTS={"dqn":DQNAgent,"ddqn":DDQNAgent,"newdqn":NEWDQNAgent}

def train(algorithm, config, seed, output):
    destination=Path(output)/algorithm/f"seed_{seed}"
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"config.json").write_text(json.dumps(config,indent=2))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    agent=AGENTS[algorithm](config["agent"],seed)
    env=DroneGridEnv(config["train_map"],config["environment"])
    records=[]
    print(f"Starting {algorithm.upper()} Training Loop | Seed {seed}...",flush=True)
    start=time.perf_counter()
    for episode in range(1,config["episodes"]+1):
        state=env.reset(seed*1000000+episode)
        total=0.;losses=[]
        epsilon=agent.epsilon
        while True:
            action=agent.select_action(state)
            ns,reward,terminal,truncated,info=env.step(action,moving=episode>config["stationary_episodes"])
            # Timeout has a terminal penalty; this is a finite-episode task, so no bootstrap at timeout.
            agent.replay.add((state.copy(),action,reward,ns.copy(),terminal or truncated))
            loss=agent.train_step()
            if loss is not None: losses.append(loss)
            state=ns;total+=reward
            if terminal or truncated: break
        agent.end_episode(info["success"])
        if episode % config["agent"]["target_sync_episodes"]==0: agent.sync_target()
        records.append({"episode":episode,"reward":total,"success":int(info["success"]),
                        "steps":info["steps"],"outcome":info["outcome"],"epsilon":epsilon,
                        "loss":float(np.mean(losses)) if losses else "","obe_resets":agent.obe_resets})
        if episode%50==0 or episode==config["episodes"]:
            print(progress_line(records),flush=True)
    seconds=time.perf_counter()-start
    with (destination/"training.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    torch.save({"algorithm":algorithm,"seed":seed,"config":config,"online":agent.online.state_dict(),
                "target":agent.target.state_dict(),"optimizer":agent.optimizer.state_dict(),
                "epsilon":agent.epsilon,"updates":agent.updates},destination/"model.pt")
    with (destination/"replay.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["pool","state","action","reward","next_state","done"])
        for pool,(s,a,r,ns,d) in agent.replay.records():
            writer.writerow([pool,json.dumps(s.tolist()),a,r,json.dumps(ns.tolist()),d])
    summaries=evaluate(agent,config,destination)
    summary={"algorithm":algorithm,"seed":seed,"training_seconds":seconds,
             "episodes":config["episodes"],"updates":agent.updates,"obe_resets":agent.obe_resets,
             "python":platform.python_version(),"torch":str(torch.__version__),"numpy":np.__version__,
             "device":"cpu","evaluation":summaries}
    (destination/"summary.json").write_text(json.dumps(summary,indent=2))
    make_individual(destination,algorithm,seed)
    return summary

def load_agent(checkpoint):
    data=torch.load(checkpoint,map_location="cpu",weights_only=True)
    agent=AGENTS[data["algorithm"]](data["config"]["agent"],data["seed"])
    agent.online.load_state_dict(data["online"]);agent.online.eval()
    agent.target.load_state_dict(data["target"])
    return agent,data
