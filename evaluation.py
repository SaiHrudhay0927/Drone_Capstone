"""Greedy, held-out evaluation. Same seed list for every algorithm; no learning."""
import csv
import json
import numpy as np
from environment.grid_env import DroneGridEnv

def rollout(agent, env, seed):
    state = env.reset(seed)
    path = [{"drone":env.drone,"target":env.target}]
    total = 0.0
    while True:
        action = agent.select_action(state,greedy=True)
        state,reward,terminated,truncated,info = env.step(action,moving=True)
        total += reward
        path.append({"drone":env.drone,"target":env.target})
        if terminated or truncated: break
    return {"scenario_seed":seed,"success":int(info["success"]),"outcome":info["outcome"],
            "steps":info["steps"],"reward":total},path

def evaluate(agent, config, destination):
    summaries=[]
    for map_name in config["test_maps"]:
        env=DroneGridEnv(map_name,config["environment"])
        records=[]
        for i in range(config["eval_episodes"]):
            record,path=rollout(agent,env,100000+i)
            records.append(record)
            if i == 0:
                (destination/f"trajectory_{map_name}.json").write_text(json.dumps(
                    {"map":map_name,"record":record,"path":path},indent=2))
        with (destination/f"evaluation_{map_name}.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        successful=[r["steps"] for r in records if r["success"]]
        summaries.append({"map":map_name,"episodes":len(records),
            "success_rate":float(np.mean([r["success"] for r in records])),
            "mean_steps_success":float(np.mean(successful)) if successful else None,
            "mean_reward":float(np.mean([r["reward"] for r in records])),
            "collision_rate":float(np.mean([r["outcome"]=="collision" for r in records])),
            "timeout_rate":float(np.mean([r["outcome"]=="timeout" for r in records]))})
    return summaries
