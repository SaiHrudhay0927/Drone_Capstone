"""Replay a saved model on an explicit scenario, saving a GIF and trajectory plot."""
import argparse
import json
from pathlib import Path
import torch
from environment.grid_env import DroneGridEnv
from environment.maps import MAPS
from evaluation import rollout
from training import load_agent

def demo(checkpoint,map_name,seed,output,show=False):
    import matplotlib
    if not show: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    torch.set_num_threads(1)
    agent,data=load_agent(checkpoint)
    env=DroneGridEnv(map_name,data["config"]["environment"])
    record,path=rollout(agent,env,seed)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stem=f"{data['algorithm']}_{map_name}_{seed}"
    (output/f"{stem}.json").write_text(json.dumps({"record":record,"path":path},indent=2))
    fig,ax=plt.subplots(figsize=(6,6));m=MAPS[map_name]
    for r,c in m.obstacles: ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,color="#334155"))
    ax.set(xlim=(-.5,m.size-.5),ylim=(m.size-.5,-.5),xlabel="Column",ylabel="Row")
    ax.set_xticks(range(m.size));ax.set_yticks(range(m.size));ax.grid(alpha=.2);ax.set_aspect("equal")
    drone,=ax.plot([],[],"o-",color="#2563eb",label="Drone")
    target,=ax.plot([],[],"s--",color="#e87924",label="Target")
    ax.legend(loc="upper right")
    def frame(i):
        subset=path[:i+1]
        drone.set_data([p["drone"][1] for p in subset],[p["drone"][0] for p in subset])
        target.set_data([p["target"][1] for p in subset],[p["target"][0] for p in subset])
        status=record["outcome"] if i==len(path)-1 else "tracking"
        ax.set_title(f"{data['algorithm'].upper()} | {map_name} | step {i} | {status}")
        return drone,target
    animation=FuncAnimation(fig,frame,frames=len(path),interval=250,repeat=True)
    animation.save(output/f"{stem}.gif",writer=PillowWriter(fps=4))
    frame(len(path)-1);fig.tight_layout();fig.savefig(output/f"{stem}.png",dpi=150)
    if show: plt.show()
    plt.close(fig)
    print(json.dumps(record,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--map",choices=MAPS,default="paper8")
    p.add_argument("--scenario-seed",type=int,default=100000)
    p.add_argument("--output",default="results/demos")
    p.add_argument("--show",action="store_true")
    a=p.parse_args();demo(a.checkpoint,a.map,a.scenario_seed,a.output,a.show)
