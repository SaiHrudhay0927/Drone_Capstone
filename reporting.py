"""Reference-style training logs, individual curves, then one combined line chart."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ('dqn', 'ddqn', 'newdqn')
COLORS = {'newdqn': '#2563eb', 'dqn': '#ed6b24', 'ddqn': '#8a8a8a'}

def read_records(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def rolling(values, window=30):
    return [float(np.mean(values[max(0,i-window+1):i+1])) for i in range(len(values))]

def progress_line(records):
    recent=records[-50:]
    return (f"Episode: {int(records[-1]['episode']):4d} | Avg Reward: "
            f"{np.mean([float(r['reward']) for r in recent]):7.2f} | "
            f"Success Rate (last {len(recent)}): "
            f"{100*np.mean([float(r['success']) for r in recent]):5.1f}% | "
            f"Epsilon: {float(records[-1]['epsilon']):.2f}")

def training_log(records, algorithm, seed):
    lines=[f"{algorithm.upper()} Training Results | Seed {seed}", '']
    for n in range(1,len(records)+1):
        if n%50==0 or n==len(records): lines.append(progress_line(records[:n]))
    return '\n'.join(lines)

def make_individual(destination, algorithm, seed, window=30):
    destination=Path(destination)
    records=read_records(destination/'training.csv')
    log=training_log(records,algorithm,seed)
    (destination/'training_log.txt').write_text(log+'\n')
    episodes=[int(r['episode']) for r in records]
    success=rolling([float(r['success']) for r in records],window)
    def draw(ax):
        ax.plot(episodes,success,color='blue',linewidth=1.8)
        ax.set(title=f'{algorithm.upper()} Training Success Rate Curve',xlabel='Epoch / Episode',
               ylabel='Success Rate',ylim=(-.03,1.05),xlim=(0,episodes[-1]))
        ax.grid(True,alpha=.45)
    fig,ax=plt.subplots(figsize=(10,4.5));draw(ax)
    fig.text(.5,.01,f'Seed {seed} | {window}-episode rolling mean | actual training data',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(destination/'success_curve.png',dpi=160);plt.close(fig)
    log_height=max(1.4,len(log.splitlines())*.17)
    fig=plt.figure(figsize=(11,log_height+4.8))
    grid=fig.add_gridspec(2,1,height_ratios=[log_height,4.3],hspace=.12)
    text_ax=fig.add_subplot(grid[0]);text_ax.axis('off')
    text_ax.text(0,1,log,va='top',family='monospace',fontsize=8.2,linespacing=1.3)
    draw(fig.add_subplot(grid[1]))
    fig.text(.5,.015,f'Seed {seed} | curve: rolling {window} | logs: last 50 | epsilon used in logged episode',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.98,bottom=.08)
    fig.savefig(destination/'training_report.png',dpi=160);plt.close(fig)
    return records

def make_training_reports(output, window=30):
    output=Path(output)
    runs={}
    for algorithm in ORDER:
        runs[algorithm]={}
        for destination in sorted((output/algorithm).glob('seed_*')):
            seed=int(destination.name.split('_')[1])
            runs[algorithm][seed]=make_individual(destination,algorithm,seed,window)
    if not all(runs.values()): raise ValueError('Combined report requires all three algorithm folders')
    report_dir=output/'training_reports';report_dir.mkdir(exist_ok=True)
    def combined(seed=None):
        fig,ax=plt.subplots(figsize=(10,5.6))
        for algorithm in ('newdqn','dqn','ddqn'):
            selected=[runs[algorithm][seed]] if seed is not None else list(runs[algorithm].values())
            episode_lists=[[int(r['episode']) for r in rows] for rows in selected]
            if any(e!=episode_lists[0] for e in episode_lists): raise ValueError('Mismatched episode axes')
            curves=[rolling([float(r['success']) for r in rows],window) for rows in selected]
            ax.plot(episode_lists[0],np.mean(curves,axis=0),color=COLORS[algorithm],label=algorithm.upper(),linewidth=1.8)
        ax.set(title='Algorithm Training Success Rate',xlabel='Epoch / Episode',ylabel='Success Rate',ylim=(-.03,1.05))
        ax.grid(axis='y',alpha=.25)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,frameon=False)
        detail=f'Seed {seed}' if seed is not None else 'Mean across training seeds: '+', '.join(map(str,sorted(runs['dqn'])))
        fig.text(.5,.02,f'{detail} | {window}-episode rolling mean | measured results',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.065,1,1))
        name=f'combined_success_seed_{seed}.png' if seed is not None else 'combined_success_all_seeds.png'
        fig.savefig(report_dir/name,dpi=180);plt.close(fig)
    shared=set.intersection(*(set(runs[a]) for a in ORDER))
    for seed in sorted(shared): combined(seed)
    combined()
    lines=['# Training output — each algorithm, then comparison','',
           'Actual saved training logs; 30-episode rolling success curves. No retraining or altered outcomes.','']
    for seed in sorted(shared):
        lines += [f'## Seed {seed}','']
        for algorithm in ORDER:
            lines += [f'### {algorithm.upper()}','',f'![{algorithm} training](../{algorithm}/seed_{seed}/training_report.png)','']
        lines += ['### All three algorithms','',f'![Combined](combined_success_seed_{seed}.png)','']
    lines += ['## All training seeds averaged','', '![Combined means](combined_success_all_seeds.png)','']
    (report_dir/'TRAINING_REPORTS.md').write_text('\n'.join(lines))
    return report_dir

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/faculty_run')
    parser.add_argument('--window',type=int,default=30)
    args=parser.parse_args()
    if args.window<1: parser.error('window must be positive')
    print(make_training_reports(args.results,args.window))
