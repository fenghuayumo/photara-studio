"""Summarize paired final held-out PSNR, without mixing milestone evaluations."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--baseline',type=Path)
p.add_argument('--scenes',nargs='+',choices=['train','nyc'],default=['train','nyc'])
a=p.parse_args()
summary=[]
for scene in a.scenes:
    values={}
    for strategy in ['adc_plus','adc_igs']:
        directory=a.baseline if strategy=='adc_plus' and a.baseline else a.directory
        text=(directory/f'{scene}_{strategy}'/'run.log').read_text(errors='replace')
        values[strategy]={int(i):float(v) for i,v in re.findall(r'splat_render=.*? view=(\d+) foreground_psnr=([\d.]+)',text)}
    if not values['adc_plus'] or values['adc_plus'].keys()!=values['adc_igs'].keys():
        raise ValueError(f'{scene}: incomplete or mismatched holdout views')
    differences=[values['adc_igs'][i]-v for i,v in values['adc_plus'].items()]
    summary.append(dict(scene=scene,views=len(differences),
        adc_plus=sum(values['adc_plus'].values())/len(differences),
        igs=sum(values['adc_igs'].values())/len(differences),
        mean_gain=sum(differences)/len(differences),
        improved_views=sum(d>0 for d in differences),
        minimum_gain=min(differences),maximum_gain=max(differences)))
(a.directory/'paired_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
