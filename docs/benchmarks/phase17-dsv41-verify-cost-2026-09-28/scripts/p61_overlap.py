import json,os,numpy as np
R=os.path.expanduser("~/p30-records-clamp")
for L in (0,2,14,20,30,39):
    rec=json.load(open(f"{R}/L{L:02d}.json"))
    picks=[r[1] for r in rec]
    for W in (4,):
        fr=[]
        for i in range(0,len(picks)-W,W):
            u=set(); [u.update(p) for p in picks[i:i+W]]
            fr.append(len(u)/(6*W))
        print(f"L{L:02d} window {W}: unique experts / slots = {np.mean(fr):.3f}")
