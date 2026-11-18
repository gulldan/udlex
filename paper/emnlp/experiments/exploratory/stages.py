"""Exploratory: accuracy of the declared cascade by stage, on forms seen and unseen in training."""
import json, collections, glob
O="/mnt/hdd-4tb/codex-runs/udlex-v1-order-ud219-20261118.splMl6/clean219/inputs/order"
D="/mnt/hdd-4tb/codex-runs/udlex-transfer-20261118-full/anchor-diag3/results"
STAGE={2:"corpus",3:"corpus",4:"corpus",5:"external",6:"external",7:"rule",8:"copy"}
tot=collections.Counter(); per={}
for path in sorted(glob.glob(D+"/*.json")):
    iso=path.split("/")[-1][:-5]
    if iso=="run": continue
    r=json.load(open(path)); ans={a[0]:a[1:] for a in r["answers"]}
    train=set()
    for l in open(f"{O}/{iso}/train.jsonl"):
        s=json.loads(l); train.update(f for f,y in zip(s["tokens"],s["lemmas"]) if y is not None)
    c=collections.Counter()
    for l in open(f"{O}/{iso}/test.jsonl"):
        s=json.loads(l)
        for i,(f,y) in enumerate(zip(s["tokens"],s["lemmas"])):
            if y is None: continue
            a=ans[f]; text,stage=(a[2],a[3]) if i==0 else (a[0],a[1])
            k=("seen" if f in train else "unseen", STAGE.get(stage,str(stage)))
            c[k+("n",)]+=1; c[k+("ok",)]+=text==y
    assert sum(v for k,v in c.items() if k[2]=="ok")==r["test"]["counts"][0]["correct"], iso
    per[iso]=c; tot.update(c)
n=sum(v for k,v in tot.items() if k[2]=="n")
print("pooled over",len(per),"languages,",n,"words")
for seen in ("seen","unseen"):
    sn=sum(v for k,v in tot.items() if k[0]==seen and k[2]=="n"); so=sum(v for k,v in tot.items() if k[0]==seen and k[2]=="ok")
    print(f" {seen}: {100*sn/n:.1f}% of words, accuracy {100*so/sn:.2f}")
    for st in ("corpus","external","rule","copy"):
        a,b=tot[(seen,st,"n")],tot[(seen,st,"ok")]
        if a: print(f"    {st:9s} {100*a/n:5.2f}% of words  accuracy {100*b/a:.2f}  errors {a-b}")
# macro over languages of unseen accuracy
ua=[100*sum(v for k,v in c.items() if k[0]=="unseen" and k[2]=="ok")/max(1,sum(v for k,v in c.items() if k[0]=="unseen" and k[2]=="n")) for c in per.values()]
sa=[100*sum(v for k,v in c.items() if k[0]=="seen" and k[2]=="ok")/max(1,sum(v for k,v in c.items() if k[0]=="seen" and k[2]=="n")) for c in per.values()]
print("macro: accuracy on seen %.2f, on unseen %.2f"%(sum(sa)/len(sa),sum(ua)/len(ua)))
json.dump({i:{"|".join(k):v for k,v in c.items()} for i,c in per.items()},open("/mnt/hdd-4tb/codex-runs/udlex-compare-20261119/stages.json","w"))
