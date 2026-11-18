"""Exploratory: how far can any context-free lemmatizer go on the official tests?"""
import json, gzip, collections
O="/mnt/hdd-4tb/codex-runs/udlex-v1-order-ud219-20261118.splMl6/clean219/inputs/order"
C="/mnt/hdd-4tb/codex-runs/udlex-compare-20261119"
inv=json.load(open(O+"/inventory.json"))["languages"]
home=json.loads(gzip.decompress(open(C+"/study/evidence/ud219/order/pairs.json.gz","rb").read()))["official"]
st=json.load(open(C+"/results/stanza.json"))["cells"]; cov=json.load(open(C+"/results/stanza-coverage.json"))
def read(iso,split):
    out=[]
    for l in open(f"{O}/{iso}/{split}.jsonl"):
        r=json.loads(l)
        for i,(f,y) in enumerate(zip(r["tokens"],r["lemmas"])):
            if y is not None: out.append((f,y,i==0))
    return out
rows=[]
for iso in sorted(home):
    tr=read(iso,"train"); te=read(iso,"test")
    n=len(te)
    # oracle: best form->lemma map for this very test
    by=collections.defaultdict(collections.Counter)
    for f,y,_ in te: by[f][y]+=1
    oracle=sum(max(c.values()) for c in by.values())
    # memory: majority lemma from train, else copy
    tm=collections.defaultdict(collections.Counter)
    for f,y,_ in tr: tm[f][y]+=1
    best={f:c.most_common(1)[0][0] for f,c in tm.items()}
    seen=[(f,y) for f,y,_ in te if f in best]; unseen=[(f,y) for f,y,_ in te if f not in best]
    mem_seen=sum(best[f]==y for f,y in seen); copy_unseen=sum(f==y for f,y in unseen)
    # oracle restricted to seen/unseen
    or_seen=sum(max(c.values()) for f,c in by.items() if f in best); or_unseen=oracle-or_seen
    ours=home[iso][0]["test"]["counts"][0]["correct"]
    s=st.get(iso,{}); ok=s.get("status")=="scored" and cov.get(iso,{}).get("lemma_model") and 2*cov[iso]["covered"]<cov[iso]["unseen_types"]
    rows.append(dict(iso=iso,n=n,oracle=100*oracle/n,ours=100*ours/n,stanza=100*s["correct"]/n if ok else None,
        seen_share=100*len(seen)/n, mem_seen=100*mem_seen/max(len(seen),1), or_seen=100*or_seen/max(len(seen),1),
        copy_unseen=100*copy_unseen/max(len(unseen),1), or_unseen=100*or_unseen/max(len(unseen),1),
        memory=100*(mem_seen+copy_unseen)/n))
json.dump(rows,open(C+"/ceiling.json","w"))
m=lambda k,rs: sum(r[k] for r in rs)/len(rs)
print("66 langs: oracle ceiling macro %.2f | ours %.2f | train-memory+copy %.2f | seen share %.1f | acc on seen: memory %.2f oracle %.2f | unseen: copy %.2f oracle %.2f"%(m("oracle",rows),m("ours",rows),m("memory",rows),m("seen_share",rows),m("mem_seen",rows),m("or_seen",rows),m("copy_unseen",rows),m("or_unseen",rows)))
cl=[r for r in rows if r["stanza"] is not None]
print("58 clean langs: ceiling %.2f stanza %.2f ours %.2f; stanza above ceiling in %d langs; ours within 0.5 of stanza or better in %d"%(m("oracle",cl),m("stanza",cl),m("ours",cl),sum(r["stanza"]>r["oracle"] for r in cl),sum(r["ours"]>=r["stanza"]-0.5 for r in cl)))
print("gap ceiling-stanza: min/median/max", sorted(round(r["oracle"]-r["stanza"],2) for r in cl)[::max(1,len(cl)//10)])
for r in sorted(cl,key=lambda r:r["stanza"]-r["ours"],reverse=True)[:12]:
    print(f"  {r['iso']} ours={r['ours']:.2f} stanza={r['stanza']:.2f} ceiling={r['oracle']:.2f} seen={r['seen_share']:.0f}% mem@seen={r['mem_seen']:.1f} oracle@seen={r['or_seen']:.1f} copy@unseen={r['copy_unseen']:.1f}")
