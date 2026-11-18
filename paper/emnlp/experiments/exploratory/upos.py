"""Exploratory: how much would a part-of-speech key resolve on forms seen with several lemmas?"""
import json, collections, glob, os
R="/mnt/hdd-4tb/codex-runs/udlex-v1-order-ud219-20261118.splMl6"
inv=json.load(open(R+"/clean219/inputs/order/inventory.json"))["languages"]
cov=json.load(open("/mnt/hdd-4tb/codex-runs/udlex-compare-20261119/results/stanza-coverage.json"))
clean=[i for i,v in cov.items() if v["lemma_model"] and 2*v["covered"]<v["unseen_types"]]
def rows(path):
    for l in open(path,encoding="utf-8"):
        p=l.rstrip("\n").split("\t")
        if len(p)==10 and p[0].isdecimal() and p[1]!="_" and p[2]!="_": yield p[1],p[2],p[3],p[5]
tot=collections.Counter(); macro=collections.defaultdict(list)
for iso in clean:
    row=inv[iso]; d=f"{R}/inputs/ud-treebanks-v2.19/{row['treebank']}/{row['config']}-ud-"
    byf=collections.defaultdict(collections.Counter); byp=collections.defaultdict(collections.Counter); byx=collections.defaultdict(collections.Counter)
    for f,y,u,feats in rows(d+"train.conllu"):
        byf[f][y]+=1; byp[(f,u)][y]+=1; byx[(f,u,feats)][y]+=1
    c=collections.Counter()
    for f,y,u,feats in rows(d+"test.conllu"):
        if f not in byf or len(byf[f])<2: continue
        c["n"]+=1
        c["form"]+=byf[f].most_common(1)[0][0]==y
        k=byp.get((f,u)); c["upos"]+=(k.most_common(1)[0][0] if k else byf[f].most_common(1)[0][0])==y
        k2=byx.get((f,u,feats)); c["feats"]+=((k2 or k or byf[f]).most_common(1)[0][0])==y
    tot.update(c)
    if c["n"]:
        for k in ("form","upos","feats"): macro[k].append(100*c[k]/c["n"])
print("forms seen in training with several lemmas, %d test words in %d languages"%(tot["n"],len(clean)))
for k,name in (("form","majority lemma of the form"),("upos","majority lemma of (form, gold UPOS)"),("feats","majority lemma of (form, gold UPOS, gold features)")):
    print(f"  {name:48s} pooled {100*tot[k]/tot['n']:.2f}  macro {sum(macro[k])/len(macro[k]):.2f}")
