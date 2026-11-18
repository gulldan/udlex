"""Prepare token observations and immutable source projections, before fitting any rules."""
import gzip
import itertools
import json
import subprocess
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

from fetch_sources import inputs, sha


def nfc(value):
    return unicodedata.normalize("NFC",value)


def lexical(value):
    return bool(value and value != "_" and not any(ch.isspace() or ord(ch)<32 for ch in value))


def corpus_rows(paths, iso):
    markers = {"de_hdt":{"unknown","-","NULL"},"pcm_nsc":{"X"},"qaf_arabizi":{"X"},"say_autogramm":{"X"},
               "cu_proiel":{"?"},"sk_snk":{"?"},"myv_jr":{"-"},"cop_scriptorium":{"unknown"}}
    for path in paths:
        config=path.name.split("-ud-")[0]
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                fields=line.rstrip("\r\n").split("\t")
                if len(fields)!=10 or not fields[0].isdigit():
                    continue
                form=nfc(fields[1]); lemma=form if iso=="ssp" else nfc(fields[2])
                if not lexical(form) or not lexical(lemma) or fields[3] in {"PUNCT","SYM","PROPN"}:
                    continue
                if lemma != form and lemma in markers.get(config,set()):
                    continue
                if any("://" in x or x.casefold().startswith("www.") or "@" in x for x in (form,lemma)):
                    continue
                if iso=="naq" and (fields[3] in {"NUM","X"} or "Foreign=Yes" in fields[5]):
                    continue
                yield form,lemma,str(int(fields[0]=="1"))


def paradigms(row,cache):
    result={}
    for item in inputs(row):
        path=cache/item["sha256"]
        if sha(path)!=item["sha256"]: raise ValueError("UniMorph input hash mismatch")
        with path.open(encoding="utf-8-sig") as source:
            for line in source:
                fields=line.rstrip("\r\n").split("\t")
                if len(fields)<3 or line.startswith("#"):continue
                lemma,form=map(nfc, fields[:2])
                if lexical(form) and lexical(lemma):
                    before=result.get(form)
                    if before is None or (lemma!=form,lemma)<(before!=form,before):result[form]=lemma
    return sorted(result.items())


def cedict(path):
    traditional={}; simplified=set()
    with gzip.open(path,"rt",encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip() or line.startswith("#"):continue
            x,y,*_=line.split()
            if not lexical(x) or not lexical(y):continue
            simplified.add(y)
            traditional[x]=y if x not in traditional or traditional[x]==y else None
    mapping={x:y for x,y in traditional.items() if y is not None}
    mapping.update({x:x for x in simplified})
    return sorted(mapping.items())


def russian(path,work):
    from pymorphy3.opencorpora_dict.wrapper import Dictionary
    root=work/"opencorpora"
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as wheel:
        for name in wheel.namelist():
            if name.startswith("mawo_pymorphy3/dicts_ru/") and not name.endswith("/"):
                (root/Path(name).name).write_bytes(wheel.read(name))
    dictionary=Dictionary(str(root))
    for form,group in itertools.groupby(dictionary.words.iteritems(),key=lambda p:p[0]):
        if not lexical(form):continue
        labels=Counter(nfc(dictionary.build_normal_form(idx,slot,form)) for _,(idx,slot) in group)
        candidates=[y for y in labels if lexical(y)]
        if candidates:
            f=nfc(form)
            yield f,min(candidates,key=lambda y:(-labels[y],y!=f,y))


def prepare(row, official, cache, assets, binary, compiled, work, projection_dir):
    iso=row["iso639_3"]
    target=work/(iso+".tsv")
    bank=row["source_home"].split("/")[4] if "/UniversalDependencies/UD_" in row["source_home"] else None
    details={}; seen=set(); examples={};count=0
    if bank:
        if (official/bank).is_dir():
            paths=next((p for s in ("train","dev","test") if (p:=sorted((official/bank).glob("*-ud-"+s+"*.conllu")))),[])
            if not paths:raise ValueError(f"No split for {bank}")
            details={"source_split":paths[0].name.split("-ud-")[1].split(".")[0],"revision":"r2.19",
                     "files":[{"name":p.name,"sha256":sha(p),"url":f"https://raw.githubusercontent.com/UniversalDependencies/{bank}/r2.19/{p.name}"} for p in paths]}
        else:
            if iso!="tuk":raise ValueError(f"Unexpected missing corpus {iso}")
            files=[f for f in row["source_files"] if f["url"].endswith(".conllu")]
            paths=[cache/f["sha256"] for f in files]
            if not all(sha(p)==f["sha256"] for p,f in zip(paths,files,strict=True)):raise ValueError("Turkmen input hash mismatch")
            details={"source_split":row["source_split"],"revision":row["revision"],"files":files}
        rows=corpus_rows(paths,iso);mode="official UD 2.19" if iso!="tuk" else "pinned Turkmen source"
        if iso=="ssp":rows=sorted({(x,y) for x,y,_ in rows})
    elif row["source"].startswith("UniMorph"):
        rows=paradigms(row,cache);mode="pinned UniMorph"
    elif iso in {"cmn","rus"}:
        path=cache/row["source_sha256"]
        if sha(path)!=row["source_sha256"]:raise ValueError("Wrong upstream package")
        rows=cedict(path) if iso=="cmn" else russian(path,work)
        mode="pinned CC-CEDICT" if iso=="cmn" else "pinned OpenCorpora"
    else:
        projected=projection_dir/(iso+".tsv.gz")
        if projected.is_file():
            import shutil
            with gzip.open(projected,"rb") as src,target.open("wb") as out:shutil.copyfileobj(src,out)
        else:
            gen=compiled/iso/".generations"/row["compiled_generation"]
            meta=json.loads((gen/"meta.json").read_bytes())
            if meta["training_tokens"] or meta["entries"]["rules"] or meta["entries"]["exact"]:
                raise ValueError(f"{iso} does not carry a pure external source projection")
            subprocess.run([binary,"unpack",gen,target],check=True)
        if sha(target)!=row["table_sha256"]:raise ValueError(f"Original source projection differs for {iso}")
        mode="hash-verified original lexical projection"
        with target.open() as stream:
            for line in stream:
                x,y=line.rstrip("\n").split("\t");count+=1
                if len(examples)<32 and x!=y:examples[x]=y
        return target,mode,{},count,count,examples
    with target.open("w",encoding="utf-8",newline="\n") as out:
        for fields in rows:
            x,y=fields[:2];out.write("\t".join(fields)+"\n");count+=1;seen.add(x)
            if len(examples)<32 and x!=y:examples[x]=y
    if not count:raise ValueError(f"Empty observations for {iso}")
    return target,mode,details,count,len(seen),examples

