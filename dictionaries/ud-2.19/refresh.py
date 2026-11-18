"""Refresh the UD inputs in the 172-dictionary catalog with the canonical 1.0.0 runtime."""
import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
import zipfile
import subprocess
from concurrent.futures import ProcessPoolExecutor
from source_tables import prepare
from pathlib import Path

import udlex

REV="2.19"
VERSION="1.0.0.post1"
UD_HASH="6bbe2f0234dcc2d2367cc912848259c94c278ea69932d228d744f265a19a4a31"
CATALOG="57be9c16077c1a81ccb1f04d5fdac101364133f08f90e70069925e0cfea4804b"
ENGINE="2270bc1cb1c7144a85cf581318ca669417608fe4"
URL="https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/bd57416d-f1ec-43a0-8ebd-0a53cb9bda59/content"


def json_bytes(v):
    return (json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()


def digest(p):
    with p.open("rb") as fh: return hashlib.file_digest(fh,"sha256").hexdigest()


def licenses(bank, previous):
    readme=next(iter(sorted(bank.glob("README*"))))
    raw=readme.read_bytes()
    text=raw.decode()
    lic=next(x for x in sorted(bank.glob("LICENSE*")) if x.is_file())
    explicit=re.search(r"^License:\s*([^\r\n]+)",text,re.M).group(1).strip()
    field=explicit.replace(" ","-") if explicit!="PD" else "LicenseRef-Public-Domain"
    txt=lic.read_text()
    match=re.search(r"creativecommons\.org/licenses/([a-z-]+)/(\d\.\d)",txt,re.I)
    named="CC-"+match[1].upper()+"-"+match[2] if match else None
    conflict=bool(named and named!=field)
    expression="LicenseRef-UD-"+bank.name[3:].replace("_","-")+"-Review" if conflict else field
    contrib=re.search(r"^Contributors:\s*(.+)$",text,re.M)
    return {"license":expression,"readme_license":explicit,"license_file_expression":named,
            "requires_license_review":conflict,"noncommercial":("-NC-" in field or "-NC-" in (named or "")),
            "attribution":"Universal Dependencies Consortium and "+(contrib[1].strip() if contrib else previous["attribution"]),
            "readme":readme,"license_file":lic}


def write_package(out, folder, entry, docs):
    iso=entry["iso639_3"];name=entry["distribution"]; dist=name.replace("-","_"); info=f"{dist}-{VERSION}.dist-info"
    gen=entry["compiled_generation"]; root=folder/".generations"/gen
    runtime={f"{iso}/current":(gen+"\n").encode()}
    runtime.update({f"{iso}/.generations/{gen}/{f.name}":f.read_bytes() for f in root.iterdir()
                    if f.is_file() and (f.name=="meta.json" or not f.name.endswith(".tsv"))})
    manifest={"distribution":dist,"format_version":5,"language_aliases":{},
              "lexicons":[{"config":iso,"generation":gen,"iso639_3":iso}],"schema":"udlex-dictionary-wheel-manifest/v1"}
    runtime[f".manifests/udlex_dict_{iso}.json"]=json_bytes(manifest)
    provenance=json_bytes(entry)
    wheel_data={"udlex_bundle/"+k:v for k,v in runtime.items()}
    wheel_data.update({f"{info}/licenses/{k}":v for k,v in docs.items()})
    wheel_data[f"{info}/SOURCES.json"]=provenance
    wheel_data[f"{info}/METADATA"]=(f"Metadata-Version: 2.4\nName: {name}\nVersion: {VERSION}\n"
        f"Summary: {entry['language']} dictionary for udlex\nRequires-Python: >=3.11\n"
        f"Requires-Dist: udlex (>=1.0.0,<2)\nLicense-Expression: {entry['license']}\n"
        +"".join(f"License-File: {filename}\n" for filename in sorted(docs))+"\n").encode()
    wheel_data[f"{info}/WHEEL"]=b"Wheel-Version: 1.0\nGenerator: udlex-dictionary-refresh\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    rec=io.StringIO(newline="");csvout=csv.writer(rec,lineterminator="\n")
    for path,data in sorted(wheel_data.items()):
        value=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        csvout.writerow([path,"sha256="+value,str(len(data))])
    csvout.writerow([f"{info}/RECORD","",""]);wheel_data[f"{info}/RECORD"]=rec.getvalue().encode()
    wp=out/f"{dist}-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wp,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path,data in sorted(wheel_data.items()):
            zi=zipfile.ZipInfo(path,date_time=(2026,11,15,0,0,0));zi.external_attr=0o100644<<16
            z.writestr(zi,data,compress_type=zipfile.ZIP_DEFLATED)
    ap=out/f"{name}-{VERSION}-bundle.tar.gz"
    portable={"bundle/"+k:v for k,v in runtime.items()}
    portable.update({f"bundle/{iso}/LICENSES/{k}":v for k,v in docs.items()})
    portable[f"bundle/{iso}/SOURCES.json"]=provenance
    with ap.open("wb") as stream,gzip.GzipFile(fileobj=stream,filename="",mode="wb",mtime=0) as gz,tarfile.open(fileobj=gz,mode="w|") as t:
        for path,data in sorted(portable.items()):
            ti=tarfile.TarInfo(path);ti.size=len(data);ti.mode=0o644;t.addfile(ti,io.BytesIO(data))
    return {k:{"name":p.name,"size":p.stat().st_size,"sha256":digest(p)} for k,p in [("wheel",wp),("archive",ap)]}


def pin_official(archive, extracted):
    if digest(archive)!=UD_HASH:
        raise ValueError("Unexpected official UD 2.19 tarball")
    with tarfile.open(archive,"r|gz") as packed:
        for member in packed:
            name=Path(member.name)
            if not member.isfile() or name.parts[0]!="ud-treebanks-v2.19":continue
            if not (name.name.endswith(".conllu") or name.name.startswith(("LICENSE","README"))):continue
            with packed.extractfile(member) as src:
                if hashlib.file_digest(src,"sha256").hexdigest()!=digest(extracted/Path(*name.parts[1:])):
                    raise ValueError(f"Extracted UD file differs: {name}")


def build_one(job):
    row,args,recipe=job
    iso=row["iso639_3"];work=args.work/"lex-inputs";folder=args.work/"compiled"/iso
    receipt=args.work/"receipts"/(iso+".json");packages=args.output/"packages"
    if receipt.is_file():
        saved=json.loads(receipt.read_bytes())
        if saved["recipe_sha256"]==recipe and all(digest(packages/saved[k]["name"])==saved[k]["sha256"] for k in ("wheel","archive")):
            return saved
    source_receipt=work/(iso+".receipt.json")
    if iso=="rus" and source_receipt.exists():
        r=json.loads(source_receipt.read_bytes());tab=work/(iso+".tsv")
        if digest(tab)!=r["sha256"]:raise ValueError("Russian source preparation receipt mismatch")
        mode,details,count,forms,examples=(r[k] for k in ("mode","details","count","forms","examples"))
    else:
        tab,mode,details,count,forms,examples=prepare(row,args.ud,args.cache,args.assets,args.binary,args.base_compiled,work,args.projections)
    if folder.exists():shutil.rmtree(folder)
    folder.mkdir(parents=True)
    fitted=json.loads(subprocess.check_output([args.binary,"fit",iso,tab,folder],text=True))
    revision=details.get("revision",row.get("revision"))
    meta={"name":iso,"language":row["language"],"iso639_3":iso,"source":row["source"]+" "+str(revision),
          "training_tokens":fitted["training_tokens"],"variant":"full","selected_on":None,"max_suffix":6,
          "format_version":5,"entity_encoding":"namespaced-v1","entries":dict(fitted["entries"]),
          "traffic":{},"gold_tokens":None,"accuracy":None,"baseline":None,"beats_baseline":None,
          "token_case":"keep","normalization":"NFC+locale-v1"}
    (folder/"meta.json").write_bytes(json_bytes(meta))
    udlex.compile_bundle(str(folder));udlex.prune_bundle(str(folder))
    gen=(folder/"current").read_text().strip()
    lex=udlex.Lexicon(str(folder),use_user=False)
    # Examples are discovery smoke tests, separate from held-out accuracy.
    example_forms=list(dict.fromkeys([*row["examples"],*examples]))
    if not example_forms:
        with tab.open() as f:example_forms=[next(f).split("\t")[0]]
    pairs=[(x,lex.lemmatize([x])[0]) for x in example_forms]
    verified=dict(([(x,y) for x,y in pairs if x!=y]+pairs)[:5])
    bank=row["source_home"].split("/")[4] if "/UniversalDependencies/UD_" in row["source_home"] else None
    if bank and (args.ud/bank).is_dir():
        info=licenses(args.ud/bank,row);readme=info.pop("readme");license_file=info.pop("license_file")
        docs={"TREEBANK-README":readme.read_bytes(),"LICENSE":license_file.read_bytes()}
        source_files=details["files"]+[{"name":f.name,"sha256":digest(f),"url":URL} for f in (readme,license_file)]
        source_home="https://github.com/UniversalDependencies/"+bank+"/tree/r2.19"
    else:
        info={"license":row["license"],"attribution":row["attribution"],"noncommercial":row["noncommercial"],
              "requires_license_review":False,"license_url":row.get("license_url")}
        with zipfile.ZipFile(args.assets/row["wheel"]["name"]) as wheel:
            docs={n.split("/licenses/",1)[1]:wheel.read(n) for n in wheel.namelist() if "/licenses/" in n and not n.endswith("/")}
        source_files=details.get("files") or row.get("inputs") or [{"url":row["download"],"sha256":row.get("source_sha256")}]
        source_home=row["source_home"]
    dist=row["distribution"].removesuffix("-nc")+("-nc" if info["noncommercial"] else "")
    description=("Identity entries for attested sign glosses; no gold morphological lemmas available." if iso=="ssp" else
                 "Fresh locale-aware corpus lookup and suffix edits learned with udlex 1.0.0 from the recorded observations.")
    docs["NOTICE"]=(docs.get("NOTICE",b"")+("\n"+info["attribution"]+"\n"+source_home+"\n"+description+"\n").encode()).lstrip()
    entry={"iso639_3":iso,"language":row["language"],"distribution":dist,"dictionary_version":VERSION,
           "bundle_format":5,"engine_commit":ENGINE,"compiled_generation":gen,"source":row["source"],
           "source_home":source_home,"ud_repository":bank,"source_split":details.get("source_split",row.get("source_split","lexicon")),
           "revision":revision,"source_files":source_files,"download":source_files[0]["url"],
           "source_sha256":source_files[0].get("sha256"),"input_mode":mode,"input_rows":count,
           "unique_forms":forms,"training_tsv_sha256":digest(tab),"entries":meta["entries"],"training_tokens":meta["training_tokens"],
           "examples":verified,"changes":description,"recipe_sha256":recipe,**info}
    artifacts=write_package(packages,folder,entry,docs)
    saved=dict(entry,**artifacts)
    receipt.write_bytes(json_bytes(saved))
    if mode=="hash-verified original lexical projection":
        dest=args.output/"source-tables";dest.mkdir(exist_ok=True)
        with tab.open("rb") as inp,(dest/(iso+".tsv.gz")).open("wb") as stream,gzip.GzipFile(fileobj=stream,filename="",mtime=0,mode="wb") as gz:
            shutil.copyfileobj(inp,gz)
    print(iso,mode,"train",meta["training_tokens"],"rules",meta["entries"]["rules"],flush=True)
    return saved


def main(args):
    if udlex.__version__!="1.0.0" or digest(args.assets/"DICTIONARIES.json")!=CATALOG:raise ValueError("Wrong base catalog or engine")
    pin_official(args.archive,args.ud)
    rows=json.loads((args.assets/"DICTIONARIES.json").read_bytes())["dictionaries"]
    for d in (args.work/"lex-inputs",args.work/"receipts",args.output/"packages"):d.mkdir(parents=True,exist_ok=True)
    recipe=hashlib.sha256((digest(Path(__file__))+digest(Path(__file__).with_name("source_tables.py"))+digest(args.binary)+UD_HASH+CATALOG).encode()).hexdigest()
    jobs=[(r,args,recipe) for r in sorted(rows,key=lambda v:-v["form_lemma_rows"])]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        records=sorted(pool.map(build_one,jobs),key=lambda v:v["iso639_3"])
    if len({r["iso639_3"] for r in records})!=172:raise ValueError("Incomplete catalog")
    result={"engine_version":"1.0.0","dictionary_version":VERSION,"ud_revision":"2.19","base_catalog_sha256":CATALOG,
            "ud_archive_sha256":UD_HASH,"languages_packaged":len(records),"noncommercial_languages":[r["iso639_3"] for r in records if r["noncommercial"]],
            "language_codes":[r["iso639_3"] for r in records],"dictionaries":records}
    (args.output/"DICTIONARIES.json").write_bytes(json_bytes(result))
    snapshot={"version":"1.0.0","catalog_sha256":digest(args.output/"DICTIONARIES.json"),"ud_revision":"2.19",
              "languages":{v["iso639_3"]:{"name":v["language"],"ud_repository":v["ud_repository"],"generation":v["compiled_generation"]} for v in records}}
    (args.output/"snapshot.json").write_bytes(json_bytes(snapshot))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("assets","base_compiled","archive","ud","cache","binary","work","output"):parser.add_argument(key,type=Path)
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--projections",type=Path,default=Path(__file__).with_name("source-tables"))
    main(parser.parse_args())

