"""Evaluate a freshly compiled catalog on the official 2.19 test files with paired predictions."""
import argparse
import hashlib
import re
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import udlex

from evaluate import score_language
from order_inputs import RELEASES
from verify_snapshot import EVIDENCE, read, save, sha


def main(args):
    archive_hash, n_test, _ = RELEASES["2.19"]
    catalog=read(args.catalog/"DICTIONARIES.json")
    snapshot=read(args.catalog/"snapshot.json")
    if sha(args.archive)!=archive_hash or catalog["ud_archive_sha256"]!=archive_hash:
        raise ValueError("Unexpected UD source")
    if udlex.__version__!="1.0.0" or snapshot["catalog_sha256"]!=sha(args.catalog/"DICTIONARIES.json"):
        raise ValueError("Catalog and engine differ from the fresh fit")
    mapping=read(EVIDENCE/"language-codes.json")
    rows={v["iso639_3"]:v for v in catalog["dictionaries"]}
    if len(rows)!=172 or set(rows)!=set(mapping):raise ValueError("Roster mismatch")
    by_code=defaultdict(list); expected={}
    with tarfile.open(args.archive,"r|gz") as packed:
        for ent in packed:
            if ent.isfile() and ent.name.endswith("-ud-test.conllu"):
                with packed.extractfile(ent) as f:
                    expected[str(Path(ent.name).relative_to("ud-treebanks-v2.19"))]=hashlib.file_digest(f,"sha256").hexdigest()
    paths=sorted(args.ud.glob("UD_*/*-ud-test.conllu"))
    if len(paths)!=n_test or {str(v.relative_to(args.ud)):sha(v) for v in paths}!=expected:
        raise ValueError("Test file inventory differs")
    for path in paths:by_code[re.split("[_-]",path.name,maxsplit=1)[0]].append(path)
    args.output.mkdir(exist_ok=True,parents=True)
    jobs=[]
    for iso, code in sorted(mapping.items()):
        if (args.compiled/iso/"current").read_text().strip()!=rows[iso]["compiled_generation"]:
            raise ValueError(f"Wrong generation for {iso}")
        jobs.append((iso,rows[iso],snapshot["languages"][iso],by_code[code],args.ud,args.compiled,args.output))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i,_ in enumerate(pool.map(score_language,jobs),1):
            if i%20==0 or i==len(jobs):print(i,flush=True)
    save(args.output/"run.json",{"version":"1.0.0","data_version":catalog["dictionary_version"],
        "catalog_sha256":snapshot["catalog_sha256"],"ud219_sha256":archive_hash,
        "snapshot_sha256":sha(args.catalog/"snapshot.json"),"language_codes_sha256":sha(EVIDENCE/"language-codes.json"),
        "protocol_sha256":sha(Path(__file__).with_name("catalog-219-protocol.md")),"program_sha256":sha(Path(__file__)),
        "scoring_program_sha256":sha(Path(__file__).with_name("evaluate.py")),"release_test_files":n_test,"languages":sorted(rows)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("catalog","compiled","archive","ud","output"):parser.add_argument(name,type=Path)
    parser.add_argument("--jobs",type=int,default=6)
    main(parser.parse_args())

