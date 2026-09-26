#!/usr/bin/env python3
"""
Train a pairwise business-entity matcher and generate test predictions.

Run from this file's directory:
  python -m pip install pandas scikit-learn rapidfuzz
  python train_matcher.py

Uses only the challenge-provided TSV files. No external APIs.
Writes output/matching_results.tsv and output/model_threshold.txt.

Important: validate on a held-out split before submitting. Candidate blocking
limits recall; inspect the printed validation metrics and candidate recall.
"""
from pathlib import Path
import re, unicodedata, csv, math
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "dataset" / "train"
TEST = BASE / "dataset" / "test"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
FEATURES = ["name_ratio","name_token","name_partial","addr_ratio","addr_token",
            "country_equal","postal_equal","name_exact","address_exact",
            "name_prefix","name_token_overlap","address_present_both"]

def norm(v):
    if pd.isna(v): return ""
    s = unicodedata.normalize("NFKC", str(v)).lower().replace("&"," and ")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def canon_name(v):
    s = norm(v)
    replacements = [(r"\b(pvt|pvt\s+ltd)\b","private limited"),
                    (r"\bltd\b","limited"),(r"\bcorp\b","corporation"),
                    (r"\binc\b","incorporated"),(r"\bco\b","company")]
    for pat, rep in replacements: s = re.sub(pat, rep, s)
    return s

def postal(v):
    xs = re.findall(r"\b[a-z0-9]{4,10}\b", norm(v))
    return xs[-1] if xs else ""

def load(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"entity_id","business_name","business_address","country"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["entity_id"] = df.entity_id.astype(str)
    df["n"] = df.business_name.map(norm)
    df["cn"] = df.business_name.map(canon_name)
    df["a"] = df.business_address.map(norm)
    df["c"] = df.country.map(norm)
    df["p"] = df.business_address.map(postal)
    return df

def build_index(refs):
    idx = {}
    for r in refs.itertuples(index=False):
        for key in {("n",r.n),("cn",r.cn),("na",r.cn+"|"+r.a if r.a else ""),
                    ("cp",r.c+"|"+r.p if r.c and r.p else ""),
                    ("nt",r.c+"|"+r.n[:8] if r.c and len(r.n)>=8 else "")}:
            if key[1]: idx.setdefault(key, []).append(r.entity_id)
    return idx

def candidates(row, idx):
    keys = [("n",row.n),("cn",row.cn),
            ("na",row.cn+"|"+row.a if row.a else ""),
            ("cp",row.c+"|"+row.p if row.c and row.p else ""),
            ("nt",row.c+"|"+row.n[:8] if row.c and len(row.n)>=8 else "")]
    out=set()
    for k in keys:
        if k[1]: out.update(idx.get(k,()))
    return out

def features(a,b):
    n1,n2=a.n,b.n; ad1,ad2=a.a,b.a
    toks1=set(n1.split()); toks2=set(n2.split())
    overlap=len(toks1&toks2)/max(1,len(toks1|toks2))
    return [fuzz.ratio(n1,n2),fuzz.token_set_ratio(n1,n2),
            fuzz.partial_ratio(n1,n2),fuzz.ratio(ad1,ad2),
            fuzz.token_set_ratio(ad1,ad2),int(bool(a.c and a.c==b.c)),
            int(bool(a.p and a.p==b.p)),int(bool(n1 and n1==n2)),
            int(bool(ad1 and ad1==ad2)),int(bool(n1 and n2 and n1[:8]==n2[:8])),
            overlap,int(bool(ad1 and ad2))]

def make_pairs(s1, refs, idx, truth=None, limit=2000):
    refmap={r.entity_id:r for r in refs.itertuples(index=False)}
    rows=[]; total=0; positive_found=0; positive_total=0
    for r in s1.itertuples(index=False):
        cand=candidates(r,idx)
        if limit and len(cand)>limit:
            # retain deterministic subset; exact/name blocks are prioritized
            ranked=sorted(cand,key=lambda x:(refmap[x].cn != r.cn,
                          refmap[x].p != r.p, x))[:limit]
            cand=set(ranked)
        true = truth.get(r.entity_id,set()) if truth is not None else set()
        positive_total += len(true)
        positive_found += len(true & cand)
        for eid in cand:
            q=refmap.get(eid)
            if q is None: continue
            rows.append([r.entity_id,eid,*features(r,q),int(eid in true)])
        total+=1
        if total%50000==0: print(f"Candidate generation: {total:,}/{len(s1):,}")
    print(f"Candidate pairs: {len(rows):,}; blocked-in truth links: {positive_found:,}/{positive_total:,} ({positive_found/max(1,positive_total):.3%})")
    return pd.DataFrame(rows,columns=["s1","candidate",*FEATURES,"label"])

def load_truth(path):
    gt=pd.read_csv(path,sep="\t",dtype=str,keep_default_na=False)
    truth={}
    for r in gt.itertuples(index=False):
        ids=getattr(r,"matched_entity_ids","")
        truth[str(r.source1_entity_id)]={x.strip() for x in ids.split(",") if x.strip()}
    return truth

def macro_f05(df, probs, threshold, source1_ids, truth_map):
    pred={}
    for sid,eid,p in zip(df.s1,df.candidate,probs):
        if p>=threshold: pred.setdefault(sid,set()).add(eid)
    scores=[]
    for sid in source1_ids:
        t=truth_map.get(sid,set()); p=pred.get(sid,set())
        tp=len(t&p); fp=len(p-t); fn=len(t-p)
        denom=1.25*tp+0.25*fn+fp
        scores.append(1.0 if denom==0 else 1.25*tp/denom)
    return sum(scores)/max(1,len(scores))

def main():
    s1=load(TRAIN/"train_source1.tsv"); s2=load(TRAIN/"train_source2.tsv")
    s3=load(TRAIN/"train_source3.tsv"); truth=load_truth(TRAIN/"train_ground_truth.tsv")
    refs=pd.concat([s2,s3],ignore_index=True)
    idx=build_index(refs)
    # Split by Source-1 entity to avoid candidate-level leakage.
    train_ids,val_ids=train_test_split(s1.entity_id.to_numpy(),test_size=0.2,random_state=42)
    train_s1=s1[s1.entity_id.isin(set(train_ids))]
    val_s1=s1[s1.entity_id.isin(set(val_ids))]
    print("Building training candidates...")
    tr=make_pairs(train_s1,refs,idx,truth)
    print("Building validation candidates...")
    va=make_pairs(val_s1,refs,idx,truth)
    if tr.empty or tr.label.nunique()<2: raise RuntimeError("Not enough positive/negative candidate pairs; improve blocking.")
    model=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.08,max_leaf_nodes=31,
                                         l2_regularization=1.0,random_state=42)
    model.fit(tr[FEATURES],tr.label)
    probs=model.predict_proba(va[FEATURES])[:,1]
    best=(float("-inf"),0.5)
    for threshold in [x/100 for x in range(20,96,2)]:
        score=macro_f05(va,probs,threshold,val_s1.entity_id,truth)
        if score>best[0]: best=(score,threshold)
    print(f"Validation macro F0.5: {best[0]:.6f} at threshold {best[1]:.2f}")
    (OUT/"model_threshold.txt").write_text(f"{best[1]:.2f}\n",encoding="utf-8")
    # Refit using all labeled training Source-1 entities.
    print("Refitting on all training candidates...")
    allpairs=make_pairs(s1,refs,idx,truth)
    model.fit(allpairs[FEATURES],allpairs.label)
    import joblib
    joblib.dump({"model":model,"features":FEATURES,"threshold":best[1]},OUT/"entity_matcher.joblib")
    # Test prediction generation.
    t1=load(TEST/"test_source1.tsv"); t2=load(TEST/"test_source2.tsv")
    t3=load(TEST/"test_source3.tsv"); trefs=pd.concat([t2,t3],ignore_index=True)
    tidx=build_index(trefs); tmap={r.entity_id:r for r in trefs.itertuples(index=False)}
    outpath=OUT/"matching_results.tsv"
    with outpath.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f,delimiter="\t",lineterminator="\n")
        w.writerow(["source1_entity_id","matched_entity_ids"])
        for i,r in enumerate(t1.itertuples(index=False),1):
            ids=sorted(candidates(r,tidx))
            if ids:
                X=pd.DataFrame([features(r,tmap[e]) for e in ids],columns=FEATURES)
                ps=model.predict_proba(X)[:,1]
                matches=sorted(e for e,p in zip(ids,ps) if p>=best[1])
            else: matches=[]
            w.writerow([r.entity_id,",".join(matches)])
            if i%50000==0: print(f"Test prediction: {i:,}/{len(t1):,}")
    print(f"Saved submission: {outpath}")
    print("Run the official validator before submitting.")

if __name__=="__main__": main()
