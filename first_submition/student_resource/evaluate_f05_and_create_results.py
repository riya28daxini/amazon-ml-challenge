#!/usr/bin/env python3
import csv, os, sys
from collections import defaultdict
BASE=os.path.dirname(os.path.abspath(__file__))
TRAIN=os.path.join(BASE,'dataset','train')
S1=os.path.join(TRAIN,'train_source1.tsv'); S2=os.path.join(TRAIN,'train_source2.tsv'); S3=os.path.join(TRAIN,'train_source3.tsv'); GT=os.path.join(TRAIN,'train_ground_truth.tsv')
PRED=os.path.join(BASE,'predictions.tsv'); OUT=os.path.join(BASE,'matching_results.tsv'); DETAIL=os.path.join(BASE,'f05_entity_details.tsv'); SUMMARY=os.path.join(BASE,'f05_summary.txt')

def ids(path,col='entity_id'):
    out=set()
    with open(path,encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f,delimiter='\t')
        if col not in (r.fieldnames or []): raise ValueError(f'{path}: missing {col}; columns={r.fieldnames}')
        for x in r:
            v=str(x[col]).strip()
            if v: out.add(v)
    return out

def split(v):
    if v is None or not str(v).strip() or str(v).strip().lower() in {'nan','none','null'}: return set()
    return {x.strip() for x in str(v).split(',') if x.strip()}

def gt_load(path):
    d={}
    with open(path,encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f,delimiter='\t')
        for x in r: d[str(x['source1_entity_id']).strip()]=split(x.get('matched_entity_ids',''))
    return d

def pred_load(path):
    d=defaultdict(set)
    with open(path,encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f,delimiter='\t')
        for x in r: d[str(x['source1_entity_id']).strip()] |= split(x.get('matched_entity_ids',''))
    return d

def f05(t,p):
    tp=len(t&p); fp=len(p-t); fn=len(t-p)
    if not t and not p: return 1.0
    if tp==0: return 0.0
    P=tp/(tp+fp); R=tp/(tp+fn); b=.5
    return (1+b*b)*P*R/(b*b*P+R)

def main():
    for p in [S1,S2,S3,GT]:
        if not os.path.exists(p): raise FileNotFoundError(p)
    if not os.path.exists(PRED):
        print('ERROR: predictions.tsv not found.')
        print('Create your FINAL S1 -> S2/S3 prediction file first. candidate_pairs.tsv is not a final prediction.')
        sys.exit(1)
    s1=ids(S1); valid=ids(S2)|ids(S3); truth=gt_load(GT); pred=pred_load(PRED)
    total_tp=total_fp=total_fn=0; sm=0; exact=0; missing=0; invalid=0; details=[]
    with open(OUT,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n'); w.writerow(['source1_entity_id','matched_entity_ids'])
        for sid in s1:
            t=truth.get(sid,set()) & valid; raw=pred.get(sid,set()); invalid += len(raw-valid); p=raw & valid
            if sid not in pred: missing+=1
            tp=len(t&p); fp=len(p-t); fn=len(t-p); sc=f05(t,p)
            total_tp+=tp; total_fp+=fp; total_fn+=fn; sm+=sc
            exact += (t==p)
            ps=','.join(sorted(p)); ts=','.join(sorted(t)); w.writerow([sid,ps])
            details.append([sid,ts,ps,tp,fp,fn,sc])
    macro=sm/len(s1) if s1 else 0
    gp=total_tp/(total_tp+total_fp) if total_tp+total_fp else 1
    gr=total_tp/(total_tp+total_fn) if total_tp+total_fn else 1
    b=.5; gf=(1+b*b)*gp*gr/(b*b*gp+gr) if gp+gr else 0
    with open(DETAIL,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n'); w.writerow(['source1_entity_id','true_matched_entity_ids','predicted_matched_entity_ids','TP','FP','FN','F0.5']); w.writerows(details)
    summary=f'''AMAZON ML CHALLENGE 2026\nENTITY-LEVEL MACRO F0.5\n=========================\nS1 entities: {len(s1):,}\nS2 entities: {len(ids(S2)):,}\nS3 entities: {len(ids(S3)):,}\nTP: {total_tp:,}\nFP: {total_fp:,}\nFN: {total_fn:,}\nGlobal precision (diagnostic): {gp:.8f}\nGlobal recall (diagnostic): {gr:.8f}\nGlobal F0.5 (diagnostic): {gf:.8f}\nExact set matches: {exact:,}\nMissing prediction rows: {missing:,}\nInvalid predicted IDs removed: {invalid:,}\n\nMACRO F0.5: {macro:.8f}\n'''
    open(SUMMARY,'w',encoding='utf-8').write(summary)
    print('\n'+'='*60); print('RESULT'); print('='*60); print(f'S1 entities : {len(s1):,}'); print(f'TP           : {total_tp:,}'); print(f'FP           : {total_fp:,}'); print(f'FN           : {total_fn:,}'); print(f'Precision    : {gp:.6f}'); print(f'Recall       : {gr:.6f}'); print(f'Global F0.5  : {gf:.6f}'); print(f'MACRO F0.5   : {macro:.6f}'); print('='*60); print(f'Created: {OUT}'); print(f'Created: {DETAIL}'); print(f'Created: {SUMMARY}')
if __name__=='__main__': main()
