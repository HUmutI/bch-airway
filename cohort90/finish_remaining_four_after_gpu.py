#!/usr/bin/env python3
from pathlib import Path
import importlib.util, pandas as pd, sys
BASE=Path('/fileserver/external/quin/ege'); RUN=Path(__file__).resolve().parents[1]; OLD=BASE/'or_all3_run'

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
morph=load('morph_after_gpu',OLD/'scripts/09_airmorph_morphometry.py')
cases=[Path(x) for x in (RUN/'remaining_gpu_cases.txt').read_text().splitlines() if x]
outputs={
 'original':(OLD/'normal_airway_branch_morphometry.csv',OLD/'normal_airway_morphometry_status.csv',OLD/'normal_airway_branch_morphometry.xlsx'),
 'E3':(BASE/'E3_PEDIATRIC_AIRWAY_RESULTS/normal_airway_branch_morphometry.csv',BASE/'E3_PEDIATRIC_AIRWAY_RESULTS/normal_airway_morphometry_status.csv',BASE/'E3_PEDIATRIC_AIRWAY_RESULTS/normal_airway_branch_morphometry.xlsx')}
new={'original':{},'E3':{}}
for i,d in enumerate(cases,1):
 cid=d.name; cohort='E3' if cid.startswith('E3_') else 'original'
 print(f'[{i}/{len(cases)}] relabeling {cid}',flush=True)
 classification=morph.classify_case(d,force=True); rows=morph.measure_case(d)
 if not rows: raise RuntimeError(f'{cid}: no morphometry rows')
 new[cohort][cid]=(classification,rows)
 print(f'  {len(rows)} branches; max generation {max(int(r["generation"]) for r in rows)}',flush=True)
for cohort,updates in new.items():
 if not updates: continue
 bp,sp,xp=outputs[cohort]; branches=pd.read_csv(bp,low_memory=False);status=pd.read_csv(sp).fillna('')
 branches=pd.concat([branches[~branches.case_id.isin(updates)],*[pd.DataFrame(v[1]) for v in updates.values()]],ignore_index=True).sort_values(['case_id','branch_instance_id'])
 for cid,(classification,rows) in updates.items():
  idx=status.index[status.case_id.eq(cid)]
  if len(idx)!=1: raise RuntimeError(f'{cid}: status rows {len(idx)}')
  status.loc[idx,'classification']=classification;status.loc[idx,'branches']=len(rows)
  status.loc[idx,'proximal_or_unclassified_branches']=sum(r['branch_name']=='proximal_or_unclassified' for r in rows)
  status.loc[idx,'complexity_not_estimable_branches']=sum(r['complexity_status']!='ok' for r in rows)
  status.loc[idx,'geometry_flagged_branches']=sum(bool(r['geometry_qc_flags']) for r in rows);status.loc[idx,'error']=''
 morph.write_csv(bp,branches.where(pd.notna(branches),None).to_dict('records'))
 morph.write_csv(sp,status.where(pd.notna(status),None).to_dict('records'),list(status.columns))
 morph.write_excel(xp,branches.where(pd.notna(branches),None).to_dict('records'),status.where(pd.notna(status),None).to_dict('records'))
 print(f'updated {cohort} master exports',flush=True)
