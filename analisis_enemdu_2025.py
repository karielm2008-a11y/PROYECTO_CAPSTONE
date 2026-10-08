#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Análisis completo ENEMDU anual 2025: logística frente a Random Forest.

Python 3.12.14; instalar: python -m pip install -r requirements.txt
Ejecutar:
  python analisis_enemdu_2025.py --data "2_BDD_DATOS_ABIERTOS_ENEMDU_2025_CSV.zip" --output resultados_enemdu
También acepta directamente BDDenemdu_personas_2025_anual.csv.
No requiere el repositorio, notebooks, modelos ni resultados anteriores.

Procedencia: consolidación del código que produjo la tesis, sin modificar la
selección de variables, particiones, candidatos, semillas ni estimadores.
Los valores de referencia SOLO se usan al final para comparar, nunca para ajustar.

Advertencias metodológicas:
- Grupo = posición de vivienda del panel (upm + panelm + vivienda), no identidad
  longitudinal confirmada. Se conservan cadenas y ceros iniciales.
- Los nueve predictores originales no tienen faltantes en el universo binario.
  Si otros datos los tienen, se detiene; no se inventa una imputación nueva.
- Fexp se normaliza a media 1 SOLO al ajustar cada subconjunto. Las métricas
  usan fexp original. El escalador de edad no se pondera (protocolo original).
- PR-AUC se reporta como average precision, no como integral trapezoidal.
- La calibración es diagnóstica, sin recalibración posterior.
- Umbral exploratorio = máximo F1 OOF; no representa una decisión institucional.
- Bootstrap: 300 remuestreos pareados de viviendas de prueba, semilla 2026,
  modelos fijos. No incluye reentrenamiento ni estima la varianza del diseño INEC.
- V de Cramér ponderada descriptiva, sin corrección ni inferencia de diseño.
- Coeficientes/contrastes e importancias son asociaciones, no efectos causales.
- Se generan datos individuales y modelos SOLO en la carpeta local de salida;
  no deben publicarse automáticamente. Las figuras y tablas son agregadas.

Salidas: outputs/tables, outputs/figures (las 12 figuras de la tesis y una
matriz ponderada adicional), outputs/audit, docs/figures.json y verificaciones.
Se preserva el paso CSV intermedio del procedimiento original para reproducir
los mismos tipos y precisión numérica. No se cargan predicciones previas.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import tempfile
import zipfile
import shutil
import importlib.metadata
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.exceptions import ConvergenceWarning


# 1. Lectura, limpieza y auditoría del universo.

"""Auditoría ejecutable. No estima modelos ni exporta microdatos individuales."""
from pathlib import Path
import argparse, hashlib, json, platform
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def read_data(path):
    d = pd.read_csv(path, sep=';', encoding='utf-8-sig', dtype='string', low_memory=False)
    for c in d:
        d[c] = d[c].str.strip().replace('', pd.NA)
    for c in ['p03','secemp','empleo','fexp']:
        if c in d:
            d[c] = pd.to_numeric(d[c].str.replace(',', '.', regex=False), errors='raise')
    return d

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def run(persons, housing=None, out=Path('outputs/audit')):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    d = read_data(persons)
    assert d.id_persona.notna().all() and not d.id_persona.duplicated().any()
    assert d.fexp.notna().all() and d.fexp.gt(0).all()
    employed = d.p03.ge(15) & d.empleo.eq(1)
    a = d.loc[employed & d.secemp.isin([1,2])].copy()
    # La clave agrupa posiciones del panel; no prueba identidad de personas.
    stable = d.upm + d.panelm + d.vivienda + d.hogar
    assert (d.id_hogar.str[-2:] == d.mes).all()
    assert (d.id_hogar.str[:-2] == stable).all()
    a['group_slot'] = a.id_hogar.str[:-2]
    s = dict(persons_rows=len(d), persons_columns=len(d.columns),
             employed_rows=int(employed.sum()), analytic_rows=len(a),
             formal_rows=int(a.secemp.eq(1).sum()), informal_rows=int(a.secemp.eq(2).sum()),
             domestic_rows=int((employed & d.secemp.eq(3)).sum()),
             unclassified_rows=int((employed & d.secemp.eq(4)).sum()),
             weighted_employed=float(d.loc[employed,'fexp'].sum()),
             weighted_analytic=float(a.fexp.sum()),
             weighted_informal=float(a.loc[a.secemp.eq(2),'fexp'].sum()),
             unweighted_informal_pct=float(a.secemp.eq(2).mean()*100),
             occupied_definition_mismatches=int((employed != d.secemp.isin([1,2,3,4])).sum()),
             household_visit_ids=int(d.id_hogar.nunique()),
             household_panel_slots=int(stable.nunique()),
             multi_month_household_slots=int(pd.DataFrame({'g':stable,'m':d.mes}).groupby('g').m.nunique().gt(1).sum()),
             analytic_household_slots=int(a.group_slot.nunique()),
             months=int(d.mes.nunique()), age_min=int(a.p03.min()),age_max=int(a.p03.max()),
             persons_sha256=sha256(persons), python=platform.python_version())
    s['official_informal_pct']=100*s['weighted_informal']/s['weighted_employed']
    s['analytic_informal_pct']=100*s['weighted_informal']/s['weighted_analytic']
    if housing:
        h=read_data(housing)
        assert h.id_hogar.is_unique
        s.update(housing_rows=len(h),housing_columns=len(h.columns),housing_sha256=sha256(housing),
                 housing_duplicate_dwelling_ids=int(h.id_vivienda.duplicated().sum()),
                 unmatched_household_visits=int((~d.id_hogar.isin(h.id_hogar)).sum()))
    split=[]
    for group in ['id_hogar','group_slot']:
        for seed in [42,2025,2026]:
            tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(a,groups=a[group]))
            overlap=set(a.iloc[tr].group_slot)&set(a.iloc[te].group_slot)
            affected=int(a.iloc[te].group_slot.isin(overlap).sum())
            if group=='group_slot': assert not overlap
            split.append(dict(group=group,seed=seed,train_rows=len(tr),test_rows=len(te),
                              overlapping_household_slots=len(overlap),affected_test_rows=affected,
                              affected_test_pct=100*affected/len(te)))
    pd.DataFrame(split).to_csv(out/'split_sensitivity.csv',index=False)
    candidates=['p02','p03','p04','p06','p15','nnivins','area','prov','p42','rama1','grupo1','fexp']
    pd.DataFrame([dict(variable=c,missing=int(a[c].isna().sum()),levels=int(a[c].nunique())) for c in candidates]).to_csv(out/'candidate_quality.csv',index=False)
    (out/'audit_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(s,ensure_ascii=False,indent=2))
    return s


# 2. Partición por vivienda y cinco folds agrupados.

"""Base analítica y partición por vivienda del panel; salidas individuales locales."""
from pathlib import Path
import argparse,json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit,GroupKFold


def prepare(persons,out=Path('data/processed'),tables=Path('outputs/tables')):
    out=Path(out);tables=Path(tables);out.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    d=read_data(persons)
    a=d.loc[d.p03.ge(15)&d.empleo.eq(1)&d.secemp.isin([1,2])].copy()
    a['target']=a.secemp.eq(2).astype('int8')
    a['group_household']=a.upm+a.panelm+a.vivienda+a.hogar
    a['group_dwelling']=a.upm+a.panelm+a.vivienda
    assert a.id_persona.is_unique
    assert a.group_dwelling.notna().all()
    assert a.fexp.notna().all() and a.fexp.gt(0).all()
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=2025).split(a,groups=a.group_dwelling))
    a['split']='train';a.iloc[te,a.columns.get_loc('split')]='test'
    for key in ['group_dwelling','group_household']:
        assert not set(a.iloc[tr][key])&set(a.iloc[te][key])
    a['fold']=-1
    train=a.iloc[tr]
    cv=GroupKFold(n_splits=5,shuffle=True,random_state=2025)
    for k,(fit,valid) in enumerate(cv.split(train,groups=train.group_dwelling)):
        assert not set(train.iloc[fit].group_dwelling)&set(train.iloc[valid].group_dwelling)
        a.loc[train.iloc[valid].index,'fold']=k
    cols=['id_persona','group_household','group_dwelling','upm','estrato','mes','periodo','target','fexp','p02','p03','p06','p15','nnivins','area','prov','rama1','grupo1','p42','split','fold']
    a[cols].to_csv(out/'analytic.csv.gz',index=False,compression='gzip')
    rows=[]
    for name,g in a.groupby('split'):
        rows.append({'split':name,'n':len(g),'dwelling_groups':g.group_dwelling.nunique(),'household_groups':g.group_household.nunique(),'weighted_population':float(g.fexp.sum()),'informal_pct_unweighted':float(g.target.mean()*100),'informal_pct_weighted':float((g.target*g.fexp).sum()/g.fexp.sum()*100)})
    pd.DataFrame(rows).to_csv(tables/'split_summary.csv',index=False)
    q=[]
    for key in ['area','p02','nnivins','p42','prov']:
        for val,g in a.groupby(key):
            q.append(dict(variable=key,category=val,n=len(g),expanded=float(g.fexp.sum()),informal_expanded=float((g.target*g.fexp).sum()),informal_pct=float((g.target*g.fexp).sum()/g.fexp.sum()*100)))
    pd.DataFrame(q).to_csv(tables/'descriptives.csv',index=False)
    info={'seed':2025,'group':'upm + panelm + vivienda','group_interpretation':'Posición de vivienda del panel; no identidad longitudinal confirmada','test_group_fraction':.2,'folds_train':5,'target':{'0':'secemp=1 formal','1':'secemp=2 informal'},'n':len(a),'input_sha256':sha256(persons),'split_records':rows,'trained_models':False}
    (tables/'preparation.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print(json.dumps(info,ensure_ascii=False,indent=2));return info

# 3. CV, selección por Brier OOF, ajuste final, métricas y permutación.



def entrenar(ROOT):
    """Validación interna por grupos; prueba reservada. Ejecutar desde la raíz."""
    from pathlib import Path
    import json,time,warnings
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT=ROOT/'outputs/tables';OUT.mkdir(exist_ok=True,parents=True)
    df=pd.read_csv(ROOT/'data/processed/analytic.csv.gz',dtype=str)
    for c in ['p03','target','fexp','fold']:df[c]=pd.to_numeric(df[c])
    train=df[df.split=='train'].reset_index(drop=True);test=df[df.split=='test'].reset_index(drop=True)
    assert not set(train.group_dwelling) & set(test.group_dwelling)
    assert train.target.nunique()==test.target.nunique()==2
    CATS=['p02','p06','p15','nnivins','area','prov','rama1','grupo1']
    FEATS=CATS+['p03']
    assert df[FEATS].notna().all().all()
    assert df.fexp.gt(0).all()
    def model(kind,param):
     prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),CATS),('age',StandardScaler(),['p03'])])
     estimator=LogisticRegression(C=param,max_iter=1500,solver='lbfgs') if kind=='Logística' else RandomForestClassifier(n_estimators=150,max_depth=20,min_samples_leaf=param,max_features='sqrt',n_jobs=2,random_state=2025)
     return make_pipeline(prep,estimator)
    def fit(m,d):
     w=d.fexp.to_numpy();m.fit(d[FEATS],d.target,**{m.steps[-1][0]+'__sample_weight':w/w.mean()});return m

    def metrics(d,p,threshold=.5,weighted=True):
     w=d.fexp if weighted else None;y=d.target
     tn,fp,fn,tp=confusion_matrix(y,p>=threshold,sample_weight=w,labels=[0,1]).ravel()
     return dict(roc_auc=roc_auc_score(y,p,sample_weight=w),pr_auc_ap=average_precision_score(y,p,sample_weight=w),brier=brier_score_loss(y,p,sample_weight=w),accuracy=(tp+tn)/(tp+tn+fp+fn),precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),specificity=tn/(tn+fp),f1=2*tp/(2*tp+fp+fn),tn=tn,fp=fp,fn=fn,tp=tp)
    cv=[];results=[];groups=[];cal=[];thresholds=[];selected={};preds={}
    for kind,params in [('Logística',[.1,1.]),('Random Forest',[20,50])]:
     candidates=[]
     for param in params:
      oof=np.zeros(len(train))
      for fold in range(5):
       a=train[train.fold!=fold];idx=train.fold==fold;b=train[idx]
       m=fit(model(kind,param),a);p=m.predict_proba(b[FEATS])[:,1];oof[idx]=p
       cv.append(dict(model=kind,param=param,fold=fold,**metrics(b,p)))
       print(kind,param,'fold',fold,flush=True)
      score=metrics(train,oof)['brier'];candidates.append((score,param,oof))
     score,param,oof=min(candidates,key=lambda v:v[0]);selected[kind]={'param':param,'oof_brier':score}
     grid=[]
     for t in np.arange(.1,.901,.01):
      row=dict(model=kind,threshold=float(t),**metrics(train,oof,float(t)));grid.append(row);thresholds.append(row)
     threshold=max(grid,key=lambda v:v['f1'])['threshold'];selected[kind]['threshold_f1_oof']=threshold
     m=fit(model(kind,param),train);p=m.predict_proba(test[FEATS])[:,1];preds[kind]=p
     for weighted in [True,False]:
      for t in [.5,threshold]:results.append(dict(model=kind,weighted=weighted,threshold=t,**metrics(test,p,t,weighted)))
     for c in ['area','p02','prov','nnivins']:
      for value,part in test.groupby(c):
       if part.target.nunique()<2:continue
       groups.append(dict(model=kind,group=c,value=value,n=len(part),**metrics(part,p[part.index])))
     for k in range(10):
      mask=(p>=k/10)&(p<(k+1)/10 if k<9 else p<=1)
      if mask.any():cal.append(dict(model=kind,bin=k,n=int(mask.sum()),predicted=np.average(p[mask],weights=test.fexp[mask]),observed=np.average(test.target[mask],weights=test.fexp[mask])))
     if kind=='Logística':
      names=m[0].get_feature_names_out();coef=m[-1].coef_[0]
      pd.DataFrame({'feature':names,'coefficient':coef,'exp_coefficient':np.exp(coef)}).to_csv(OUT/'logistic_coefficients.csv',index=False)
     # Permutación marginal en prueba; tres repeticiones, reducción de ROC-AUC ponderado.
     rng=np.random.default_rng(2025);imp=[];base=roc_auc_score(test.target,p,sample_weight=test.fexp)
     for c in FEATS:
      changes=[]
      for r in range(3):
       x=test[FEATS].copy();x[c]=rng.permutation(x[c].values)
       changes.append(base-roc_auc_score(test.target,m.predict_proba(x)[:,1],sample_weight=test.fexp))
      imp.append({'feature':c,'auc_drop_mean':np.mean(changes),'auc_drop_sd':np.std(changes)})
     pd.DataFrame(imp).sort_values('auc_drop_mean',ascending=False).to_csv(OUT/('importance_logistic.csv' if kind=='Logística' else 'importance_rf.csv'),index=False)
     print('FINAL',kind,selected[kind],metrics(test,p),flush=True)
    for name,data in [('cv_metrics',cv),('test_metrics',results),('group_metrics',groups),('calibration',cal),('thresholds_oof',thresholds)]:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'model_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for kind,part in pd.DataFrame(cal).groupby('model'):ax.plot(part.predicted,part.observed,'o-',label=kind)
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Probabilidad estimada media',ylabel='Proporción observada ponderada',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout()
    (ROOT/'outputs/figures').mkdir(exist_ok=True);fig.savefig(ROOT/'outputs/figures/calibration.png',dpi=180);plt.close(fig)
    (OUT/'training_protocol.json').write_text(json.dumps({'seed':2025,'features':FEATS,'cv_folds':5,'selection':'menor Brier OOF ponderado','rf_trees':150,'rf_max_depth':20,'training_weights':'fexp/media del subconjunto','calibration':'diagnóstico, sin recalibración','threshold':'máximo F1 OOF; exploratorio, no umbral institucional','pr_auc':'average precision; no integral trapezoidal','sensitivities':'contextual y ampliado pendientes','uncertainty':'intervalos pendientes'},ensure_ascii=False,indent=2))


# 4. Contrastes de coeficientes regularizados y referencia de prevalencia.



def contrastes(ROOT):
    """Contrastes de logística regularizada a partir de coeficientes ejecutados."""
    from pathlib import Path
    import pandas as pd
    import numpy as np
    c=pd.read_csv(ROOT/'outputs/tables/logistic_coefficients.csv');rows=[]
    for variable in ['p02','p06','p15','nnivins','area','prov','rama1','grupo1']:
     part=c[c.feature.str.startswith('cat__'+variable+'_')]
     base=part.iloc[0]
     for _,r in part.iterrows():rows.append({'variable':variable,'category':r.feature.split('_')[-1],'reference':base.feature.split('_')[-1],'log_odds_contrast':r.coefficient-base.coefficient,'odds_ratio':np.exp(r.coefficient-base.coefficient),'interpretation':'Asociación condicional regularizada; sin IC de diseño'})
    pd.DataFrame(rows).to_csv(ROOT/'outputs/tables/logistic_odds_ratios.csv',index=False)
    print('Contrastes frente a la primera categoría de cada predictor exportados.')
    s=pd.read_csv(ROOT/'outputs/tables/split_summary.csv').set_index('split');p=s.loc['train','informal_pct_weighted']/100;q=s.loc['test','informal_pct_weighted']/100
    pd.DataFrame([{'baseline':'prevalencia ponderada de entrenamiento','probability':p,'test_brier':q*(1-p)**2+(1-q)*p**2}]).to_csv(ROOT/'outputs/tables/baseline.csv',index=False)


# 5. Reajuste de comprobación, bootstrap pareado, descripción y Cramér.



def ampliar_evidencia(ROOT):
    """Amplía la evidencia sin volver a seleccionar modelos ni umbrales con prueba."""
    from pathlib import Path
    import json, ast
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score,roc_curve,precision_recall_curve
    import joblib
    OUT=ROOT/'outputs/tables'
    d=pd.read_csv(ROOT/'data/processed/analytic.csv.gz',dtype=str)
    for c in ['p03','target','fexp','fold']: d[c]=pd.to_numeric(d[c])
    a=d[d.split=='train'].reset_index(drop=True);b=d[d.split=='test'].reset_index(drop=True)
    CATS=['p02','p06','p15','nnivins','area','prov','rama1','grupo1'];FEATS=CATS+['p03']
    sel=json.loads((OUT/'model_selection.json').read_text());prior=pd.read_csv(OUT/'test_metrics.csv')
    pred={};curves=[];checks=[]
    for name in ['Logística','Random Forest']:
     param=sel[name]['param']
     est=LogisticRegression(C=param,max_iter=1500,solver='lbfgs') if name=='Logística' else RandomForestClassifier(n_estimators=150,max_depth=20,min_samples_leaf=int(param),max_features='sqrt',n_jobs=2,random_state=2025)
     prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),CATS),('age',StandardScaler(),['p03'])]);m=make_pipeline(prep,est)
     m.fit(a[FEATS],a.target,**{m.steps[-1][0]+'__sample_weight':a.fexp/a.fexp.mean()})
     p=m.predict_proba(b[FEATS])[:,1];pred[name]=p
     scores={'roc_auc':roc_auc_score(b.target,p,sample_weight=b.fexp),'pr_auc_ap':average_precision_score(b.target,p,sample_weight=b.fexp),'brier':brier_score_loss(b.target,p,sample_weight=b.fexp)}
     old=prior[(prior.model==name)&prior.weighted&(prior.threshold==.5)].iloc[0]
     for k,v in scores.items():assert np.isclose(v,old[k],rtol=1e-10,atol=1e-10);checks.append({'model':name,'metric':k,'previous':old[k],'recomputed':v})
     fpr,tpr,_=roc_curve(b.target,p,sample_weight=b.fexp);precision,recall,_=precision_recall_curve(b.target,p,sample_weight=b.fexp)
     grid=np.linspace(0,1,501)
     for x,y in zip(grid,np.interp(grid,fpr,tpr)):curves.append({'model':name,'curve':'ROC','x':x,'y':y})
     for x,y in zip(grid,np.interp(grid,recall[::-1],precision[::-1])):curves.append({'model':name,'curve':'PR','x':x,'y':y})
     (ROOT/'models').mkdir(exist_ok=True);joblib.dump(m,ROOT/'models'/('logistic.joblib' if name=='Logística' else 'rf.joblib'))
     print('Reproducción conforme:',name,scores,flush=True)
    pd.DataFrame(checks).to_csv(OUT/'reproduction_check.csv',index=False);pd.DataFrame(curves).to_csv(OUT/'curves.csv',index=False)
    local=b[['group_dwelling','target','fexp']].copy()
    for name,p in pred.items():local[name]=p
    local.to_csv(ROOT/'data/processed/test_predictions.csv.gz',index=False,compression='gzip')
    # Bootstrap pareado de grupos, modelos fijos. No es varianza del diseño INEC.
    codes,uniques=pd.factorize(b.group_dwelling);rng=np.random.default_rng(2026);boots=[]
    for i in range(300):
     count=np.bincount(rng.integers(0,len(uniques),len(uniques)),minlength=len(uniques));w=b.fexp.to_numpy()*count[codes]
     for name,p in pred.items():boots.append({'replicate':i,'model':name,'roc_auc':roc_auc_score(b.target,p,sample_weight=w),'brier':brier_score_loss(b.target,p,sample_weight=w)})
    z=pd.DataFrame(boots);z.to_csv(OUT/'bootstrap_replicates.csv',index=False);cis=[]
    for metric in ['roc_auc','brier']:
     pivot=z.pivot(index='replicate',columns='model',values=metric)
     for name in pred:
      lo,hi=np.quantile(pivot[name],[.025,.975]);cis.append({'contrast':name,'metric':metric,'lower':lo,'upper':hi})
     diff=pivot['Random Forest']-pivot['Logística'];lo,hi=np.quantile(diff,[.025,.975]);cis.append({'contrast':'RF menos Logística','metric':metric,'lower':lo,'upper':hi})
    pd.DataFrame(cis).to_csv(OUT/'bootstrap_intervals.csv',index=False)
    # Descripción completa y asociaciones de códigos nominales.
    d['age_group']=pd.cut(d.p03,[14,24,34,44,54,64,200],labels=['15–24','25–34','35–44','45–54','55–64','65 o más'])
    rows=[]
    for key in ['area','p02','nnivins','prov','age_group','rama1','grupo1','p42']:
     for val,g in d.groupby(key,observed=True):rows.append({'variable':key,'category':str(val),'n':len(g),'expanded':g.fexp.sum(),'informal_expanded':(g.fexp*g.target).sum(),'informal_pct':np.average(g.target,weights=g.fexp)*100})
    pd.DataFrame(rows).to_csv(OUT/'descriptives_extended.csv',index=False)
    assocvars=['target','p02','age_group','p06','p15','nnivins','area','prov','rama1','grupo1'];v=pd.DataFrame(index=assocvars,columns=assocvars,dtype=float)
    for x in assocvars:
     for y in assocvars:
      if x==y:v.loc[x,y]=1;continue
      tab=pd.crosstab(d[x],d[y],values=d.fexp,aggfunc='sum').fillna(0).to_numpy();n=tab.sum();e=np.outer(tab.sum(1),tab.sum(0))/n
      chi=np.divide((tab-e)**2,e,out=np.zeros_like(e),where=e>0).sum();v.loc[x,y]=np.sqrt(chi/(n*(min(tab.shape)-1)))
    v.to_csv(OUT/'associations_cramers_v.csv')
    print(pd.DataFrame(cis).to_string(index=False),flush=True)


# 6. Todas las figuras originales de la tesis.



def generar_figuras(ROOT):
    """Figuras del Word a partir de tablas verificables, sin datos ilustrativos."""
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T=ROOT/'outputs/tables';F=ROOT/'outputs/figures';F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    COL={'Logística':'#3A637D','Random Forest':'#981C3E'}
    PROV=dict(enumerate(['Azuay','Bolívar','Cañar','Carchi','Cotopaxi','Chimborazo','El Oro','Esmeraldas','Guayas','Imbabura','Loja','Los Ríos','Manabí','Morona Santiago','Napo','Pastaza','Pichincha','Tungurahua','Zamora Chinchipe','Galápagos','Sucumbíos','Orellana','Santo Domingo','Santa Elena'],1))
    EDU={1:'Ninguno',2:'Alfabetización',3:'Educación básica',4:'Bachillerato',5:'Educación superior'}
    LABEL={'target':'Sector informal','p02':'Sexo','age_group':'Grupo de edad','p03':'Edad','p06':'Estado civil','p15':'Etnia','nnivins':'Instrucción','area':'Área','prov':'Provincia','rama1':'Rama de actividad','grupo1':'Grupo de ocupación'}
    def save(fig,name):fig.tight_layout();fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)
    def bars(ax,labels,values,title):
     ax.barh(labels,values,color='#981C3E',height=.6);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_title(title,fontsize=11);ax.set_xlabel('Informalidad ponderada (%)')
     for i,v in enumerate(values):ax.text(v+1,i,f'{v:.1f}',va='center',fontsize=9)
    d=pd.read_csv(T/'descriptives_extended.csv',dtype={'category':str})
    fig,axes=plt.subplots(1,2,figsize=(8,3.1),gridspec_kw={'width_ratios':[.8,1.3]})
    a=d[d.variable=='area'];bars(axes[0],['Urbana','Rural'],a.informal_pct,'Área de residencia')
    e=d[d.variable=='nnivins'];bars(axes[1],[EDU[int(x)] for x in e.category],e.informal_pct,'Nivel de instrucción');save(fig,'profiles')
    fig,ax=plt.subplots(figsize=(7,3.5))
    for name,g in pd.read_csv(T/'calibration.csv').groupby('model'):ax.plot(g.predicted,g.observed,'o-',label=name,color=COL[name])
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Probabilidad estimada media',ylabel='Proporción informal observada',xlim=(0,1),ylim=(0,1));ax.legend();save(fig,'calibration')
    fig,ax=plt.subplots(figsize=(7.5,3.4));r=pd.read_csv(T/'importance_rf.csv').set_index('feature');l=pd.read_csv(T/'importance_logistic.csv').set_index('feature');y=np.arange(len(r))
    ax.barh(y-.17,l.loc[r.index].auc_drop_mean,height=.3,color=COL['Logística'],label='Logística');ax.barh(y+.17,r.auc_drop_mean,height=.3,color=COL['Random Forest'],label='Random Forest');ax.set_yticks(y,[LABEL[x] for x in r.index]);ax.invert_yaxis();ax.set_xlabel('Reducción media de ROC-AUC ponderado');ax.legend();save(fig,'importance')
    g=pd.read_csv(T/'group_metrics.csv');sub=g[g.group.isin(['area','p02'])].copy();sub['label']=[{('area',1):'Urbana',('area',2):'Rural',('p02',1):'Hombres',('p02',2):'Mujeres'}[(x,int(y))] for x,y in zip(sub.group,sub.value)]
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    for ax,key,title in zip(axes,['roc_auc','specificity'],['Discriminación por grupo','Especificidad al umbral 0,50']):
     for name,p in sub.groupby('model'):ax.plot(p[key],p.label,'o',label=name,color=COL[name],markersize=7)
     ax.set_xlim(.3 if key=='specificity' else .75,1);ax.set_title(title,fontsize=10);ax.set_xlabel('Especificidad' if key=='specificity' else 'ROC-AUC');ax.legend(fontsize=8)
    save(fig,'subgroups')
    p=d[d.variable=='prov'].sort_values('informal_pct');fig,axes=plt.subplots(1,2,figsize=(8,5.6),sharey=True);y=np.arange(len(p));axes[0].barh(y,p.informal_pct,color='#981C3E');axes[0].set_yticks(y,[PROV[int(x)] for x in p.category]);axes[0].set(xlabel='Informalidad ponderada (%)',xlim=(0,100));axes[1].barh(y,p.informal_expanded/1000,color='#3A637D');axes[1].set_xlabel('Población informal estimada (miles)');save(fig,'territory')
    fig,ax=plt.subplots(figsize=(7,3.4));a=d[d.variable=='age_group'];ax.plot(a.category,a.informal_pct,'o-',color='#981C3E');ax.set(xlabel='Edad en años',ylabel='Informalidad ponderada (%)',ylim=(0,100));save(fig,'age')
    v=pd.read_csv(T/'associations_cramers_v.csv',index_col=0);fig,ax=plt.subplots(figsize=(7.5,5.3));im=ax.imshow(v,cmap='Blues',vmin=0,vmax=1)
    ax.set_xticks(range(len(v)),[LABEL[x] for x in v.columns],rotation=55,ha='right');ax.set_yticks(range(len(v)),[LABEL[x] for x in v.index]);ax.spines[['top','right']].set_visible(True)
    for i in range(len(v)):
     for j in range(len(v)):ax.text(j,i,f'{v.iloc[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if v.iloc[i,j]>.6 else 'black')
    ax.tick_params(labelsize=11)
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03,label='V de Cramér ponderada');save(fig,'associations')
    c=pd.read_csv(T/'curves.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for ax,kind in zip(axes,['ROC','PR']):
     for name,p in c[c.curve==kind].groupby('model'):ax.plot(p.x,p.y,label=name,color=COL[name])
     ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 − especificidad' if kind=='ROC' else 'Sensibilidad',ylabel='Sensibilidad' if kind=='ROC' else 'Precisión');ax.legend(fontsize=8)
    axes[0].plot([0,1],[0,1],'--',color='gray');q=pd.read_csv(T/'split_summary.csv').set_index('split').loc['test','informal_pct_weighted']/100;axes[1].axhline(q,ls='--',color='gray');save(fig,'roc_pr')
    m=pd.read_csv(T/'test_metrics.csv');m=m[(m.weighted==False)&(m.threshold==.5)];fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for ax,(_,r) in zip(axes,m.iterrows()):
     arr=np.array([[r.tn,r.fp],[r.fn,r.tp]]);ax.imshow(arr,cmap='Blues',vmin=0,vmax=14000);ax.set_xticks([0,1],['Formal','Informal']);ax.set_yticks([0,1],['Formal','Informal']);ax.set(xlabel='Clasificación predicha',ylabel='Clasificación observada',title=r.model)
     for i in range(2):
      for j in range(2):ax.text(j,i,f'{int(arr[i,j]):,}'.replace(',','.'),ha='center',va='center',color='white' if arr[i,j]>7000 else 'black',fontsize=12)
    save(fig,'confusion')
    t=pd.read_csv(T/'thresholds_oof.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    for ax,(name,g) in zip(axes,t.groupby('model')):
     for key,label,style in [('precision','Precisión','-'),('recall','Sensibilidad','--'),('f1','F1',':')]:ax.plot(g.threshold,g[key],style,label=label)
     ax.set(xlabel='Umbral',ylabel='Métrica ponderada OOF',ylim=(0,1),title=name);ax.legend(fontsize=8)
    save(fig,'thresholds')
    ci=pd.read_csv(T/'bootstrap_intervals.csv')
    metrics=pd.read_csv(T/'test_metrics.csv')
    metrics=metrics[(metrics.weighted==True)&(metrics.threshold==.5)].set_index('model')
    fig,axes=plt.subplots(1,2,figsize=(8,2.8))
    for ax,key,title in zip(axes,['roc_auc','brier'],['Diferencia en ROC-AUC','Diferencia en Brier']):
     row=ci[(ci.contrast=='RF menos Logística')&(ci.metric==key)].iloc[0]
     point=metrics.loc['Random Forest',key]-metrics.loc['Logística',key]
     ax.errorbar(point,0,xerr=[[point-row.lower],[row.upper-point]],fmt='o',capsize=6,color=COL['Random Forest'])
     ax.axvline(0,color='gray',ls='--');ax.set_yticks([]);ax.set_title(title,fontsize=11)
     ax.set_xlabel('Random Forest menos logística');ax.set_ylim(-1,1)
     ax.text(.5,.82,f'{point:.4f} [{row.lower:.4f}; {row.upper:.4f}]',transform=ax.transAxes,ha='center',fontsize=9)
     ax.text(.5,.12,'Valores positivos favorecen RF' if key=='roc_auc' else 'Valores negativos favorecen RF',transform=ax.transAxes,ha='center',fontsize=9)
    save(fig,'paired_uncertainty')
    a=json.loads((ROOT/'outputs/audit/audit_summary.json').read_text())
    values=[100*a['informal_rows']/a['analytic_rows'],100*a['weighted_informal']/a['weighted_analytic'],100*a['weighted_informal']/a['weighted_employed']]
    fig,ax=plt.subplots(figsize=(8,3.2))
    ax.barh(['Muestra binaria sin ponderar','Universo binario ponderado','Todos los ocupados ponderados'],values,color=['#777777','#981C3E','#3A637D'])
    ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Proporción del sector informal (%)')
    for i,v in enumerate(values):ax.text(v+1,i,f'{v:.2f} %',va='center')
    save(fig,'denominators')
    manifest=[('profiles','Perfiles de informalidad por área e instrucción'),('calibration','Calibración de probabilidades en prueba'),('importance','Importancia predictiva por permutación'),('subgroups','Desempeño por área y sexo'),('territory','Intensidad y magnitud de la informalidad por provincia'),('age','Informalidad por grupos de edad'),('associations','Asociaciones entre las variables analíticas'),('roc_pr','Curvas ROC y precisión–sensibilidad en prueba'),('confusion','Matrices de confusión sin ponderar en prueba'),('thresholds','Sensibilidad del desempeño al umbral en validación')]
    manifest.extend([('paired_uncertainty','Incertidumbre de la comparación pareada de modelos'),('denominators','Ponderación y denominador de la proporción informal')])
    (ROOT/'docs/figures.json').write_text(json.dumps([{'number':i+1,'file':name+'.png','caption':f'Figura {i+1}. {title}'} for i,(name,title) in enumerate(manifest)],ensure_ascii=False,indent=2))
    print(f'{len(manifest)} figuras generadas desde resultados ejecutados.')




def verificar_cifras(root):
    """Compara a la precisión publicada: NO alimenta ningún entrenamiento."""
    t=root/'outputs/tables'
    a=json.loads((root/'outputs/audit/audit_summary.json').read_text())
    splits=pd.read_csv(t/'split_summary.csv').set_index('split')
    m=pd.read_csv(t/'test_metrics.csv')
    m=m[m.weighted & m.threshold.eq(.5)].set_index('model')
    ci=pd.read_csv(t/'bootstrap_intervals.csv')
    ci=ci[(ci.contrast=='RF menos Logística') & (ci.metric=='roc_auc')].iloc[0]
    checks=[('Universo',a['analytic_rows'],154330,0),
            ('Train',int(splits.loc['train','n']),123613,0),
            ('Test',int(splits.loc['test','n']),30717,0),
            ('Logística ROC-AUC',m.loc['Logística','roc_auc'],.8705,4),
            ('Logística Brier',m.loc['Logística','brier'],.1433,4),
            ('RF ROC-AUC',m.loc['Random Forest','roc_auc'],.8874,4),
            ('RF Brier',m.loc['Random Forest','brier'],.1365,4),
            ('Informal ponderado (%)',a['analytic_informal_pct'],53.92,2),
            ('IC95 diferencia AUC inferior',ci.lower,.0121,4),
            ('IC95 diferencia AUC superior',ci.upper,.0215,4)]
    rows=[{'indicador':k,'recalculado':float(v),'tesis':ref,'decimales':dec,
           'diferencia_sin_redondear':float(v-ref),
           'coincide_precision_publicada':round(float(v),dec)==ref}
          for k,v,ref,dec in checks]
    result=pd.DataFrame(rows)
    result.to_csv(t/'verificacion_tesis.csv',index=False)
    print(result.to_string(index=False),flush=True)
    lines=['# Verificación de reproducción ENEMDU 2025','',
           'Comparación independiente realizada después de entrenar desde cero.',
           'Las diferencias sin redondear frente al texto reflejan su precisión publicada.',
           '', '| Indicador | Recalculado | Tesis | Coincide al redondear |',
           '|---|---:|---:|:---:|']
    for r in rows:
        lines.append(f"| {r['indicador']} | {r['recalculado']:.12f} | {r['tesis']} | {r['coincide_precision_publicada']} |")
    lines.extend(['','SHA-256 CSV de personas: `'+a['persons_sha256']+'`.',
                  'La proporción se refiere al universo formal/informal, excluyendo empleo doméstico y no clasificados.',
                  'Bootstrap por viviendas de prueba, 300 réplicas, semilla 2026; modelos fijos.',
                  'La igualdad se evalúa a la precisión de la tesis, no como igualdad binaria con números redondeados.'])
    (root/'verificacion_reproduccion.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return bool(result.coincide_precision_publicada.all())


def figura_confusion_ponderada(root):
    """Adicional a la figura original de recuentos: matriz usando fexp."""
    import matplotlib.pyplot as plt
    m=pd.read_csv(root/'outputs/tables/test_metrics.csv')
    m=m[m.weighted & m.threshold.eq(.5)]
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,(_,r) in zip(axes,m.iterrows()):
        arr=np.array([[r.tn,r.fp],[r.fn,r.tp]])
        ax.imshow(arr,cmap='Blues')
        ax.set_xticks([0,1],['Formal','Informal']);ax.set_yticks([0,1],['Formal','Informal'])
        ax.set(xlabel='Predicción',ylabel='Observado',title=r.model)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{arr[i,j]:,.0f}',ha='center',va='center',
                        color='white' if arr[i,j]>arr.max()/2 else 'black')
    fig.suptitle('Matriz ponderada: suma de fexp en la prueba; umbral 0,50')
    fig.tight_layout();fig.savefig(root/'outputs/figures/confusion_weighted.png',dpi=200)
    plt.close(fig)


def sensibilidad_predictores(ROOT):
    """Análisis NUEVO, sin búsqueda de hiperparámetros ni selección con prueba.

    Conserva todos los registros, particiones y pesos. Reinicia el generador
    bootstrap en 2026 para cada conjunto: usa las mismas multiplicidades de
    viviendas que ampliar_evidencia y las comparte entre los dos modelos.
    La regresión ajustada aquí NO sustituye los coeficientes del modelo original.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss, confusion_matrix)
    import matplotlib.pyplot as plt

    ROOT = Path(ROOT)
    T, F = ROOT/'outputs/tables', ROOT/'outputs/figures'
    F.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT/'data/processed/analytic.csv.gz', dtype=str)
    for col in ['p03', 'target', 'fexp', 'fold']:
        d[col] = pd.to_numeric(d[col])
    a = d[d.split == 'train'].reset_index(drop=True)
    b = d[d.split == 'test'].reset_index(drop=True)
    assert not set(a.group_dwelling) & set(b.group_dwelling)
    assert (len(d), len(a), len(b)) == (154330, 123613, 30717)
    assert d.fexp.notna().all() and d.fexp.gt(0).all()
    # Orden idéntico al principal original: categorías primero, edad al final.
    contextual = ['p02', 'p06', 'p15', 'nnivins', 'area', 'prov']
    principal = contextual + ['rama1', 'grupo1']
    conjuntos = {'Contextual': contextual, 'Principal': principal,
                 'Ampliado': principal + ['p42']}
    original = pd.read_csv(T/'test_metrics.csv')
    original = original[original.weighted & original.threshold.eq(.5)].set_index('model')
    codigos, viviendas = pd.factorize(b.group_dwelling)
    y, w = b.target.to_numpy(), b.fexp.to_numpy()
    resultados, replicas = [], []
    for conjunto, cats in conjuntos.items():
        features = cats + ['p03']
        assert d[features].notna().all().all(), 'No se permite cambiar el universo por faltantes.'
        predicciones, filas = {}, []
        for nombre in ['Logística', 'Random Forest']:
            pre = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
                ('age', StandardScaler(), ['p03'])])
            est = (LogisticRegression(C=1.0, max_iter=1500, solver='lbfgs')
                   if nombre == 'Logística' else
                   RandomForestClassifier(n_estimators=150, max_depth=20,
                       min_samples_leaf=20, max_features='sqrt',
                       random_state=2025, n_jobs=2))
            modelo = make_pipeline(pre, est)
            modelo.fit(a[features], a.target,
                       **{modelo.steps[-1][0]+'__sample_weight': a.fexp/a.fexp.mean()})
            p = modelo.predict_proba(b[features])[:, 1]
            predicciones[nombre] = p
            tn, fp, fn, tp = confusion_matrix(y, p >= .5, sample_weight=w,
                                               labels=[0, 1]).ravel()
            auc = roc_auc_score(y, p, sample_weight=w)
            brier = brier_score_loss(y, p, sample_weight=w)
            if conjunto == 'Principal':
                esperado = .8705 if nombre == 'Logística' else .8874
                assert round(float(auc), 4) == esperado, (nombre, auc)
                assert np.isclose(auc, original.loc[nombre, 'roc_auc'], atol=1e-10, rtol=0)
                assert np.isclose(brier, original.loc[nombre, 'brier'], atol=1e-10, rtol=0)
            filas.append(dict(conjunto=conjunto, modelo=nombre,
                predictores='; '.join(features), n_train=len(a), n_test=len(b),
                umbral=.5, roc_auc=auc, ap=average_precision_score(y, p, sample_weight=w),
                brier=brier, sensibilidad=tp/(tp+fn), especificidad=tn/(tn+fp),
                f1=2*tp/(2*tp+fp+fn)))
            print('Sensibilidad:', conjunto, nombre, 'AUC', auc, flush=True)
        rng = np.random.default_rng(2026)
        diferencias = []
        for i in range(300):
            count = np.bincount(rng.integers(0, len(viviendas), len(viviendas)),
                                minlength=len(viviendas))
            wb = w * count[codigos]
            delta = (roc_auc_score(y, predicciones['Random Forest'], sample_weight=wb)
                     - roc_auc_score(y, predicciones['Logística'], sample_weight=wb))
            diferencias.append(delta)
            replicas.append(dict(conjunto=conjunto, replica=i, diferencia_auc_rf_logistica=delta))
        lo, hi = np.quantile(diferencias, [.025, .975])
        delta = filas[1]['roc_auc'] - filas[0]['roc_auc']
        for fila in filas:
            fila.update(diferencia_auc_rf_logistica=delta, ic95_inferior=lo,
                        ic95_superior=hi, bootstrap_replicas=300, bootstrap_semilla=2026)
        resultados.extend(filas)
    s = pd.DataFrame(resultados)
    s.to_csv(T/'sensibilidad_predictores.csv', index=False)
    pd.DataFrame(replicas).to_csv(T/'sensibilidad_bootstrap_replicas.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colores = {'Logística': '#3A637D', 'Random Forest': '#981C3E'}
    for nombre, parte in s.groupby('modelo', sort=False):
        for ax, metrica in zip(axes[:2], ['roc_auc', 'brier']):
            ax.plot(parte.conjunto, parte[metrica], 'o-', label=nombre, color=colores[nombre])
            for x, val in enumerate(parte[metrica]):
                arriba = (nombre == 'Random Forest') if metrica == 'roc_auc' else (nombre == 'Logística')
                ax.annotate(f'{val:.4f}', (x, val), xytext=(0, 7 if arriba else -14),
                            textcoords='offset points', ha='center', fontsize=8)
    axes[0].set(ylabel='ROC-AUC ponderado', title='Discriminación (mayor es mejor)')
    axes[1].set(ylabel='Brier ponderado', title='Error probabilístico (menor es mejor)')
    for ax in axes[:2]: ax.legend(fontsize=8); ax.margins(y=.22)
    q = s.drop_duplicates('conjunto')
    for i, row in enumerate(q.itertuples()):
        # El segmento percentil no presupone que el punto caiga dentro del IC.
        axes[2].hlines(i, row.ic95_inferior, row.ic95_superior, color='#981C3E', linewidth=2)
        axes[2].plot(row.diferencia_auc_rf_logistica, i, 'o', color='#981C3E')
    axes[2].axvline(0, ls='--', color='gray')
    axes[2].set_yticks(range(3), q.conjunto)
    axes[2].invert_yaxis()
    axes[2].set(xlabel='AUC RF − logística (IC 95 %)', title='Bootstrap pareado: 300 réplicas')
    fig.tight_layout(); fig.savefig(F/'sensibilidad_predictores.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def odds_ratios_etiquetados(ROOT):
    """Análisis NUEVO de presentación: lee OR ya calculados; NO ajusta modelos.

    El XLSX de diccionario anual 2025 adjunto contiene nombres de campos, no
    etiquetas de valores. Etnia se verificó en formulario 2025; las etiquetas
    restantes se contrastaron con ANDA INEC 2022/2023 y el catálogo territorial.
    Se explicita esta limitación: no se certifica equivalencia completa de la
    clasificación de ramas CIIU 4.0 (ANDA) con CIIU 4.1 (diccionario 2025).
    Las etiquetas son descriptivas y NUNCA se usan para modificar los modelos.
    """
    T = Path(ROOT)/'outputs/tables'
    labels = {
        'p02': {1:'Hombre', 2:'Mujer'},
        'p06': {1:'Casado(a)', 2:'Separado(a)', 3:'Divorciado(a)', 4:'Viudo(a)', 5:'Unión libre', 6:'Soltero(a)'},
        'p15': {1:'Indígena', 2:'Afroecuatoriano(a)/afrodescendiente', 3:'Negro(a)',
                4:'Mulato(a)', 5:'Montubio(a)', 6:'Mestizo(a)', 7:'Blanco(a)', 8:'Otro(a)'},
        'nnivins': {1:'Ninguno', 2:'Centro de alfabetización', 3:'Educación básica',
                    4:'Educación media/bachillerato', 5:'Superior'},
        'area': {1:'Urbana', 2:'Rural'},
        'prov': dict(enumerate(['Azuay','Bolívar','Cañar','Carchi','Cotopaxi','Chimborazo',
            'El Oro','Esmeraldas','Guayas','Imbabura','Loja','Los Ríos','Manabí','Morona Santiago',
            'Napo','Pastaza','Pichincha','Tungurahua','Zamora Chinchipe','Galápagos','Sucumbíos',
            'Orellana','Santo Domingo de los Tsáchilas','Santa Elena'], 1)),
        'rama1': dict(enumerate(['Agricultura, ganadería, caza, silvicultura y pesca',
            'Explotación de minas y canteras','Industrias manufactureras',
            'Suministros de electricidad, gas y aire acondicionado','Distribución de agua y alcantarillado',
            'Construcción','Comercio y reparación de vehículos','Transporte y almacenamiento',
            'Alojamiento y servicios de comida','Información y comunicación','Actividades financieras y de seguros',
            'Actividades inmobiliarias','Actividades profesionales, científicas y técnicas',
            'Actividades y servicios administrativos y de apoyo','Administración pública, defensa y seguridad social',
            'Enseñanza','Actividades de servicios sociales y de salud','Artes, entretenimiento y recreación',
            'Otras actividades de servicios','Hogares privados con servicio doméstico',
            'Organizaciones extraterritoriales','No especificado'],1)),
        'grupo1': {1:'Personal directivo de administración pública y empresas',
            2:'Profesionales científicos e intelectuales',3:'Técnicos y profesionales de nivel medio',
            4:'Empleados de oficina',5:'Trabajadores de servicios y comerciantes',
            6:'Trabajadores calificados agropecuarios y pesqueros',7:'Oficiales, operarios y artesanos',
            8:'Operadores de instalaciones, máquinas y montadores',
            9:'Trabajadores no calificados y ocupaciones elementales',10:'Fuerzas Armadas',99:'No especificado'}}
    names = {'p02':'Sexo','p06':'Estado civil','p15':'Autoidentificación étnica',
             'nnivins':'Nivel de instrucción','area':'Área','prov':'Provincia',
             'rama1':'Rama de actividad','grupo1':'Grupo de ocupación'}
    base = 'https://anda.inec.gob.ec/anda5/index.php/catalog/'
    sources = {'p02':base+'1270/variable/F11/V812?name=p02',
        'p06':base+'1270/variable/F11/V817?name=p06',
        'p15':'https://www.ecuadorencifras.gob.ec/documentos/web-inec/EMPLEO/2025/Diciembre_2025/202512_Formulario_ENEMDU.pdf',
        'nnivins':base+'1270/variable/F11/V950?name=nnivins',
        'area':base+'1270/variable/F11/V805?name=area',
        'prov':base+'1021/variable/F7/V2301?name=Prov',
        'rama1':base+'1270/variable/F11/V960?name=rama1',
        'grupo1':base+'1164/variable/F7/V628?name=grupo1'}
    note = ('Contrastes de logística regularizada, C = 1; sin valores p ni IC de diseño muestral. '
            'Asociaciones condicionales, no relaciones causales. OR no equivale a razón de probabilidades.')
    or_path, coef_path = T/'logistic_odds_ratios.csv', T/'logistic_coefficients.csv'
    before = (sha256(or_path), sha256(coef_path))
    raw = pd.read_csv(or_path, dtype={'category':str,'reference':str})
    rows = []
    for r in raw.itertuples():
        category, ref = int(r.category), int(r.reference)
        assert ref == 1, 'La referencia original no es la categoría 1.'
        assert category in labels[r.variable], (r.variable, category)
        rows.append(dict(variable=names[r.variable], codigo_variable=r.variable,
            categoria=labels[r.variable][category], codigo_categoria=category,
            referencia=labels[r.variable][ref], codigo_referencia=ref,
            es_referencia=category == ref, log_contraste=r.log_odds_contrast,
            odds_ratio=r.odds_ratio, unidad='Categoría frente a referencia',
            fuente_etiqueta=sources[r.variable],
            estado_etiqueta=('Verificado en formulario 2025' if r.variable=='p15' else
                'Fuente INEC anterior; pendiente cotejo con etiquetas completas 2025'), nota=note))
    c = pd.read_csv(coef_path)
    age = c[c.feature.eq('age__p03')]
    assert len(age) == 1
    age = age.iloc[0]
    # El coeficiente ya está expresado por una desviación estándar; no se
    # reescala, no se vuelve a estimar la logística y no se calcula un IC.
    rows.append(dict(variable='Edad', codigo_variable='p03', categoria='Incremento de 1 desviación estándar',
        codigo_categoria='', referencia='Edad inicial; incremento de 1 DE', codigo_referencia='',
        es_referencia=False, log_contraste=age.coefficient, odds_ratio=age.exp_coefficient,
        unidad='1 DE del escalador original de entrenamiento', fuente_etiqueta='logistic_coefficients.csv: age__p03',
        estado_etiqueta='Coeficiente original sin reajuste', nota=note))
    pd.DataFrame(rows).to_csv(T/'odds_ratios_etiquetados.csv', index=False)
    assert before == (sha256(or_path), sha256(coef_path)), 'Se modificaron las tablas originales.'


def ejecutar(persons,root):
    """Flujo completo: siempre vuelve a estimar CV, modelos y resultados."""
    run(persons,out=root/'outputs/audit')
    prepare(persons,out=root/'data/processed',tables=root/'outputs/tables')
    entrenar(root)
    contrastes(root)
    ampliar_evidencia(root)
    sensibilidad_predictores(root)
    odds_ratios_etiquetados(root)
    generar_figuras(root)
    figura_confusion_ponderada(root)
    # Actualiza los estados que en el repositorio histórico quedaron pendientes.
    p=root/'outputs/tables/training_protocol.json'
    protocol=json.loads(p.read_text())
    protocol.update(bootstrap_replicates=300,bootstrap_seed=2026,
                    uncertainty='bootstrap pareado de viviendas en prueba; modelos fijos; ejecutado')
    p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    return verificar_cifras(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data',type=Path,required=True,help='ZIP original INEC o CSV de personas')
    parser.add_argument('--output',type=Path,default=Path('resultados_enemdu'),help='Carpeta nueva de resultados')
    args=parser.parse_args()
    if not args.data.is_file():parser.error('No existe el archivo indicado en --data.')
    root=args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('La carpeta de salida debe estar vacía o no existir; así no se mezclan resultados antiguos.')
    for d in ['outputs/tables','outputs/figures','outputs/audit','docs','data/processed','models']:
        (root/d).mkdir(parents=True,exist_ok=True)
    start=time.time()
    # No se ocultan advertencias de convergencia: se guardan en el registro.
    warnings.simplefilter('always',ConvergenceWarning)
    packages=['numpy','pandas','scikit-learn','scipy','matplotlib','joblib','threadpoolctl']
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'versions':{p:importlib.metadata.version(p) for p in packages},
                 'input_file':args.data.name,'input_sha256':sha256(args.data),
                 'script_sha256':sha256(Path(__file__))}
    (root/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print('Entorno e identidad de los datos:',json.dumps(environment,indent=2),flush=True)
    # Solo se materializa el CSV de personas en una carpeta temporal. No se usa
    # extractall: se evita extraer rutas arbitrarias contenidas en un ZIP.
    if zipfile.is_zipfile(args.data):
        with tempfile.TemporaryDirectory(prefix='enemdu_') as temp:
            with zipfile.ZipFile(args.data) as archive:
                candidates=[n for n in archive.namelist()
                            if Path(n).name=='BDDenemdu_personas_2025_anual.csv']
                if len(candidates)!=1:raise ValueError('Se requiere exactamente un CSV de personas anual 2025.')
                persons=Path(temp)/'personas.csv'
                with archive.open(candidates[0]) as src,persons.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
            matched=ejecutar(persons,root)
    else:
        matched=ejecutar(args.data,root)
    environment.update(elapsed_seconds=time.time()-start,all_thesis_values_match=matched)
    (root/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    # Huellas de TODOS los resultados; permiten auditar esta corrida sin
    # introducir los valores históricos como entrada del análisis.
    files=[p for p in root.rglob('*') if p.is_file()]
    pd.DataFrame([{'file':str(p.relative_to(root)),'sha256':sha256(p),'bytes':p.stat().st_size}
                  for p in sorted(files)]).to_csv(root/'manifest_sha256.csv',index=False)
    print(f'FIN: {environment["elapsed_seconds"]:.1f} segundos. Coincidencia: {matched}',flush=True)
    if not matched:
        print('Hay diferencias: revisar verificacion_tesis.csv, huellas y versiones. No se ajustó nada para forzarlas.')


if __name__=='__main__':
    main()
