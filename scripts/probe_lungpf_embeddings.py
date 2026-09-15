"""Identical frozen-feature multinomial logistic probes, GPU full-batch L-BFGS."""
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix

OUT=Path('outputs/xenium_step1500_benchmark')
C_GRID=[0.01,0.1,1.0,10.0]

def fit_linear(x,y,C,max_iter=400):
    """Minimize weighted mean CE + ||W||²/(2*C*sum(weights)), like sklearn.

    Bias is unpenalized. Class weights are n/(K*n_class). All features are
    standardized on training data only. Encoders never participate in fitting.
    """
    n,d=x.shape; classes=int(y.max())+1
    count=torch.bincount(y,minlength=classes).to(x.dtype)
    weight=(n/(classes*count))[y]
    W=torch.zeros((d,classes),device=x.device,dtype=x.dtype,requires_grad=True)
    b=torch.zeros(classes,device=x.device,dtype=x.dtype,requires_grad=True)
    opt=torch.optim.LBFGS([W,b],lr=1,max_iter=max_iter,history_size=20,
        tolerance_grad=1e-6,tolerance_change=1e-11,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        opt.zero_grad()
        # Chunking limits activation memory without stochastic optimization.
        objective=0.0
        for start in range(0,n,32768):
            logits=x[start:start+32768]@W+b
            ce=torch.nn.functional.cross_entropy(logits,y[start:start+32768],reduction='none')
            loss=(ce*weight[start:start+32768]).sum()/n
            loss.backward()
            objective+=float(loss.detach())
        penalty=W.square().sum()/(2*C*n)
        penalty.backward()
        calls+=1
        return torch.tensor(objective,device=x.device,dtype=x.dtype)+penalty.detach()
    opt.step(closure)
    loss=float(closure())
    grad=float(max(W.grad.abs().max(),b.grad.abs().max()))
    for _ in range(7):
        if grad<=1e-4: break
        opt.step(closure)
        loss=float(closure())
        grad=float(max(W.grad.abs().max(),b.grad.abs().max()))
    if grad>1e-4 and x.dtype==torch.float32:
        print('NUMERICAL_REFINEMENT_FLOAT64',C,grad,flush=True)
        return fit_linear(x.double(),y,C,max_iter=1600)
    if grad>1e-4:
        raise RuntimeError(f'Linear classifier has not converged: C={C}, gradient={grad}')
    return W.detach(),b.detach(),dict(objective=loss,gradient_max=grad,dtype=str(x.dtype),
        iterations=int(opt.state[W]['n_iter']),closure_evaluations=calls,
        converged_gradient=grad<=1e-4)

def predict(x,W,b,labels):
    with torch.inference_mode():
        pred=torch.cat([(part.to(W.dtype)@W+b).argmax(1).cpu() for part in x.split(32768)]).numpy()
    return labels[pred]

def verify_solver():
    # Independent numerical reference verifies the classifier objective/weights.
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(17)
    x=rng.normal(size=(600,12)).astype(np.float32)
    y=np.argmax(x[:,:3]+rng.normal(size=(600,3)),axis=1)
    reference=LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,tol=1e-9).fit(x,y)
    W,b,info=fit_linear(torch.tensor(x,device='cuda'),torch.tensor(y,device='cuda'),.1)
    logits=(torch.tensor(x,device='cuda')@W+b).cpu().numpy()
    prob=torch.softmax(torch.tensor(logits),dim=1).numpy()
    error=float(np.max(np.abs(prob-reference.predict_proba(x))))
    assert error<0.002,error
    (OUT/'linear_solver_verification.json').write_text(json.dumps(dict(max_probability_error=error,**info),indent=2))
    print('SKLEARN_EQUIVALENCE',error,flush=True)

def main():
    torch.set_num_threads(4); torch.manual_seed(42)
    verify_solver()
    directories=sorted((OUT/'embeddings').iterdir())
    protocol=json.loads(Path('data/benchmarking_data/lungpf/prepared/protocol.json').read_text())
    assert sorted(d.name for d in directories)==sorted(protocol['samples'])
    obs=pd.concat([pd.read_parquet(d/'cells.parquet') for d in directories],ignore_index=True)
    assert obs.full_cell_id.is_unique
    assert obs.groupby('patient')['split'].nunique().max()==1
    results=[]
    for task,column in [('cell_type','final_CT'),('region','region_label')]:
        for model in ['ours','spatialformer']:
            features=np.concatenate([np.load(d/f'{model}.npy') for d in directories])
            assert len(features)==len(obs) and np.isfinite(features).all()
            prefix=OUT/f'{model}_{task}'
            if prefix.with_suffix('.json').exists():
                cached=json.loads(prefix.with_suffix('.json').read_text())
                if all(t.get('converged_gradient',False) for t in cached['tuning']):
                    results.append(cached); continue
                (OUT/f'{model}_{task}.preliminary.json').write_text(json.dumps(cached,indent=2))
            labeled=obs[column].notna().to_numpy()
            masks={s:labeled & obs['split'].eq(s).to_numpy() for s in ['train','val','test']}
            scaler=StandardScaler().fit(features[masks['train']])
            X={s:torch.from_numpy(scaler.transform(features[m]).astype(np.float32)).cuda() for s,m in masks.items()}
            Y={s:obs.loc[m,column].to_numpy(dtype=str) for s,m in masks.items()}
            encoder=LabelEncoder().fit(Y['train'])
            ytrain=torch.tensor(encoder.transform(Y['train']),device='cuda')
            best=None; history=[]
            for C in C_GRID:
                start=time.monotonic()
                W,b,info=fit_linear(X['train'],ytrain,C)
                pred=predict(X['val'],W,b,encoder.classes_)
                score=float(f1_score(Y['val'],pred,labels=np.unique(Y['val']),average='macro',zero_division=0))
                entry=dict(C=C,val_macro_f1=score,seconds=time.monotonic()-start,**info)
                history.append(entry); print(model,task,entry,flush=True)
                if best is None or score>best[0]: best=(score,C,W.clone(),b.clone())
            _,C,W,b=best
            pred=predict(X['test'],W,b,encoder.classes_)
            test_labels=np.unique(Y['test'])
            row=dict(model=model,task=task,C=C,
                macro_f1=float(f1_score(Y['test'],pred,labels=test_labels,average='macro',zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(Y['test'],pred)),
                train_cells=len(Y['train']),val_cells=len(Y['val']),test_cells=len(Y['test']),
                train_classes=len(encoder.classes_),test_classes=len(test_labels),
                unseen_test_classes=sorted(set(test_labels)-set(encoder.classes_)),
                test_known_label_fraction=float(np.isin(Y['test'],encoder.classes_).mean()),
                tuning=history,metric_labels='all classes observed in held-out test cells, including unseen training labels',
                train_donors=sorted(obs.loc[masks['train'],'patient'].unique()),
                val_donors=sorted(obs.loc[masks['val'],'patient'].unique()),
                test_donors=sorted(obs.loc[masks['test'],'patient'].unique()))
            prefix.with_suffix('.json').write_text(json.dumps(row,indent=2))
            report=classification_report(Y['test'],pred,labels=test_labels,output_dict=True,zero_division=0)
            pd.DataFrame(report).T.to_csv(str(prefix)+'_per_class.csv')
            pred_frame=obs.loc[masks['test'],['full_cell_id','patient']].copy()
            pred_frame['label']=Y['test']; pred_frame['prediction']=pred
            pred_frame.to_parquet(str(prefix)+'_predictions.parquet',index=False)
            all_labels=np.union1d(encoder.classes_,test_labels)
            pd.DataFrame(confusion_matrix(Y['test'],pred,labels=all_labels),
                index=all_labels,columns=all_labels).to_csv(str(prefix)+'_confusion.csv')
            np.savez(str(prefix)+'_classifier.npz',W=W.cpu().numpy(),b=b.cpu().numpy(),
                classes=encoder.classes_,mean=scaler.mean_,scale=scaler.scale_)
            results.append(row); print('TEST_RESULT',json.dumps(row),flush=True)
            del X,ytrain,W,b,best
            torch.cuda.empty_cache()
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in results]).to_csv(OUT/'metrics.csv',index=False)

if __name__=='__main__': main()
