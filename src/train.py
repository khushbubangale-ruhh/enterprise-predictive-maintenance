"""Train/test split, validation threshold selection, leakage-safe pipelines."""
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from src.common import load_data, engineer, TARGET, MODEL, ROOT

def build_pipeline(estimator):
    nums = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Temperature difference [K]','Estimated power [W]','Wear torque interaction']
    prep = ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['Type'])
    ])
    return Pipeline([('preprocess',prep),('model',estimator)])

def main():
    df=load_data(); x=engineer(df); y=df[TARGET].astype(int)
    x_train,x_tmp,y_train,y_tmp=train_test_split(x,y,test_size=.4,random_state=42,stratify=y)
    x_val,x_test,y_val,y_test=train_test_split(x_tmp,y_tmp,test_size=.5,random_state=42,stratify=y_tmp)
    choices={
        'logistic_regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),
        'random_forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=150,learning_rate=.07,random_state=42)
    }
    leaderboard=[]; best=None; best_score=-1
    for name,est in choices.items():
        pipeline=build_pipeline(est);pipeline.fit(x_train,y_train)
        val_probs=pipeline.predict_proba(x_val)[:,1]
        precision,recall,thresholds=precision_recall_curve(y_val,val_probs)
        f1_values=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
        ix=int(np.argmax(f1_values));threshold=float(thresholds[ix]) if len(thresholds) else .5
        ap=float(average_precision_score(y_val,val_probs)); score=float(f1_values[ix])
        leaderboard.append({'model':name,'validation_average_precision':round(ap,4),'validation_best_f1':round(score,4),'threshold':round(threshold,5)})
        if ap>best_score:best_score=ap;best=(name,pipeline,threshold)
    name,pipeline,threshold=best
    # Hold-out test evaluated only once after validation-based model and threshold selection.
    probs=pipeline.predict_proba(x_test)[:,1];pred=(probs>=threshold).astype(int)
    metrics={'selected_model':name,'decision_threshold':round(threshold,6),'dataset_rows':len(df),'train_rows':len(x_train),'validation_rows':len(x_val),'test_rows':len(x_test),'test_precision':float(precision_score(y_test,pred,zero_division=0)),'test_recall':float(recall_score(y_test,pred,zero_division=0)),'test_f1':float(f1_score(y_test,pred,zero_division=0)),'test_average_precision':float(average_precision_score(y_test,probs)),'test_roc_auc':float(roc_auc_score(y_test,probs)),'test_confusion_matrix':confusion_matrix(y_test,pred).tolist(),'validation_leaderboard':leaderboard}
    MODEL.parent.mkdir(parents=True,exist_ok=True); (ROOT/'reports').mkdir(exist_ok=True)
    joblib.dump({'pipeline':pipeline,'threshold':threshold,'model_name':name},MODEL)
    (ROOT/'reports'/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
