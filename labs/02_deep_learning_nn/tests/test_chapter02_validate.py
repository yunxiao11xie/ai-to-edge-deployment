import importlib.util, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
LAB=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(LAB))
from task_data import load_imu_activity_data
spec=importlib.util.spec_from_file_location("chapter02_validate",LAB/"validate.py")
validator=importlib.util.module_from_spec(spec); sys.modules[spec.name]=validator; spec.loader.exec_module(validator)
validate_challenge, validate_exercise=validator.validate_challenge, validator.validate_exercise

def test_exercise_accepts_valid_evidence():
    report=validate_exercise({"gradient_value":6.0,"mlp_predictions":[0,1,1,0],"mlp_targets":[0,1,1,0],"cnn_output":np.zeros((8,10))})
    assert report["passed"], report

def test_incomplete_exercise_fails_cleanly(): assert not validate_exercise({})["passed"]

def test_challenge_recomputes_metrics(tmp_path):
    _,y,_=load_imu_activity_data(); idx=np.arange(len(y)); _,test=train_test_split(idx,test_size=.2,random_state=42,stratify=y); pred=y[test]
    sub={"test_indices":test.tolist(),"predictions":pred.tolist(),"metrics":{"accuracy":accuracy_score(y[test],pred),"macro_f1":f1_score(y[test],pred,average="macro")},"confusion_matrix":confusion_matrix(y[test],pred,labels=[0,1,2]).tolist(),"parameter_count":163,"conclusion":"该模型在固定测试集上完成三分类。混淆矩阵用于定位容易混淆的动作，参数量只代表权重规模，部署前仍需在目标设备测量延迟、峰值内存、功耗和传感器漂移，并使用真实采集数据复验。"}
    path=tmp_path/"submission.json"; path.write_text(json.dumps(sub,ensure_ascii=False),encoding="utf-8")
    assert validate_challenge(path)["passed"]
