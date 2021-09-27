import os
import sys
import argparse
from tqdm import tqdm

import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sportradar interview task: infere and demo.')
    parser.add_argument('--json_file', type=str, default="demo.json")
    parser.add_argument('--model_name', type=str, default="SSD")
    parser.add_argument('--mmdet_eval_file', type=str, default="test_SSD.out")
    args = parser.parse_args()  

    with open(args.json_file) as f:
        data = json.load(f)
    with open(args.mmdet_eval_file, "r") as f:
        mmdet_eval = f.readlines()
    try:
        map_dict = json.loads(mmdet_eval[-1].splitlines()[0].replace("\'","\""))
    except json.decoder.JSONDecodeError:
        raise Exception("Expecting a dictionary of mAPs in the mmdet eval. file")
    required_keys = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75"]
    for key in required_keys:
        if key not in map_dict: raise Exception(f"{key} not in eval. dict")    

    measures = data["measures"]
    metrics = []
    for measure in measures:
        names, tp, tn, fp, fn = measure
        name = "+".join(names)
        accuracy = (tp + tn) / (tp + tn + fp +fn) if (tp + tn + fp +fn) else np.NaN
        precision = tp / (tp + fp) if (tp + fp) else np.NaN
        recall = tp / (tp + fn) if (tp + fn) else np.NaN
        f1 = 2*precision*recall/(precision+recall)
        metrics.append([name, accuracy, f1, precision, recall])
    metrics_table = "".join('''|{name}|{accuracy}%|
'''.format(name=name, 
               accuracy="{:.2f}".format(np.round(accuracy,4)*100), 
               f1=f1, 
               precision=precision, 
               recall=recall) \
        for (name, accuracy, f1, precision, recall) in metrics)

    report_text = '''
# Evaluation of {model_name}
on frames {frame_range}

## Object Detection Evaluation
For this task, I decided to stick with common metrics for object detection, as below. 

| mAP@.5 | mAP@.75 | mAP(@.5:.95) |
|---|---|---|
|{map50}|{map75}|{map}|

Possibly, a more aggresive IoU threshold (higher than .9) should be used given that the task is easy. 

Precision is often a good choice for imbalanced datasets (an "innate" problem in object detection).

Given that the dataset is repetetive, this metric is somewhat informative. However, more effort should be put in error analysis to elliminate systematic errors instead of focusing on improving the metric.


## Scoreboard Text Recogntion Evaluation

For this task, I decided to evaluate on scoreboards detected by the object detector. 

In this way, we can take a look at the performance of the OCR module "in the wild", i.e. on imperfectly localized scoreboards instead of perfectly cropped samples.

Pipelining object detection and OCR in the evaluation reveals that the serving indicator extraction (">") fails when the scoreboard is not cropped perfectly.

For OCR, a number of metrics exist:
* Character Error Rate (CER),
* Word Error Rate (WER),
* and more, e.g. ["Metrics for Complete Evaluation of OCR Performance"](https://hal.inria.fr/hal-01981731/document).

These could be used for tuning the scoreboard preprocessing pipeline (before feeding the scoreboards to Tesseract).

In my project, I designed my own -- more task-specific -- metrics (based on TP, FN number).

| field | Accuracy |
|---|---|
{metrics_table}

where ! means that gt string that include a substring are excluded from the evaluation (to estimate the maximum performance after completely eliminating an error)

and @ means evaluation at thresholds of similiarity score based on the Levenshtein distance.

This similarity score is calculated as (1-lev_dist(pred,gt)/len(gt)) 

E.g. for a surname "Medvedev" 70% threshold means that:

* Medvede (score 87.5%) will be tolerated [one deletion]
* Medvedeu (score 87.5%) will be tolerated [one substitution]
* Meduedeu (score 75%) will be tolerated [two deletions]
* Medved (score 75%) will be tolerated [two substitutions]
* Any more differencces will not be tolerated. 


For names, a lower threshold is acceptible. For scores, a higher threshold should be applied.

'''.format(model_name = args.model_name, 
               frame_range = data["frame_range"],
               map50 = map_dict["bbox_mAP"], 
               map75 = map_dict["bbox_mAP_50"], 
               map = map_dict["bbox_mAP_75"],
               metrics_table = metrics_table)
    
    with open("REPORT.md", "w") as f:
        f.write(report_text)
