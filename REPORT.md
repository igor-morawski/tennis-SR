
# Evaluation of SSD
on frames [33425, 43245]

## Object Detection Evaluation
For this task, I decided to stick with common metrics for object detection, as below. 

| mAP@.5 | mAP@.75 | mAP(@.5:.95) |
|---|---|---|
|0.972|0.994|0.987|

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
|serving_player|3.72%|
|name_1@0.7+name_2@0.7|95.81%|
|name_1+name_2|93.16%|
|score_1+score_2|66.63%|
|score_1!ad+score_2!ad|69.59%|
|score_1@0.75+score_2@0.75|80.47%|


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

