# tennis-SR
* Coded prototype (in Python) capable of solving the task or part of the task.
* Process of evaluating prototype performance.
* Review of the research field relevant for this challenge (scientific articles, git repositories, etc.).
* Assumptions, limitations and future work considerations.
* Project and code structure.

## Project and Code Structure

### Exploration stage (Jupyter notebooks)
1. [preprocessing.ipynb](preprocessing.ipynb)
* Extracting frames from the video.
1. [preprocessing_ann.ipynb](preprocessing_ann.ipynb)
* Dataset splits: train, val, test (70:15:15).
1. [exploring_scoreboards.ipynb](exploring_scoreboards.ipynb)
* Tesseract for OCR
* Preprocessing scoreboards for OCR.
* Postprocessing text extracted from scoreboards.
1. [train_ssd.ipynb](train_ssd.ipynb)
* Mmdet config.
* Checkpoint selection.

### Python files
1. [evaluate.py](evaluate.py)
* Prototype demo: 
    * run inference on the test set,
    * visualize bounding boxes and detected text, 
    * write to a video demo.avi.
* Evaluate:
    * TP, TN, FP, FN counting, 
    * saving evaluation results to demo.json
1. report.py
* Generate report from demo.json and mmdet logs

## Coded Prototype
[mmdetection](https://github.com/open-mmlab/mmdetection) (Apache License 2.0)
```
from mmdet.apis import init_detector, inference_detector
model = init_detector(
        config_file, checkpoint_file, device='cuda:0')
results = inference_detector(model, imgs)
```

## Evaluation
[Report](REPORT.md)