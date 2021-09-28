# [Demo](https://drive.google.com/file/d/1Pcqzh_ae1eSYSJWcBanTlCMMB6A2VhmM/view?usp=sharing) on the hold-out set
and [Error analysis](https://docs.google.com/presentation/d/13hVaKFBhUEU02q0IzpeMCW-YAYrG8Pp7AEZr7tx6nNA/edit?usp=sharing) (kind of)

# tennis-SR
* Coded prototype (in Python) capable of solving the task or part of the task.
* Process of evaluating prototype performance.
* Review of the research field relevant for this challenge (scientific articles, git repositories, etc.).
* Assumptions, limitations and future work considerations.
* Project and code structure.

## Solution
Two-stage model: image --> detect scoreboards --(best scoring proposal)--> OCR --> string filtering --> result

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
from ocr import OCR_Tesseract, StringProcessor

model = init_detector(
        config_file, checkpoint_file, device='cuda:0')
result = inference_detector(model, img)

(...) # unpack results

ocr = OCR_Tesseract()
string_processor = StringProcessor()
text = ocr.extract(img[y1:y2, x1:x2])
extracted_info = string_processor.read_values(text)
```

## Evaluation
[Report](REPORT.md)

### Hardware config.
2x Tesla K80 12GB   

## Review of the research field relevant
* [mmdetection](https://github.com/open-mmlab/mmdetection) -- a complete toolkit, novel architectures (well-maintained), model perfromance (often incl. FP etc.) info, pre-trained checkpoints, APIs, deployment tools.
* Performance comparison - any good paper -> experimental results and comparison section. 
* Tesseract tutorial: https://nanonets.com/blog/ocr-with-tesseract/#ocrwithpytesseractandopencv

## Assumptions, limitations and future work considerations.
Assumptions: 
* One scoreboard in one frame.
* Names, initals and surnames first, followed by numerical scores.
* The longest substring is surname.
* `>` is the only indicator that can be extracted.

Limitations:

Future work direction:
* Serving indicator --> string processing (rigid) or a simple classification network (data and overfitting)?
* 
