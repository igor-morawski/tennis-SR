import os
import sys
import argparse
from tqdm import tqdm

import json
import numpy as np
import cv2

from ocr import OCR_Tesseract, StringProcessor

from mmdet.apis import init_detector, inference_detector
import mmcv

# https://github.com/imalic3/levenshtein-distance-python/blob/master/levenshtein_distance.py
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sportradar interview task: infere and demo.')
    parser.add_argument('--config_file',
                        default="/home/phd/09/igor/mmdetection/configs/ssd/ssd512_sportradar.py")
    parser.add_argument('--data_dir', default="data")
    parser.add_argument('--output_dir', default="out")
    parser.add_argument(
        '--video_name', default="top-100-shots-rallies-2018-atp-season.mp4")
    parser.add_argument('--t_eval_start', default="22:17", type=str)
    parser.add_argument('--t_eval_end', default="-1", type=str)
    parser.add_argument('--frames_dir', default=os.path.join("data", "frames"))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--score_threshold', type=float, default=0.99)
    parser.add_argument('--demo_filepath', type=str, default="demo.avi")
    parser.add_argument('--demo_results_filepath', type=str, default="demo.json")
    parser.add_argument('--anno_sr_json', type=str, default="data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json")
    parser.add_argument('checkpoint_file', type=str)
    args = parser.parse_args()

    for t in (args.t_eval_start, args.t_eval_end):
        if t != "-1" and len(t.split(":")) != 2:
            raise ValueError(
                "Starting and ending time of the segment should be formates as minutes:seconds")
    if args.t_eval_start == "-1":
        raise ValueError(
            "Only the ending time of the segment can be set to -1")

    # Read video info
    video_file = os.path.join(args.data_dir, args.video_name)
    video = cv2.VideoCapture(video_file)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_file}: {video_length-1} frames at {video_fps} FPS")

    # Init video writer
    img = cv2.imread(os.path.join(args.frames_dir, "0.jpg"))
    height, width, _ = img.shape
    video_out = cv2.VideoWriter(args.demo_filepath, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), video_fps, (width, height))

    # Get range of frames to evaluate/demo on
    def convert_to_s(t):
        return int(t.split(":")[0])*60+int(t.split(":")[1])

    frame_eval_start = video_fps*convert_to_s(args.t_eval_start)
    frame_eval_end = video_fps*convert_to_s(
        args.t_eval_end) if args.t_eval_end != "-1" else video_length-2
    segment_lims = [int(frame_eval_start), int(frame_eval_end)]
    print(f"Frame range: {segment_lims}")
    assert all(os.path.exists(os.path.join(args.frames_dir,
                                           f"{i}.jpg")) for i in range(*segment_lims))

    # build detector, load weights
    model = init_detector(
        args.config_file, args.checkpoint_file, device='cuda:0')

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    idxs = [i for i in range(*segment_lims)]
    # run inference
    all_results = []
    for frame_idxs in tqdm(batch(idxs, args.batch_size)):
        imgs = [cv2.imread(os.path.join(
            args.frames_dir, f"{frame_idx}.jpg")) for frame_idx in frame_idxs]
        results = inference_detector(model, imgs)
        for result in results:
            if not len(result[0]):
                all_results.append(None)
                continue
            x1, y1, x2, y2, score = result[0][np.argmax(result[0][:, 4])]
            x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
            all_results.append([x1, y1, x2, y2] if score >
                                args.score_threshold else None)
    
    assert len(idxs) == len(all_results)

    ocr = OCR_Tesseract()
    string_processor = StringProcessor()
    extracted_info = {"score_threshold" : args.score_threshold, "frame_range" : segment_lims}
    for frame_idx, bbox in zip(idxs, all_results):
        img = cv2.imread(os.path.join(args.frames_dir, f"{frame_idx}.jpg"))
        player1_info, player2_info = [(None,None,None),(None,None,None)]
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = ocr.extract(img[y1:y2, x1:x2])
            if len(text.splitlines()) >= 2:
                (player1_info, player2_info) = string_processor.read_values(text)
        name_1, score_1, serving_1 = player1_info
        name_2, score_2, serving_2 = player2_info
        if (serving_1 and serving_2) or not (serving_1 or serving_2): 
            serving_player = ""
        elif serving_1:
            serving_player = "name_1"
        elif serving_2:
            serving_player = "name_2"
        extracted_info[str(frame_idx)] = {
            "serving_player" : serving_player,
            "name_1" : name_1,
            "name_2" : name_2,
            "score_1" : score_1,
            "score_2" : score_2,
            "bbox_detected" : bool(bbox),
        }
        put_result = f'{name_1}, score: {score_1} \n {name_2}, score: {score_2} \n serving: {serving_player}'
        cv2.putText(img, put_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (0, 0, 255), 2, cv2.LINE_AA)
        video_out.write(img)
    print(f"Demo saved to {args.demo_filepath}")

    with open(args.anno_sr_json) as f:
        text_groundtruth = json.load(f)

    # TP, TN, FP, FN
measures = ([["serving_player"], 0, 0, 0, 0], [["name_1@0.7", "name_2@0.7"], 0, 0, 0, 0], [["name_1", "name_2"], 0, 0, 0, 0], [["score_1", "score_2"], 0, 0, 0, 0], [["score_1!ad", "score_2!ad"], 0, 0, 0, 0], [["score_1@0.75", "score_2@0.75"], 0, 0, 0, 0])

for frame_idx in idxs:
    for entry in measures:
        for key in entry[0]:
            thresh = None
            ignore_if_in_string = None
            if "@" in key:
                key, thresh = key.split("@")
                thresh = float(thresh)
            if "!" in key:
                key, ignore_if_in_string = key.split("!")
            pred = extracted_info[str(frame_idx)][key] 
            if str(frame_idx) not in text_groundtruth.keys():
                gt = None
            else:
                gt = text_groundtruth[str(frame_idx)][key]
            if not gt: continue
            if not pred and not gt: # TN
                entry[2]+=1
            if pred and not gt: # FP
                entry[3]+=1
            if not pred and gt: # FN
                entry[4]+=1
            if pred and gt:
                if ignore_if_in_string and ignore_if_in_string.upper() in gt.upper():
                    entry[1] += 1 # TP
                elif thresh and (1-levenshtein_distance(gt.upper(), pred.upper())/len(gt) >= thresh):
                    entry[1] += 1 # TP
                elif not thresh and pred.upper() == gt.upper(): 
                    entry[1] += 1 # TP
                else: 
                    entry[4] += 1 # FN


extracted_info["measures"] = measures
with open(args.demo_results_filepath, 'w') as f:
    json.dump(extracted_info, f)


print(f"Extracted info saved to {args.demo_results_filepath}")
