import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import tempfile
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing, Point, LineString
from adet.evaluation import text_eval_script_det
import zipfile
import pickle


# Modified from TESTR. Only the detection metrics are evaluated.
class TextDetEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )

        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # For ICDAR ArT2019 evaluation on the official website.
        # The saved json file can be directly submitted to the website.
        self.submit = False

        # use dataset_name to decide eval_gt_path
        if "rotate" in dataset_name:
            if "totaltext" in dataset_name:
                self._text_eval_gt_path = "datasets/evaluation/gt_totaltext_rotate.zip"
        elif "totaltext" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_totaltext.zip"
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_ctw1500.zip"
        elif "art" in dataset_name:
            self._text_eval_gt_path = None
            self.submit = True
        elif "inversetext" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_inversetext.zip"
        elif "icdar2015" in dataset_name:
            self._text_eval_gt_path = "datasets/evaluation/gt_icdar2015.zip"
        else:
            raise NotImplementedError

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"], "file_name": input["file_name"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, input["image_id"], input["file_name"])
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results"):
        # ── ORIGINAL algorithm preserved exactly (tmp_txt → parse → group → write),
        #     with only bugfixes: try-except on malformed lines, len check instead of assert.
        dirn = os.path.abspath(temp_dir)
        if os.path.isfile(dirn):
            os.remove(dirn)
        os.makedirs(dirn, exist_ok=True)

        # Step 1: write all valid detections into one intermediate text file
        tmp_txt = os.path.join(dirn, 'temp_all_det_cors.txt')
        with open(file_path, 'r') as f:
            data = json.load(f)
            os.makedirs(dirn, exist_ok=True)
            with open(tmp_txt, 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_id'])
                        for i in range(len(data[ix]['polys'])):
                            outstr = outstr + str(int(data[ix]['polys'][i][0])) + ',' + str(int(data[ix]['polys'][i][1])) + ','
                        outstr = outstr + str(round(data[ix]['score'], 3)) + ',' + '####' + '\n'
                        f2.writelines(outstr)
        fres = open(tmp_txt, 'r').readlines()

        # Step 2: parse intermediate file → one .txt per image
        os.makedirs(dirn, exist_ok=True)
        n_written = 0
        for line in fres:
            line = line.strip()
            if not line:
                continue
            s = line.split(': ')
            # guard against malformed image_id (e.g. leftover from a previous crash)
            try:
                filename = '{:07d}.txt'.format(int(s[0]))
            except (ValueError, IndexError):
                continue
            outName = os.path.join(dirn, filename)
            # 'a' mode preserves original semantics (multiple lines for same image)
            with open(outName, 'a') as fout:
                ptr = s[1].strip().split(',')
                if len(ptr) < 2 or ptr[-1] != '####':
                    continue
                cors = ','.join(e for e in ptr[:-2])
                fout.writelines(cors + ',####' + '\n')
            n_written += 1

        # clean up intermediate file
        if os.path.exists(tmp_txt):
            os.remove(tmp_txt)

        self._logger.info(
            f"to_eval_format: {len(data)} dets in JSON, "
            f"{len(fres)} pass score filter, {n_written} lines written to per-image txt files"
        )

    def sort_detection(self, temp_dir):
        origin_file = os.path.abspath(os.path.normpath(temp_dir))
        output_file = os.path.join(os.path.dirname(origin_file),
                                   "final_" + os.path.basename(origin_file))

        if not os.path.isdir(output_file):
            os.mkdir(output_file)

        files = glob.glob(os.path.join(origin_file, '*.txt'))
        files.sort()
        self._logger.info(f"sort_detection: found {len(files)} .txt files in {origin_file}")

        n_skipped = 0
        for i in files:
            out = i.replace(origin_file, output_file)
            try:
                with open(i, 'r') as fin:
                    lines = fin.readlines()
            except Exception as e:
                self._logger.warning('Skip missing/unreadable file {}: {}'.format(i, e))
                n_skipped += 1
                continue
            with open(out, 'w') as fout:
                for iline, line in enumerate(lines):
                    ptr = line.strip().split(',')
                    cors = ptr[:-1]
                    if len(cors) % 2 != 0:
                        continue  # skip malformed line
                    pts = [(int(cors[j]), int(cors[j+1])) for j in range(0, len(cors), 2)]
                    try:
                        pgt = Polygon(pts)
                    except Exception as e:
                        print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                        continue
                    
                    if not pgt.is_valid:
                        print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                        continue
                        
                    pRing = LinearRing(pts)
                    if pRing.is_ccw:
                        pts.reverse()
                    outstr = ''
                    for ipt in pts[:-1]:
                        outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
                    outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
                    outstr = outstr + ',####'
                    fout.writelines(outstr + '\n')

        # Write det.zip into temp_dir so it survives until shutil.rmtree(temp_dir)
        # in the caller's finally block.  Avoids CWD-dependent absolute paths
        # that can race or resolve differently under DDP / NFS.
        det_zip_path = os.path.join(temp_dir, "det.zip")
        with zipfile.ZipFile(det_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_file):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        # clean intermediate files (origin_file = temp_dir itself is cleaned by caller)
        # but output_file ("final_<prefix>") is our own temporary staging dir → delete now
        shutil.rmtree(output_file, ignore_errors=True)
        self._logger.info(
            f"sort_detection: {len(files) - n_skipped} files processed, "
            f"{n_skipped} skipped → {det_zip_path}"
        )
        return det_zip_path
    
    def evaluate_with_official_code(self, result_path, gt_path):
        return text_eval_script_det.text_eval_main_det(det_file=result_path, gt_file=gt_path)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}
        PathManager.mkdirs(self._output_dir)

        if self.submit:
            file_path = os.path.join(self._output_dir, "art_submit.json")
            coco_results = {}
            for prediction in predictions:
                key = 'res_' + prediction['file_name'].split('/')[-1].split('.')[0].split('_')[-1]
                coco_results[key] = prediction["instances"]
        else:
            coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
            file_path = os.path.join(self._output_dir, "text_results.json")

        self._logger.info("Saving results to {}".format(file_path))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        self._results = OrderedDict()
        
        if not self._text_eval_gt_path:
            return copy.deepcopy(self._results)
        # eval text — use unique temp dir to avoid NFS phantom-file races
        temp_dir = tempfile.mkdtemp(prefix="det_eval_")
        try:
            self.to_eval_format(file_path, temp_dir)
            result_path = self.sort_detection(temp_dir)          # writes det.zip inside temp_dir
            text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path)
            # det.zip is inside temp_dir → cleaned by shutil.rmtree below, no explicit remove needed
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        for task in ["det_method"]:
            result = text_result[task]
            groups = re.match(template, result).groups()
            for i in range(3):
                key = groups[i * 2 + 1]  # "precision", "recall", "hmean"
                val = float(groups[(i + 1) * 2])
                if key == "precision":
                    self._results["P"] = round(val * 100, 2)
                elif key == "recall":
                    self._results["R"] = round(val * 100, 2)
                elif key == "hmean":
                    self._results["F1"] = round(val * 100, 2)

        # ── COCO mAP evaluation (bbox) ──
        if not self.submit:
            with open(file_path, 'r') as f:
                det_data = json.load(f)

            coco_preds = []
            for det in det_data:
                coco_preds.append({
                    "image_id": det["image_id"],
                    "category_id": det.get("category_id", 1),
                    "bbox": self._poly_to_bbox(det["polys"]),
                    "score": det["score"],
                })

            if len(coco_preds) == 0:
                self._logger.warning("No detections for COCO mAP.")
                self._results["AP50"] = 0.0
                self._results["AP50:95"] = 0.0
                for iou_name in ["AP60", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95"]:
                    self._results[iou_name] = 0.0
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_dt = self._coco_api.loadRes(coco_preds)
                    coco_eval = COCOeval(self._coco_api, coco_dt, iouType="bbox")
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()

                # Parse per-IoU-threshold AP from accumulated precision array.
                # Shape: [T=10, R=101, K, A=4, M=3]
                # T indices: 0=0.50, 1=0.55, 2=0.60, ..., 9=0.95
                prec = coco_eval.eval['precision']
                # AP at IoU threshold t: mean of precision[t, :, 0, 0, 2] over R (exclude -1)
                iou_idx_map = {
                    "AP50": 0,  # IoU=0.50
                    "AP60": 2,  # IoU=0.60
                    "AP70": 4,  # IoU=0.70
                    "AP75": 5,  # IoU=0.75
                    "AP80": 6,  # IoU=0.80
                    "AP85": 7,  # IoU=0.85
                    "AP90": 8,  # IoU=0.90
                    "AP95": 9,  # IoU=0.95
                }
                for ap_name, t_idx in iou_idx_map.items():
                    p_per_recall = prec[t_idx, :, 0, 0, 2]
                    valid = p_per_recall[p_per_recall > -1]
                    self._results[ap_name] = round(float(np.mean(valid)) * 100, 2) if len(valid) > 0 else 0.0

                # stats[0]=AP@.5:.95, stats[1]=AP@.50
                ap50_95 = coco_eval.stats[0] * 100 if len(coco_eval.stats) > 0 else 0.0
                self._results["AP50:95"] = round(ap50_95, 2)

        # ── Compact one-line summary ──
        self._logger.info(
            f"DET_RESULT: P={self._results['P']:.2f} R={self._results['R']:.2f} "
            f"F1={self._results['F1']:.2f} AP50={self._results['AP50']:.2f} "
            f"AP50:95={self._results['AP50:95']:.2f}"
        )
        # ── Per-IoU AP curve (for offline attribution) ──
        ap_curve_parts = []
        for ap_name in ["AP50", "AP60", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95"]:
            if ap_name in self._results:
                ap_curve_parts.append(f"{ap_name}={self._results[ap_name]:.2f}")
        if ap_curve_parts:
            self._logger.info(f"AP_CURVE: " + " ".join(ap_curve_parts))

        return copy.deepcopy(self._results)


    def instances_to_coco_json(self, instances, img_id, img_name):
        img_name = img_name.split('/')[-1].split('.')[0]
        num_instances = len(instances)
        if num_instances == 0:
            return []

        scores = instances.scores.tolist()
        if self.use_polygon:
            pnts = instances.polygons.numpy()
        else:
            pnts = instances.beziers.numpy()
    
        results = []
        if self.submit:
            for pnt, score in zip(pnts, scores):
                poly = self.pnt_to_polygon(pnt)  # list
                poly = [(int(p[0]), int(p[1])) for p in poly]
                assert(len(poly) %2 == 0 and len(poly) >= 3), 'cors invalid.'
                try:
                    pgt = Polygon(poly)
                except Exception as e:
                    continue
                if not pgt.is_valid:
                    continue

                is_ccw = pgt.exterior.is_ccw
                if not is_ccw:
                    poly = poly[::-1]

                result = {
                    "points": poly,
                    "confidence": score
                }
                results.append(result)
            return results
        else:
            for pnt, score in zip(pnts, scores):
                poly = self.pnt_to_polygon(pnt)
                result = {
                    "image_id": img_id,
                    "category_id": 1,
                    "polys": poly,
                    "score": score,
                    "image_name": img_name,
                }
                results.append(result)
            return results


    def pnt_to_polygon(self, ctrl_pnt):
        if self.use_polygon:
            return ctrl_pnt.reshape(-1, 2).tolist()
        else:
            u = np.linspace(0, 1, 20)
            ctrl_pnt = ctrl_pnt.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
            points = np.outer((1 - u) ** 3, ctrl_pnt[:, 0]) \
                + np.outer(3 * u * ((1 - u) ** 2), ctrl_pnt[:, 1]) \
                + np.outer(3 * (u ** 2) * (1 - u), ctrl_pnt[:, 2]) \
                + np.outer(u ** 3, ctrl_pnt[:, 3])
            
            # convert points to polygon
            points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
            return points.tolist()

    @staticmethod
    def _poly_to_bbox(poly):
        """Convert polygon [[x1,y1], ...] to COCO bbox [x, y, w, h]."""
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x_min, y_min = min(xs), min(ys)
        w, h = max(xs) - x_min, max(ys) - y_min
        return [float(x_min), float(y_min), float(w), float(h)]