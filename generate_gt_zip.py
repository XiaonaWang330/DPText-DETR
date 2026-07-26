"""
从 test_poly.json (COCO格式) 生成评估所需的 ground truth zip 文件。

GT zip 格式:
- 每个图片对应一个 {image_id}.txt 文件
- 每行是一个多边形: x1,y1,x2,y2,...,xn,yn (逗号分隔, 无空格)
"""

import json
import os
import zipfile
from shapely.geometry import Polygon, LinearRing
from shapely.validation import make_valid


def pts_to_flat(pts):
    """将 (x,y) 对列表转回 flat list"""
    result = []
    for pt in pts:
        result.extend(pt)
    return result


def ensure_valid_polygon(polygon):
    """
    确保多边形有效(不自相交)且顺时针。
    评估代码要求:
    1. 多边形不能自相交 (is_valid)
    2. 多边形点必须是顺时针的
    
    对于无效多边形，使用 shapely.make_valid() + buffer(0) 修复。
    修复失败则使用 convex_hull 作为回退。
    """
    if len(polygon) < 6:
        return polygon
    
    # 将 flat list 转为 (x, y) 对
    pts = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    
    try:
        poly = Polygon(pts)
        
        # 修复无效多边形(自相交等)
        if not poly.is_valid:
            try:
                # shapely 1.8+ 的 make_valid
                fixed = make_valid(poly)
            except Exception:
                # 回退: buffer(0) 有时也能修复
                fixed = poly.buffer(0)
            
            if fixed.is_empty:
                # 完全无法修复，用凸包
                fixed = poly.convex_hull
            
            # 如果是 MultiPolygon，取面积最大的子多边形
            if fixed.geom_type == 'MultiPolygon':
                fixed = max(fixed.geoms, key=lambda g: g.area)
            
            # 提取外环坐标
            pts = list(fixed.exterior.coords)[:-1]  # 去掉闭合点
        else:
            pts = list(poly.exterior.coords)[:-1]
        
        # 确保顺时针 (评估代码要求)
        ring = LinearRing(pts)
        if ring.is_ccw:
            pts.reverse()
        
    except Exception:
        # 完全失败时返回原始数据
        pass
    
    return pts_to_flat(pts)


def coco_to_gt_zip(coco_json_path, output_zip_path):
    """
    将 COCO 格式的 test_poly.json 转换为评估用的 GT zip 文件
    
    Args:
        coco_json_path: test_poly.json 路径
        output_zip_path: 输出的 .zip 文件路径
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建 image_id -> annotations 的映射
    img_annotations = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img in data.get("images", []):
            img_id = img["id"]
            anns = img_annotations.get(img_id, [])
            
            lines = []
            for ann in anns:
                polygon = ann.get("polys", ann.get("segmentation", [[]])[0])
                rec = ann.get("rec", [])
                if len(polygon) >= 6:
                    # 确保多边形有效(不自相交)且顺时针
                    polygon = ensure_valid_polygon(polygon)
                    # 格式: x1,y1,...,xn,yn,####transcription
                    # 注意: 坐标和文字之间用 ,#### 分隔
                    # 注意: transcription == "###" 会被视为 don't care, 不计入评估
                    coords_str = ','.join(str(int(c)) for c in polygon)
                    text = ''.join(chr(c) for c in rec) if rec else "text"
                    lines.append(f"{coords_str},####{text}")
            
            content = '\n'.join(lines)
            # 文件名使用 7位补零格式 (匹配 sort_detection 中的 '{:07d}.txt')
            filename = f"{img_id:07d}.txt"
            zf.writestr(filename, content)
    
    print(f"[OK] 已生成 GT zip: {output_zip_path}")
    print(f"     包含 {len(data.get('images', []))} 张图片的标注")


if __name__ == "__main__":
    # CTW1500 GT
    coco_to_gt_zip(
        coco_json_path="datasets/ctw1500/test_poly.json",
        output_zip_path="datasets/evaluation/gt_ctw1500.zip"
    )
    
    # TotalText GT
    coco_to_gt_zip(
        coco_json_path="datasets/totaltext/test_poly.json",
        output_zip_path="datasets/evaluation/gt_totaltext.zip"
    )
    
    # ICDAR2015 GT
    coco_to_gt_zip(
        coco_json_path="datasets/icdar2015/test_poly.json",
        output_zip_path="datasets/evaluation/gt_icdar2015.zip"
    )
