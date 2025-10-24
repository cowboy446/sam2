#!/usr/bin/env python3
"""
convert2coco.py

读取包含结构化数组的 .npz/.npy 文件，格式示例：
array([(0, 1, [502, 416, 940, 919]), (1, 1, [502, 411, 941, 918]), ...],
	  dtype=[('frame_idx', '<i4'), ('obj_id', '<i4'), ('bbox', '<i4', (4,))])
其中obj_id从1开始编号，bbox格式为 [x0,y0,x1,y1]。
将其转换为AP10K数据集的 COCO 格式的 JSON 标注并保存。参考你给的 COCO 示例（info/licenses/categories 等），
其余非关键项按示例补全（如 background、keypoints 置零等）。

用法：
  python sam2/convert2coco.py --input boxes.npz --img_dir path/to/images --output coco_boxes.json [--fill_image_size]

说明：
- images 的 file_name 采用 %05d.jpg（frame_idx 对应 00000.jpg 等）。
- bbox 从 [x0,y0,x1,y1] 转为 COCO 格式 [x,y,w,h]，并计算 area。
- category 默认使用示例的 mouse(id=40)。
- 若 --fill_image_size 指定，脚本会尝试读取图片尺寸写入 width/height；否则为 null。
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


# 参考示例的元信息（可按需修改）
EXAMPLE_INFO = {
	"description": "AP-10k",
	"url": "https://github.com/AlexTheBad/AP-10K",
	"version": "1.0",
	"year": 2021,
	"contributor": "ZHANG Rong",
	"date_created": "2025/10/22",
}

EXAMPLE_LICENSES = [
	{
		"id": 1,
		"name": "The MIT License",
		"url": "https://www.mit.edu/~amini/LICENSE.md"
	}
]

# 使用示例中的类别定义（含 17 个关键点名称）。
EXAMPLE_CATEGORY = {
	"id": 40,
	"name": "mouse",
	"supercategory": "Muridae",
	"keypoints": [
		"left_eye", "right_eye", "nose", "neck", "root_of_tail",
		"left_shoulder", "left_elbow", "left_front_paw", "right_shoulder",
		"right_elbow", "right_front_paw", "left_hip", "left_knee",
		"left_back_paw", "right_hip", "right_knee", "right_back_paw"
	],
	# skeleton 可选，这里留空或按需要补全
	"skeleton": [
        [
          1,
          2
        ],
        [
          1,
          3
        ],
        [
          2,
          3
        ],
        [
          3,
          4
        ],
        [
          4,
          5
        ],
        [
          4,
          6
        ],
        [
          6,
          7
        ],
        [
          7,
          8
        ],
        [
          4,
          9
        ],
        [
          9,
          10
        ],
        [
          10,
          11
        ],
        [
          5,
          12
        ],
        [
          12,
          13
        ],
        [
          13,
          14
        ],
        [
          5,
          15
        ],
        [
          15,
          16
        ],
        [
          16,
          17
        ]
      ]
}


def parse_args():
	p = argparse.ArgumentParser(description='将结构化 npz/npy 转为 COCO JSON（仅 bbox）')
	p.add_argument('--input', required=True, help='.npz（或 .npy）文件，含字段 frame_idx/obj_id/bbox')
	p.add_argument('--img_dir', required=True, help='图片目录，文件名为 %05d.jpg')
	p.add_argument('--output', required=True, help='COCO 标注 JSON 输出路径')
	p.add_argument('--category_id', type=int, default=EXAMPLE_CATEGORY['id'], help='类别 id（默认 40）')
	p.add_argument('--category_name', type=str, default=EXAMPLE_CATEGORY['name'], help='类别名称（默认 mouse）')
	p.add_argument('--license_id', type=int, default=1, help='images 中 license 字段（默认 1）')
	p.add_argument('--fill_image_size', action='store_true', help='尝试读取图片以填充 width/height')
	return p.parse_args()


def load_structured_array(path: str) -> np.ndarray:
	path = str(path)
	if path.endswith('.npz'):
		data = np.load(path)
		# 取第一个数组
		keys = list(data.keys())
		if len(keys) == 0:
			raise SystemExit('npz 文件中未找到数组')
		arr = data[keys[0]]
	else:
		arr = np.load(path, allow_pickle=True)
	return arr


def main():
	args = parse_args()
	arr = load_structured_array(args.input)

	# 校验字段
	if not hasattr(arr, 'dtype') or arr.dtype.names is None:
		raise SystemExit('输入必须为含字段 frame_idx/obj_id/bbox 的结构化数组')
	for f in ('frame_idx', 'obj_id', 'bbox'):
		if f not in arr.dtype.names:
			raise SystemExit(f'缺少字段: {f}')

	# 生成 images 列表（按出现的 frame_idx）
	frame_indices = sorted({int(r['frame_idx']) for r in arr})
	image_id_map = {}
	images = []
	next_image_id = 1
	img_dir = Path(args.img_dir)

	for fi in frame_indices:
		image_id_map[fi] = next_image_id
		filename = f"{fi:05d}.jpg"
		entry = {
			"license": args.license_id,
			"id": next_image_id,
			"file_name": filename,
			"width": None,
			"height": None,
			"background": 0,
		}
		if args.fill_image_size:
			img_path = img_dir / filename
			if img_path.exists():
				try:
					with Image.open(img_path) as im:
						entry['width'], entry['height'] = im.size
				except Exception:
					pass
		images.append(entry)
		next_image_id += 1

	# 生成 annotations（仅 bbox）
	annotations = []
	next_ann_id = 1
	# 为了兼容示例，构造 keypoints 零填充（17 关键点 * 3 = 51 个数）
	zero_keypoints = [0] * (len(EXAMPLE_CATEGORY['keypoints']) * 3)

	for row in arr:
		fi = int(row['frame_idx'])
		obj_id = int(row['obj_id'])
		bbox = np.asarray(row['bbox']).astype(int).tolist()
		if len(bbox) != 4:
			continue
		x0, y0, x1, y1 = bbox
		w = x1 - x0
		h = y1 - y0
		if w <= 0 or h <= 0:
			continue
		image_id = image_id_map.get(fi)
		if image_id is None:
			continue

		ann = {
			"id": next_ann_id,
			"image_id": image_id,
			"category_id": int(args.category_id),
			"bbox": [int(x0), int(y0), int(w), int(h)],
			"area": int(w * h),
			"iscrowd": 0,
			# 按示例补全 keypoints/num_keypoints（这里没有关键点，置零）
			"num_keypoints": 0,
			"keypoints": zero_keypoints.copy(),
			# 保留原始 obj_id，方便下游使用
			"attributes": {"obj_id": int(obj_id)}
		}
		annotations.append(ann)
		next_ann_id += 1

	coco = {
		"info": EXAMPLE_INFO,
		"licenses": EXAMPLE_LICENSES,
		"annotations": annotations,
		"images": images,
		"categories": [{
			"id": int(args.category_id),
			"name": args.category_name,
			"supercategory": EXAMPLE_CATEGORY.get('supercategory', 'object'),
			"keypoints": EXAMPLE_CATEGORY['keypoints'],
			"skeleton": EXAMPLE_CATEGORY.get('skeleton', [])
		}]
	}

	out_path = Path(args.output)
	if out_path.parent:
		os.makedirs(out_path.parent, exist_ok=True)
	with open(out_path, 'w') as f:
		json.dump(coco, f, indent=2, ensure_ascii=False)

	print(f"写入完成: {out_path}")
	print(f"images: {len(images)}, annotations: {len(annotations)}")


if __name__ == '__main__':
	main()

# python sam2/convert2coco.py --input /path/to/boxes.npz \
#   --img_dir notebooks/videos/SixView_Cam0_300f \
#   --output out/coco_boxes.json \
#   --fill_image_size