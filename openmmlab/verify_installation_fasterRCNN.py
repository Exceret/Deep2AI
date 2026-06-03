"""Verify installation."""

import subprocess
from pathlib import Path
import os
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

os.chdir("openmmlab")
print(f"working directory: {os.getcwd()}")

# * 获取当前目录下所有 .pth 文件
current_dir: Path = Path(".")

files: list[Path] = list(current_dir.glob("faster*.pth"))


subprocess.run(
    args=[
        "mim",
        "download",
        "mmdet",
        "--config",
        "faster-rcnn_regnetx-3.2GF_fpn_1x_coco",
        "--dest",
        ".",
    ],
    check=True,
)

config_file: str = "faster-rcnn_regnetx-3.2GF_fpn_1x_coco.py"
checkpoint_file: str = str(files[0])

model = init_detector(config_file, checkpoint_file, device="cuda:0")
img: str = "demo.jpg"  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# model.show_result(img, result, out_file="result.jpg")

# 注意，不需要运行多次否则会报错，构建一次即可(构建后对象已在当前内存中，如需重新构建请重启jupyter)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# visualizer.add_datasddddample(
#     "result",
#     img,
#     data_sample=result,
#     draw_gt=None,
#     show=None,
#     wait_time=0,
# )
visualizer.show()
