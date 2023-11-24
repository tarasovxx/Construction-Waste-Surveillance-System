import torch
from carvekit.api.high import HiInterface

import PIL.Image

from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

# Check doc strings for more information
seg_net = TracerUniversalB7(device='cuda',
                            batch_size=1)


preprocessing = PreprocessingStub()

interface = Interface(pre_pipe=preprocessing,
                      post_pipe=None,
                      seg_pipe=seg_net)

import glob
import os
for image_path in glob.glob(r"data/images/*.jpg"):
    if not os.path.isfile(r'data/images_without_bg/'+image_path.split("\\")[-1][:-3]+"png"):
        print(image_path)
        image = PIL.Image.open(image_path)
        image = image.convert("RGB")  # remove alpha
        cat_wo_bg = interface([image])[0]
        # cat_wo_bg = cat_wo_bg.convert('RGB')
        try:
            cat_wo_bg.save(r'data/images_without_bg/'+image_path.split("\\")[-1][:-3]+"png")
        except Exception as err:
            print(err)