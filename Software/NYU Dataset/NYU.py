# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NYU-Depth V2."""


import io

import datasets
import h5py
import numpy as np

_CITATION = """\
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
@inproceedings{icra_2019_fastdepth,
  author    = {Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne},
  title     = {FastDepth: Fast Monocular Depth Estimation on Embedded Systems},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2019}
}
"""

_DESCRIPTION = """\
The NYU-Depth V2 data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
"""

_HOMEPAGE = "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"

_LICENSE = "Apace 2.0 License"

_URLS = {
    "train": [f"train/data-{i:05d}-of-00157.arrow" for i in range(157)],
    "val": [f"validation/data-{i:05d}-of-00003.arrow" for i in range(3)],
}

_IMG_EXTENSIONS = [".h5"]


class NYUDepthV2(datasets.GeneratorBasedBuilder):
    """NYU-Depth V2 dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {"image": datasets.Image(), "depth_map": datasets.Image()}
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _is_image_file(self, filename):
        # Reference: https://github.com/dwofk/fast-depth/blob/master/dataloaders/dataloader.py#L21-L23
        return any(filename.endswith(extension) for extension in _IMG_EXTENSIONS)

    def _h5_loader(self, bytes_stream):
        # Reference: https://github.com/dwofk/fast-depth/blob/master/dataloaders/dataloader.py#L8-L13
        f = io.BytesIO(bytes_stream)
        h5f = h5py.File(f, "r")
        rgb = np.array(h5f["rgb"])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f["depth"])
        return rgb, depth

    def _split_generators(self, dl_manager):
        archives = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive) for archive in archives["train"]
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archives": [
                        dl_manager.iter_archive(archive) for archive in archives["val"]
                    ]
                },
            ),
        ]

    def _generate_examples(self, archives):
        idx = 0
        for archive in archives:
            for path, file in archive:
                if self._is_image_file(path):
                    image, depth = self._h5_loader(file.read())
                    yield idx, {"image": image, "depth_map": depth}
                    idx += 1