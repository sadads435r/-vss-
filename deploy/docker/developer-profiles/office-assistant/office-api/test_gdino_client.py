from __future__ import annotations

import json
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from gdino_client import GroundingDinoClient


class GroundingDinoClientTests(unittest.TestCase):
    def test_preprocess_letterboxes_to_model_shape(self) -> None:
        source = BytesIO()
        Image.new("RGB", (320, 640), "white").save(source, format="JPEG")
        tensor, transform = GroundingDinoClient.preprocess(source.getvalue())
        self.assertEqual(tensor.shape, (1, 3, 544, 960))
        self.assertGreater(transform["left"], 0)
        self.assertEqual(transform["top"], 2)

    def test_binary_response_parser(self) -> None:
        scores = np.asarray([[0.8, 0.2]], dtype=np.float32)
        boxes = np.asarray([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
        labels = np.asarray([[0, 1]], dtype=np.int32)
        arrays = [("scores", scores, "FP32"), ("boxes", boxes, "FP32"), ("labels", labels, "INT32")]
        metadata = {"outputs": [
            {
                "name": name, "shape": list(array.shape), "datatype": datatype,
                "parameters": {"binary_data_size": array.nbytes},
            }
            for name, array, datatype in arrays
        ]}
        header = json.dumps(metadata).encode("utf-8")
        payload = header + b"".join(array.tobytes() for _, array, _ in arrays)
        parsed = GroundingDinoClient.parse_response(payload, len(header))
        self.assertAlmostEqual(float(parsed["scores"][0, 0]), 0.8, places=5)
        self.assertEqual(parsed["labels"].tolist(), [[0, 1]])
        self.assertEqual(parsed["boxes"].shape, (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
