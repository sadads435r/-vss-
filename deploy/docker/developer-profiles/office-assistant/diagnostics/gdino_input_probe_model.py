import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.saved = False

    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            data = tensor.as_numpy()
            if not self.saved:
                path = "/tmp/gdino_live_input.npy"
                np.save(path, data)
                channels = np.moveaxis(data[0], -1, 0) if data.ndim == 4 else np.moveaxis(data, -1, 0)
                stats = []
                for index, channel in enumerate(channels):
                    stats.append(
                        "c%d[min=%.6f max=%.6f mean=%.6f std=%.6f]"
                        % (index, channel.min(), channel.max(), channel.mean(), channel.std())
                    )
                print("GDINO_LIVE_INPUT shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f std=%.6f %s saved=%s"
                      % (data.shape, data.dtype, data.min(), data.max(), data.mean(), data.std(), " ".join(stats), path),
                      flush=True)
                self.saved = True
            nchw = np.ascontiguousarray(np.moveaxis(data, -1, 1))
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUTPUT", nchw)]
            ))
        return responses
