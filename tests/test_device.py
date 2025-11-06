import torch


class TestDevice:

    def test_gpu_is_present(self):
        assert torch.cuda.is_available() == True