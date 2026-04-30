# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-ThirdParty-Unlicensed

import cuda_extension
import torch.nn as nn


class ModelNew(nn.Module):

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return cuda_extension.axpby_forward(a, b, self.alpha, 0)
