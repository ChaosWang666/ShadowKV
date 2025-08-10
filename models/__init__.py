################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

"""模型入口模块，提供不同模型类的统一导出与选择逻辑"""

from .glm import GLM
from .llama import Llama
from .qwen import Qwen2
from .phi3 import Phi3


def choose_model_class(model_name):
    """根据给定名称选择并返回对应的模型类。

    参数:
        model_name (str): 模型名称或别名，大小写不敏感。

    返回:
        type: 具体模型的类对象。
    """

    name = model_name.lower()
    if 'llama' in name:
        return Llama
    if 'glm' in name:
        return GLM
    if 'yi' in name:
        return Llama
    if 'qwen' in name:
        return Qwen2
    if 'phi' in name:
        return Phi3
    raise ValueError(f"Model {model_name} not found")

