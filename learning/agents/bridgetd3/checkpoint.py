# Copyright 2025 The Brax Authors.
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

"""Checkpointing for BridgeTD3."""

from typing import Any, Union

from brax.training import checkpoint
from brax.training import types
from agents.bridgetd3 import networks as bridgetd3_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = "bridgetd3_network_config.json"


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(path: Union[str, epath.Path]):
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[bridgetd3_networks.BridgeTd3Networks],
) -> config_dict.ConfigDict:
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )

