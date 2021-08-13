# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following license:

# Copyright 2020 Center for Biomedical Image Computing and Analytics, University of Pennsylvania
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# This is a 3-clause BSD license as defined in https://opensource.org/licenses/BSD-3-Clause
import os
import hashlib
import numpy as np

import torch.nn as nn
import torch

from seg_modules import in_conv, DownsamplingModule, EncodingModule, InceptionModule, \
    ResNetModule
from seg_modules import UpsamplingModule, DecodingModule, IncDownsamplingModule, IncConv
from seg_modules import out_conv, FCNUpsamplingModule, IncDropout, IncUpsamplingModule

from utils.collaborator_aggregator_interface_pb2 import TensorProto, ExtraModelInfo, ModelHeader


def load_model(directory):
    model_header = load_proto(os.path.join(directory, 'ModelHeader.pbuf'), proto_type=ModelHeader)
    extra_model_info = load_proto(os.path.join(directory, 'ExtraModelInfo.pbuf'), proto_type=ExtraModelInfo)
    model_tensors = extra_model_info.tensor_names
    tensor_dict_from_proto = {}
    for t in extra_model_info.tensor_names:
        t_hash = hash_string(t)
        tensor_proto = load_proto(os.path.join(directory, '{}.pbuf'.format(t_hash)), proto_type=TensorProto)
        if t != tensor_proto.name:
            raise RuntimeError(
                "Loaded the wrong tensor! Meant to load: {} did load: {} read file: {}".format(t, tensor_proto.name,
                                                                                               t_hash))
        tensor_dict_from_proto[t] = tensor_proto_to_numpy_array(tensor_proto)
    return tensor_dict_from_proto


def tensor_proto_to_numpy_array(tensor_proto):
    return np.frombuffer(tensor_proto.data_bytes, dtype=np.float32).reshape(tuple(tensor_proto.shape))


def hash_string(s, length=8):
    md5 = hashlib.md5()
    md5.update(s.encode())
    return md5.hexdigest()[:length]


def load_proto(fpath, proto_type):
    """Load the protobuf
    Args:
        fpath: The filepath for the protobuf
    Returns:
        protobuf: A protobuf of the model
    """
    with open(fpath, "rb") as f:
        return proto_type().FromString(f.read())


"""
The smaller individual modules of these networks (the ones defined below), are taken from the seg_modules files as seen in the imports above.

This is the standard U-Net architecture : https://arxiv.org/pdf/1606.06650.pdf. The Downsampling, Encoding, Decoding modules
are defined in the seg_modules file. These smaller modules are basically defined by 2 parameters, the input channels (filters) and the output channels (filters),
and some other hyperparameters, which remain constant all the modules. For more details on the smaller modules please have a look at the seg_modules file.
"""


class PyTorch3DResUNet(nn.Module):
    def __init__(self, final_layer_activation = 'sigmoid', sigmoid_input_multiplier=10.0, **kwargs):

        super().__init__()

        # val_input_shape = [batch_size, n_channels, 240, 240, 155]
        # val_output_shape = [batch_size, len(label_channels), 240, 240, 155]

        self.n_channels = 4
        self.base_filters = 30  # or 16
        self.label_channels = 3
        self.device = 'cuda'


        self.init_network(self.device,
                          final_layer_activation=final_layer_activation,
                          sigmoid_input_multiplier=sigmoid_input_multiplier)



    def init_network(self, device, print_model=False, final_layer_activation='softmax', sigmoid_input_multiplier=1.0,
                     **kwargs):
        self.ins = in_conv(self.n_channels, self.base_filters, res=True)
        self.ds_0 = DownsamplingModule(self.base_filters, self.base_filters * 2)
        self.en_1 = EncodingModule(self.base_filters * 2, self.base_filters * 2, res=True)
        self.ds_1 = DownsamplingModule(self.base_filters * 2, self.base_filters * 4)
        self.en_2 = EncodingModule(self.base_filters * 4, self.base_filters * 4, res=True)
        self.ds_2 = DownsamplingModule(self.base_filters * 4, self.base_filters * 8)
        self.en_3 = EncodingModule(self.base_filters * 8, self.base_filters * 8, res=True)
        self.ds_3 = DownsamplingModule(self.base_filters * 8, self.base_filters * 16)
        self.en_4 = EncodingModule(self.base_filters * 16, self.base_filters * 16, res=True)
        self.us_3 = UpsamplingModule(self.base_filters * 16, self.base_filters * 8)
        self.de_3 = DecodingModule(self.base_filters * 16, self.base_filters * 8, res=True)
        self.us_2 = UpsamplingModule(self.base_filters * 8, self.base_filters * 4)
        self.de_2 = DecodingModule(self.base_filters * 8, self.base_filters * 4, res=True)
        self.us_1 = UpsamplingModule(self.base_filters * 4, self.base_filters * 2)
        self.de_1 = DecodingModule(self.base_filters * 4, self.base_filters * 2, res=True)
        self.us_0 = UpsamplingModule(self.base_filters * 2, self.base_filters)
        self.out = out_conv(self.base_filters * 2,
                            self.label_channels,
                            res=True,
                            activation=final_layer_activation,
                            sigmoid_input_multiplier=sigmoid_input_multiplier)

        if print_model:
            print(self)

        # send this to the device
        self.to(device)

    def forward(self, x):
        # normalize input if can do so without producing nan values

        if (torch.isnan(torch.std(x)).cpu().item() != True) and (torch.std(x).cpu().item() != 0.0):
            x = (x - torch.mean(x)) / torch.std(x)
        else:
            self.logger.debug("Skipping input normalization due to std val of: {}.".format(torch.std(x).cpu().item()))
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x

    def set_tensor_dict(self, tensor_dict):
        """Set the tensor dictionary
        Args:
            tensor_dict: The tensor dictionary
            with_opt_vars (bool): Return the tensor dictionary including the optimizer tensors (Default=False)
        """
        # Sets tensors for model layers and optimizer state.
        # FIXME: self.parameters() instead? Unclear if load_state_dict() or simple assignment is better
        # for now, state dict gives us names, which is good
        # FIXME: do both and sanity check each time?

        # get device for correct placement of tensors
        device = self.device

        new_state = {}
        for k in tensor_dict.keys():
            new_state[k] = torch.from_numpy(np.copy(tensor_dict[k])).to(device)

        # set model state
        self.load_state_dict(new_state)
        print("State_dict_loaded")


if __name__ == '__main__':
    model = PyTorch3DResUNet()
    state_dict = load_model("initial")
    model.set_tensor_dict(state_dict)
