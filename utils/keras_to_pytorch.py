"""
Given an instance keras model, creates an associated PyTorch model architecture in a .py file and save a PyTorch instance in a .pt file

Useful documentation used:
    - https://stackoverflow.com/questions/53942291/what-does-the-00-of-the-layers-connected-to-in-keras-model-summary-mean
    - https://stackoverflow.com/questions/50814880/keras-retrieve-layers-that-the-layer-connected-to

"""
__author__ = "Axel ROCHEL"


from collections import OrderedDict
from keras.models import load_model

import torch

model_name = 'model_test'

model = load_model('{}.net'.format(model_name))
model.load_weights('{}.net'.format(model_name))

activations = {'relu': 'nn.ReLU'}

def to_str(x):
    """
    Convert the given input into string.
    If the given input is a string, add apostrophe at the beginning and the end of the string.
    """
    if isinstance(x, str):
        return "'{}'".format(x)
    return str(x)

def is_older_version(v1, v2):
    """
    Test wether v1 (string) is older than v2 (string).
    """
    v1_tuple = tuple((int(digit) for digit in v1.split('.')))
    v2_tuple = tuple((int(digit) for digit in v2.split('.')))
    return v1_tuple < v2_tuple

def get_padding(layer):
    """
    Return PyTorch padding equivalent to the one in the given tensorlow layer.
    See:
        https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/3
        https://github.com/pytorch/pytorch/issues/3867
    """
    if layer.padding == 'same':
        in_height, in_width, _ = layer.input_shape[1:] #potentiellement MAuvais ordre
        
        if 'kernel_size' in dir(layer):
            filter_height, filter_width = layer.kernel_size
        else:
            filter_height, filter_width = layer.pool_size

        if (in_height % layer.strides[0] == 0):
            pad_along_height = max(filter_height - layer.strides[0], 0)
        else:
            pad_along_height = max(filter_height - (in_height % layer.strides[0]), 0)
        if (in_width % layer.strides[1] == 0):
            pad_along_width = max(filter_width - layer.strides[1], 0)
        else:
            pad_along_width = max(filter_width - (in_width % layer.strides[1]), 0)
        
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        if pad_top == pad_bottom and pad_left == pad_right:
            return (pad_left, pad_top)
        else:
            return (pad_left, pad_right, pad_top, pad_bottom)

    elif layer.padding == 'valid':
        return 0
    else:
        raise NotImplementedError("Unknown padding:", layer.padding)


def get_args(layer):
    """
    Return the arguments that should be used to recreate the given keras layer into a PyTorch Layer.

    Parameters:
        - layer (keras.layers.Layer): The given layer instance.

    Returns:
        - class_name (string): The name of the associated class in PyTorch.
        - args (List): List of non-keyworded arguments.
        - kwargs (OrderedDict[string, ...]): Ordered dictionnary of keyworded arguments.
    """
    args = []
    kwargs = OrderedDict()
    if layer.__class__.__name__ == 'Dense':
        class_name = 'nn.Linear'
        args.append(layer.get_weights()[0].shape[0])
        args.append(layer.units)
        if not layer.use_bias:
            kwargs['bias'] = layer.use_bias
        if layer.activation.__name__ != 'linear':
            raise NotImplementedError("Activation specified in the layer arguments.")
    elif layer.__class__.__name__ == 'Conv2D':
        class_name = 'nn.Conv2d'
        args.append(layer.get_weights()[0].shape[2])
        args.append(layer.filters)
        kwargs['kernel_size'] = layer.kernel_size[::-1]
        kwargs['stride'] = layer.strides[::-1]
        kwargs['padding'] = get_padding(layer)
        # kwargs['dilation'] = layer.kernel_size[::-1]
        if not layer.use_bias:
            kwargs['bias'] = layer.use_bias
        if layer.activation.__name__ != 'linear':
            raise NotImplementedError("Activation specified in the layer arguments: {}.".format(layer.activation.__name__))
    elif layer.__class__.__name__ == 'Activation':
        class_name = activations[layer.activation.__name__]
    elif layer.__class__.__name__ == 'LeakyReLU':
        class_name = 'nn.LeakyReLU'
        args.append(layer.alpha)
    elif layer.__class__.__name__ == 'UpSampling2D':
        class_name = 'nn.Upsample'
        kwargs['scale_factor'] = layer.size[::-1]
        kwargs['mode'] = layer.interpolation
    elif layer.__class__.__name__ == 'BatchNormalization':
        class_name = 'nn.BatchNorm2d'
        args.append(list(layer.input_spec.axes.values())[0])
        kwargs['eps'] =  layer.epsilon
        kwargs['momentum'] = layer.momentum
    elif layer.__class__.__name__ == 'Conv2DTranspose':
        class_name = 'nn.ConvTranspose2d'
        args.append(layer.get_weights()[0].shape[3])
        args.append(layer.filters)
        kwargs['kernel_size'] = layer.kernel_size[::-1]
        kwargs['stride'] = layer.strides[::-1]
        kwargs['padding'] = get_padding(layer)
        # kwargs['output_padding'] = layer.output_padding
    elif layer.__class__.__name__ == 'ZeroPadding2D':
        class_name = 'nn.ZeroPad2d'
        if isinstance(layer.padding, int):
            args.append(layer.padding)
        else: 
            pad_y, pad_x = layer.padding
            if isinstance(pad_x, int):
                args.append((pad_y, pad_y, pad_x, pad_x))
            else:
                args.append((pad_y[0], pad_y[1], pad_x[0], pad_x[1])) # Normally (pad_x[0], pad_x[1], pad_y[0], pad_y[1]) but here picture is transposed
    elif layer.__class__.__name__ == 'MaxPooling2D':
        class_name = 'nn.MaxPool2d'
        kwargs['kernel_size'] = layer.pool_size[::-1]
        kwargs['stride'] = layer.strides[::-1]
        kwargs['padding'] = get_padding(layer)
    elif layer.__class__.__name__ == 'AveragePooling2D':
        class_name = 'nn.AvgPool2d'
        kwargs['kernel_size'] = layer.pool_size[::-1]
        kwargs['stride'] = layer.strides[::-1]
        kwargs['padding'] = get_padding(layer)
    else:
        raise NotImplementedError("Unknown layer class: {}.".format(layer.__class__.__name__))
    
    return class_name, args, kwargs

init_layers = ['Dense', 'Conv2D', 'Activation', 'LeakyReLU', 'UpSampling2D', 'BatchNormalization', 'Conv2DTranspose', 'ZeroPadding2D', 'MaxPooling2D', 'AveragePooling2D']
# other : 'InputLayer': None, 'Reshape': None, 'Add': None, 'Concatenate': None

init_layer_lines = []
forward_lines = []
done = []
weight_layer_indexes = dict()

for layer in model.layers:
    if layer.name in done:
        continue
    weight_layer_indexes[layer.name] = None

    if layer.__class__.__name__ == 'InputLayer':
        continue
    elif layer.__class__.__name__ == 'Reshape':
        previous = layer._inbound_nodes[0].inbound_layers[0]
        forward_lines.append("{} = {}.view({})".format(layer.name, previous.name, (-1,) + layer.target_shape[::-1]))
    elif layer.__class__.__name__ == 'Add':
        previous_names = [prev.name for prev in layer._inbound_nodes[0].inbound_layers]
        forward_lines.append("{} = {} + {}".format(layer.name, *previous_names))
    elif layer.__class__.__name__ == 'Concatenate':
        axis = layer.axis
        if axis == -1:
            axis = 1
        previous_names = [prev.name for prev in layer._inbound_nodes[0].inbound_layers]
        forward_lines.append("{} = torch.cat(({}), dim={})".format(layer.name, ', '.join(previous_names), axis))
    elif layer.__class__.__name__ in init_layers:
        
        init_layer_lines.append([])

        previous = layer._inbound_nodes[0].inbound_layers[0]
        current_layer = layer
        while current_layer.__class__.__name__ in init_layers:
            init_args = get_args(current_layer)
            ### Deal with 4-tuple padding
            _, _, kwargs = init_args
            if 'padding' in kwargs:
                if isinstance(kwargs['padding'], tuple):
                    if len(kwargs['padding']) == 4:
                        init_layer_lines[-1].append(('nn.ZeroPad2d', [kwargs['padding']], OrderedDict()))
                        del init_args[2]['padding']
            ###
            init_layer_lines[-1].append(init_args)

            weight_layer_indexes[current_layer.name] = None

            if len(current_layer.weights) == 1:
                weight_layer_indexes[current_layer.name] = (len(init_layer_lines), len(init_layer_lines[-1]) - 1, False)
            elif len(current_layer.weights) >= 2:
                weight_layer_indexes[current_layer.name] = (len(init_layer_lines), len(init_layer_lines[-1]) - 1, True)

            done.append(current_layer.name)
            if len(current_layer._outbound_nodes) != 1 or current_layer.name in model.output_names:
                break
            current_layer = current_layer._outbound_nodes[0].outbound_layer
        # if len(init_layer_lines[-1]) == 0:
        #     # This layer output is used in several other layers
        #     init_layer_lines[-1].append(get_args(current_layer))
        #     done.append(current_layer.name)
        forward_lines.append("{} = self.block_{}({})".format(done[-1], len(init_layer_lines), previous.name))
    else:
        raise NotImplementedError("Unknown layer class: {}.".format(layer.__class__.__name__))


with open('{}_torch.py'.format(model_name), 'w') as f:
    f.write("import torch\n")
    f.write("import torch.nn as nn\n")
    f.write("\n")
    f.write("class {}Model(nn.Module):\n".format(model_name.title()))
    f.write("\t\n")
    f.write("\tdef __init__(self):\n")
    f.write("\t\tsuper({}Model, self).__init__()\n".format(model_name.title()))
    f.write("\t\t\n")
    for i, block in enumerate(init_layer_lines):
        f.write("\t\tself.block_{} = nn.Sequential(\n".format(i+1))
        # f.write("\t\tself.block_{} = nn.ModuleList([\n".format(i+1))
        layers_str = []
        for class_name, args, kwargs in block:
            args_kwargs = list(map(to_str , args)) + ['='.join([key, to_str(val)]) for (key,val) in kwargs.items()]
            layers_str.append("\t\t\t{}({})".format(class_name, ', '.join(args_kwargs)))
        f.write(',\n'.join(layers_str))
        f.write("\n\t\t)\n")
    f.write("\t\n")
    f.write("\tdef forward(self, {}):\n".format(', '.join(model.input_names)))
    for line in forward_lines:
        f.write("\t\t{}\n".format(line))   
    f.write("\t\t\n")
    f.write("\t\treturn {}\n".format(', '.join(model.output_names)))


exec("from {}_torch import {}Model as TorchModel".format(model_name, model_name.title()))

weights = model.get_weights()
torch_model = TorchModel()
for layer in model.layers:
    if weight_layer_indexes[layer.name] is not None:
        i_block, i_layer, has_bias = weight_layer_indexes[layer.name]
        torch_layer = getattr(torch_model, 'block_{}'.format(i_block))[i_layer]
        layer_weights = layer.get_weights()

        if layer_weights[0].T.shape != torch_layer.weight.data.shape:
            raise ValueError("Differents shape of weights in the two model. Given: {}. Expected: {}".format(torch_layer.weight.data.shape, layer_weights[0].T.shape))
        
        torch_layer.weight.data = torch.from_numpy(layer_weights[0].T)

        if has_bias:
            torch_layer.bias.data = torch.from_numpy(layer_weights[1])
        if layer.__class__.__name__ == "BatchNormalization":
            torch_layer.running_mean = torch.from_numpy(layer_weights[2])
            torch_layer.running_var = torch.from_numpy(layer_weights[3])

torch.save(torch_model, '{}.pt'.format(model_name))
print('Model saved')