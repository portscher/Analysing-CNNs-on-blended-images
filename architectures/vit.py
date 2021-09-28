import os

import torch

from .model import Model
from architectures.vision_transformer.vit_helpers import load_model_config_from_hf, is_model, model_entrypoint, load_checkpoint
from architectures.vision_transformer.vit_layers import set_layer_config


class ViT(Model):

    def __init__(self, train_from_scratch=True, path=None, attention='none'):
        super().__init__(path, train_from_scratch, attention)
        self.path = path
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        model = create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=8,
            drop_rate=0.0, attention='none')  # vision_transformer has attention per default

        if not self.train_from_scratch and os.path.isfile(self.path):
            print("Loading ViT from disk")
            model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu'))['state_dict'])
            model.eval()

        return model


def split_model_name(model_name):
    model_split = model_name.split(':', 1)
    if len(model_split) == 1:
        return '', model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ('timm', 'hf_hub')
        return source_name, model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')

    if remove_source:
        model_name = split_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
        model_name,
        pretrained=False,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model
    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)
    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    source_name, model_name = split_model_name(model_name)

    kwargs.pop('bn_tf', None)
    kwargs.pop('bn_momentum', None)
    kwargs.pop('bn_eps', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if source_name == 'hf_hub':
        # For model names specified in the form `hf_hub:path/architecture_name#revision`,
        # load model weights + default_cfg from Hugging Face hub.
        hf_default_cfg, model_name = load_model_config_from_hf(model_name)
        kwargs['external_default_cfg'] = hf_default_cfg  # FIXME revamp default_cfg interface someday

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(pretrained=pretrained, **kwargs)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
