"""Restore the released checkpoint for frozen inference; no training entry point."""
import copy
import gc
import inspect
from pathlib import Path


MODEL_OPTIONS = (
    'cnn_variant', 'csi_mode', 'head_mode', 'score_mode', 'score_scope',
    'transfer_weight', 'graph_mode',
)


def saved(config, key, required=True):
    values = []
    for block in (config, config.get('args', {}), config.get('model_kwargs', {})):
        if isinstance(block, dict) and block.get(key) is not None:
            values.append(block[key])
    if not values:
        if required:
            raise ValueError('Checkpoint does not record ' + key)
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError('Conflicting checkpoint setting: ' + key)
    return copy.deepcopy(values[0])


def parameter_counts(model, local):
    """Count unique parameter objects before freezing; exclude the unused decoder."""
    decoder = {id(p) for p in model.decoder.parameters()}
    seen, total, learned, excluded = set(), 0, 0, 0
    for network in (model, local):
        for parameter in network.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            count = parameter.numel()
            if id(parameter) in decoder:
                excluded += count
            else:
                total += count
                if parameter.requires_grad:
                    learned += count
    return {
        'inference_registered': total,
        'inference_learnable_before_freeze': learned,
        'fixed_parameters': total - learned,
        'training_decoder_excluded': excluded,
    }


class FrozenInference:
    def __init__(self, model, local, graph, metadata, torch):
        self.model, self.local, self.torch = model, local, torch
        self.graph, self.metadata = graph, metadata
        self._counter = int(local.iter_num)
        self._graph = graph.clone()
        self._buffers = [value.detach().clone() for network in (model, local)
                         for value in network.buffers()]

    def __call__(self, images):
        with self.torch.no_grad():
            return self.model(
                x_s=images, ad_net=self.local, cp_mask=self.graph,
                optimal_flag=self.metadata['optimal'], num_prop=self.metadata['H'],
                gtp_layers=self.metadata['gtp_layers'],
            )[0]

    def assert_frozen_state(self):
        if self.model.training or self.local.training or int(self.local.iter_num) != self._counter:
            raise RuntimeError('The model must remain in frozen evaluation mode')
        buffers = [value for network in (self.model, self.local) for value in network.buffers()]
        if len(buffers) != len(self._buffers) or any(
                not self.torch.equal(a, b) for a, b in zip(buffers, self._buffers)):
            raise RuntimeError('Evaluation changed a model buffer')
        if not self.torch.equal(self.graph, self._graph):
            raise RuntimeError('Evaluation changed the saved graph')


def load_model(checkpoint, device='cpu'):
    import torch
    from models import modeling

    if Path(modeling.__file__).resolve().parent != Path(__file__).resolve().parent / 'models':
        raise RuntimeError('A different models package was imported; run in a fresh Python process')
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError('Checkpoint not found: ' + str(checkpoint))
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        # The author's original final.pt also contains Python/NumPy objects.
        options['weights_only'] = False
    payload = torch.load(str(checkpoint), **options)
    required = {'model', 'ad_net_local', 'cp_mask', 'config'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('Use the released checkpoint containing ' + ', '.join(sorted(required)))
    state = {name: payload[name] for name in required}
    state['ad_iter_num'] = payload.get('ad_iter_num', {})
    del payload
    gc.collect()
    config = state['config']
    if not isinstance(config, dict):
        raise ValueError('Checkpoint configuration must be a dictionary')
    model_type = saved(config, 'model_type')
    if model_type not in modeling.CONFIGS:
        raise ValueError('This model package does not contain architecture ' + str(model_type))
    image_size, classes = saved(config, 'img_size'), saved(config, 'num_classes')
    batch, H, msa = (saved(config, name) for name in ('eval_batch_size', 'num_prop', 'msa_layer'))
    optimal = saved(config, 'optimal')
    if any(type(value) is not int or value < 1 for value in (image_size, classes, batch, msa)):
        raise ValueError('Invalid saved image size, class count, batch size, or attention layer')
    if type(H) is not int or H < 0 or optimal not in (0, 1):
        raise ValueError('Invalid saved merging settings')
    kwargs = {name: saved(config, name) for name in MODEL_OPTIONS}
    vit = copy.deepcopy(modeling.CONFIGS[model_type])
    model = modeling.VisionTransformer(
        vit, img_size=image_size, num_classes=classes, zero_head=False, vis=False,
        msa_layer=msa, score_audit=False, **kwargs)
    model.load_state_dict(state['model'], strict=True)
    width = int(vit.hidden_size) // int(vit.transformer['num_heads'])
    local = modeling.AdversarialNetwork(width, width)
    local.load_state_dict(state['ad_net_local'], strict=True)
    local.iter_num = int(state['ad_iter_num'].get('local', 0))
    graph = state['cp_mask']
    tokens = int(model.transformer.embeddings.position_embeddings.shape[1])
    if not isinstance(graph, torch.Tensor) or tuple(graph.shape) != (tokens, tokens):
        raise ValueError('The saved graph does not match the original patch grid')
    if not torch.is_floating_point(graph) or not bool(torch.isfinite(graph).all()):
        raise ValueError('The saved graph must contain finite floating-point values')
    depth = len(model.transformer.encoder.layer)
    layers = saved(config, 'gtp_layers', required=False)
    if layers is None:
        layers = list(range(saved(config, 'gtp_start_layer'), depth)) if H else []
    if (not isinstance(layers, list) or any(type(i) is not int or not 0 <= i < depth for i in layers)
            or layers != sorted(set(layers)) or not 1 <= msa <= depth or H >= tokens - 1):
        raise ValueError('Invalid saved attention/merging layers')
    metadata = {
        'model_type': model_type, 'image_size': image_size, 'num_classes': classes,
        'eval_batch_size': batch, 'H': H, 'optimal': optimal, 'gtp_layers': layers,
        'parameter_counts': parameter_counts(model, local),
    }
    graph = graph.detach().clone().float()
    del state
    gc.collect()
    model = model.to(device).float().eval()
    local = local.to(device).float().eval()
    for network in (model, local):
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    wrapper = FrozenInference(model, local, graph.to(device), metadata, torch)
    wrapper.assert_frozen_state()
    return wrapper, metadata
