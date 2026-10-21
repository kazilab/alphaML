from .specs import MODELS, SPECS, ModelSpec

__all__ = ['MODELS', 'SPECS', 'ModelSpec', 'get_spec']


def get_spec(model_name):
    try:
        return MODELS[model_name]
    except KeyError:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {', '.join(MODELS)}") from None
