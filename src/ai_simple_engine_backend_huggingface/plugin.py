from ai_simple_engine_backend_huggingface.models.backends.huggingface_backend import HuggingfaceBackend
from ai_simple_engine.engine_builder import EngineBuilder
from ai_simple_engine.plugins.plugin import Plugin


class HuggingfaceBackendPlugin(
    Plugin
):
    """
    The plugin to add the Huggingface platform as
    a backend to be able to download the models.

    This plugin includes:
    - `HuggingfaceBackend`
    """

    def register(
        self,
        builder: EngineBuilder
    ):
        # Model backends
        (
            builder
            .add_model_backend(HuggingfaceBackend())
        )