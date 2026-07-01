from ai_simple_engine_backend_huggingface.models.backends.clients.huggingface_client import HuggingfaceClient
from ai_simple_engine_backend_huggingface.consts import HUGGINGFACE_MODEL_BACKEND_PROVIDER
from ai_simple_engine.models.installed_model import InstalledModel
from ai_simple_engine.models.backends.abstract import ModelBackend
from ai_simple_engine.plugins.plugin_context import PluginContext
from ai_simple_engine.models.spec.base import ModelSpec
from typing import Union
from pathlib import Path


class HuggingfaceBackend(
    ModelBackend
):
    
    @property
    def provider(
        self
    ) -> str:
        return HUGGINGFACE_MODEL_BACKEND_PROVIDER

    def __init__(
        self,
        client: Union[HuggingfaceClient, None] = None,
    ):
        self._client = (
            client or
            HuggingfaceClient()
        )

    def configure(
        self,
        context: PluginContext
    ) -> None:
        self._settings = context.settings

    async def install(
        self,
        spec: ModelSpec
    ) -> InstalledModel:
        path = self._local_directory(spec)

        if not path.exists():
            path = self._client.download(
                repo_id = spec.identifier,
                revision = spec.revision,
                local_dir = path,
            )

        return InstalledModel(
            spec = spec,
            path = path,
        )

    def _local_directory(
        self,
        spec: ModelSpec,
    ) -> Path:
        # TODO: Create a PathResolver to do this
        path = (
            self._settings.models_directory
            / spec.family
            / spec.identifier
        )

        if spec.revision:
            path /= spec.revision

        return path