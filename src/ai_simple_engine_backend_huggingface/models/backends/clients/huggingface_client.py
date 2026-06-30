from huggingface_hub import snapshot_download
from pathlib import Path
from typing import Union


class HuggingfaceClient:

    def download(
        self,
        repo_id: str,
        *,
        revision: Union[str, None] = None,
        local_dir: Union[Path, None] = None
    ) -> Path:
        """
        Download from the given `repo_id` and get 
        where it's been downloaded as a `Path`.
        """
        path = snapshot_download(
            repo_id = repo_id,
            revision = revision,
            local_dir = local_dir,
            local_dir_use_symlinks = False
        )

        return Path(path)