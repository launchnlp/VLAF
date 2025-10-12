from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

cache_dir = HUGGINGFACE_HUB_CACHE

# Scan the cache directory for all cached models
cache_info = scan_cache_dir(cache_dir)

# List all model repo_ids in the cache
repo_ids = [repo.repo_id for repo in cache_info.repos]
for repo_id in sorted(repo_ids):
    print(repo_id)