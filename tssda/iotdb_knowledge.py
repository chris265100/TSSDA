"""Load the framework's tree-model SQL guidance."""
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def load_non_function_iotdb_knowledge() -> str:
    """Load SQL syntax guidance; P4 supplies the typed function catalog."""
    return files("tssda").joinpath("knowledge/iotdb_tree_sql.md").read_text(encoding="utf-8").rstrip()
