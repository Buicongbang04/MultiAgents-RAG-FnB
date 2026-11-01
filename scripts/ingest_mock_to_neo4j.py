"""DEPRECATED: Postgres is now the source of truth. Use `python -m scripts.bootstrap`."""

import sys

from scripts.bootstrap import main

if __name__ == "__main__":
    print("scripts.ingest_mock_to_neo4j is deprecated; running scripts.bootstrap instead.", file=sys.stderr)
    main()
