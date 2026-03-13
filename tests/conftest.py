import os

# test_server_integration.py is a standalone script for running against a live
# Docker container — it uses module-level HTTP calls, not pytest fixtures.
collect_ignore = [os.path.join(os.path.dirname(__file__), "test_server_integration.py")]
