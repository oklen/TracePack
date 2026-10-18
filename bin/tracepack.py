#!/usr/bin/env python3
"""TracePack launcher used by the Claude Code plugin: hooks, the MCP server and the CLI.

    python3 "${CLAUDE_PLUGIN_ROOT}/bin/tracepack.py" <command> ...

It runs the copy of TracePack that ships with the plugin (no pip install, no dependencies).
"""
import os
import sys

if sys.version_info < (3, 9):
    sys.stderr.write("TracePack needs Python 3.9 or newer (found %d.%d).\n" % sys.version_info[:2])
    sys.exit(0 if sys.argv[1:2] == ["hook"] else 1)    # a hook fails open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracepack.cli import main  # noqa: E402

sys.exit(main())
