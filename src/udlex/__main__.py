"""Allow ``python -m udlex`` to reach the command line interface."""

from udlex.cli import main

raise SystemExit(main())
