"""Entry point for the DagFlow CLI.

Provides the main argument parser and dispatches to subcommands.
Can be invoked as: python -m dagflow.cli <command>

This module depends on:
- cli.commands (InspectCommand, ValidateCommand, ExportCommand, RunCommand)
- cli.formatters (TableFormatter, JsonFormatter, TreeFormatter)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from dagflow.cli.commands import (
    ExportCommand,
    InspectCommand,
    RunCommand,
    ValidateCommand,
)
from dagflow.cli.formatters import FormatKind
from dagflow import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands.

    Returns:
        Configured ArgumentParser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="dagflow",
        description="DagFlow — Incremental computation engine CLI",
        epilog="Use 'dagflow <command> --help' for command-specific help.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"dagflow {__version__}",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "tree"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect graph structure and node details",
    )
    inspect_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Node ID to inspect (omit for full graph overview)",
    )
    inspect_parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Depth of dependency traversal (default: 1)",
    )
    inspect_parser.add_argument(
        "--show-values",
        action="store_true",
        help="Include cached values in output",
    )

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate graph integrity and constraints",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation (warnings become errors)",
    )
    validate_parser.add_argument(
        "--check-cycles",
        action="store_true",
        default=True,
        help="Check for cycles (default: True)",
    )
    validate_parser.add_argument(
        "--check-orphans",
        action="store_true",
        help="Check for orphaned nodes",
    )

    # export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Export graph in various formats",
    )
    export_parser.add_argument(
        "output",
        help="Output file path",
    )
    export_parser.add_argument(
        "--format",
        dest="export_format",
        choices=["dot", "json", "mermaid"],
        default="json",
        help="Export format (default: json)",
    )
    export_parser.add_argument(
        "--include-values",
        action="store_true",
        help="Include node values in export",
    )

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run computation with specified inputs",
    )
    run_parser.add_argument(
        "--input", "-i",
        action="append",
        default=[],
        help="Input values as node_id=value (can repeat)",
    )
    run_parser.add_argument(
        "--output", "-o",
        action="append",
        default=[],
        help="Output node IDs to display (default: all)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be computed without executing",
    )

    return parser


def parse_input_values(input_args: List[str]) -> dict:
    """Parse input arguments in the form 'node_id=value'.

    Args:
        input_args: List of 'key=value' strings.

    Returns:
        Dict of {node_id: parsed_value}.

    Raises:
        ValueError: If an argument is malformed.
    """
    values: dict = {}
    for arg in input_args:
        if "=" not in arg:
            raise ValueError(f"Invalid input format: '{arg}' (expected node_id=value)")
        key, raw_value = arg.split("=", 1)
        # Try to parse as number
        try:
            value: object = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value
        values[key.strip()] = value
    return values


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    format_kind = FormatKind.from_string(args.format)

    try:
        if args.command == "inspect":
            cmd = InspectCommand(format_kind=format_kind, verbose=args.verbose)
            return cmd.execute(
                target=args.target,
                depth=args.depth,
                show_values=args.show_values,
            )

        elif args.command == "validate":
            cmd = ValidateCommand(format_kind=format_kind, verbose=args.verbose)
            return cmd.execute(
                strict=args.strict,
                check_cycles=args.check_cycles,
                check_orphans=args.check_orphans,
            )

        elif args.command == "export":
            cmd = ExportCommand(format_kind=format_kind, verbose=args.verbose)
            return cmd.execute(
                output_path=args.output,
                export_format=args.export_format,
                include_values=args.include_values,
            )

        elif args.command == "run":
            input_values = parse_input_values(args.input)
            cmd = RunCommand(format_kind=format_kind, verbose=args.verbose)
            return cmd.execute(
                inputs=input_values,
                outputs=args.output,
                dry_run=args.dry_run,
            )

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
