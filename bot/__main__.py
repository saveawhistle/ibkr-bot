"""Entry point so ``python -m bot <command>`` dispatches to the Typer app."""

from bot.cli import app

if __name__ == "__main__":
    app()
