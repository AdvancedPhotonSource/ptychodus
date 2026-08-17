from pathlib import Path
import argparse


class DirectoryType:
    def __init__(self, *, must_exist: bool) -> None:
        self._must_exist = must_exist

    def __call__(self, string: str) -> Path:
        path = Path(string)

        if self._must_exist and not path.is_dir():
            raise argparse.ArgumentTypeError(f'"{string}" is not a directory!')

        return path


def verify_all_arguments_parsed(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if argv:
        parser.error('unrecognized arguments: %s' % ' '.join(argv))
