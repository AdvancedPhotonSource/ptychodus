#!/usr/bin/env python

from pathlib import Path
import argparse
import sys

from ptychodus.model import ModelArgs, ModelCore
import ptychodus


def versionString() -> str:
    return f'{ptychodus.__name__.title()} ({ptychodus.__version__})'


def verifyAllArgumentsParsed(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if argv:
        parser.error('unrecognized arguments: %s' % ' '.join(argv))


class DirectoryType:

    def __call__(self, string: str) -> Path:
        path = Path(string)

        if not path.is_dir():
            raise argparse.ArgumentTypeError(f'\"{string}\" is not a directory!')

        return path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog=ptychodus.__name__.lower(),
        description=f'{ptychodus.__name__} is a ptychography analysis application',
    )
    parser.add_argument(
        '-b',
        '--batch',
        choices=('reconstruct', 'train'),
        help='run action non-interactively',
    )
    parser.add_argument(
        '-d',
        '--dev',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-f',
        '--file-prefix',
        help='replace file path prefix in settings',
    )
    parser.add_argument(
        # input data product file (batch mode)
        '-i',
        '--input',
        metavar='INPUT_FILE',
        type=argparse.FileType('r'),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        # output data product file (batch mode)
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        type=argparse.FileType('w'),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        # preprocessed diffraction patterns file (batch mode)
        '-p',
        '--patterns',
        metavar='PATTERNS_FILE',
        type=argparse.FileType('r'),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-s',
        '--settings',
        metavar='SETTINGS_FILE',
        help='use settings from file',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=versionString(),
    )
    parser.add_argument(
        '-w',
        '--write',
        metavar='OUTPUT_DIR',
        type=DirectoryType(),
        help='stage reconstruction inputs to directory',
    )

    parsedArgs, unparsedArgs = parser.parse_known_args()

    modelArgs = ModelArgs(
        settingsFile=Path(parsedArgs.settings.name) if parsedArgs.settings else None,
        patternsFile=Path(parsedArgs.patterns.name) if parsedArgs.patterns else None,
        replacementPathPrefix=parsedArgs.file_prefix,
    )

    with ModelCore(modelArgs, isDeveloperModeEnabled=parsedArgs.dev) as model:
        if parsedArgs.write is not None:
            return model.stageReconstructionInputs(parsedArgs.write)

        if parsedArgs.batch is not None:
            verifyAllArgumentsParsed(parser, unparsedArgs)

            if parsedArgs.input is None or parsedArgs.output is None:
                parser.error('Batch mode requires input and output arguments!')
                return -1

            inputPath = Path(parsedArgs.input.name)
            outputPath = Path(parsedArgs.output.name)

            if parsedArgs.batch == 'reconstruct':
                return model.batchModeReconstruct(inputPath, outputPath)
            elif parsedArgs.batch == 'train':
                return model.batchModeTrain(inputPath, outputPath)
            else:
                parser.error(f'Unknown batch mode action \"{parsedArgs.batch}\"!')
                return -1

        try:
            from PyQt5.QtWidgets import QApplication
        except ModuleNotFoundError:
            return 0

        # QApplication expects the first argument to be the program name
        app = QApplication(sys.argv[:1] + unparsedArgs)
        verifyAllArgumentsParsed(parser, app.arguments()[1:])

        from ptychodus.view import ViewCore
        view = ViewCore.createInstance(parsedArgs.dev)

        from ptychodus.controller import ControllerCore
        controller = ControllerCore.createInstance(model, view)
        controller.showMainWindow(versionString())

        return app.exec()


sys.exit(main())
