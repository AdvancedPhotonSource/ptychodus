def process_with_ptychodus(**data: str) -> None:
    """Run a staged reconstruction or training on the compute endpoint.

    Registered with Globus Compute by its source, so it imports everything it uses. The
    command's standard output and error go to ``stdout.log`` and ``stderr.log`` in the
    output directory, which the flow transfers back with the results.
    """
    import subprocess
    import sys
    from pathlib import Path

    action = data['action']
    input_directory = Path(data['input_directory'])
    output_directory = Path(data['output_directory'])

    if action == 'reconstruct':
        command = ['-m', 'ptychodus.cli.reconstruct_prepared']
    elif action == 'train':
        options_file = input_directory / 'train_options.yaml'
        command = ['-m', 'ptychodus.cli.train', '--options', str(options_file)]
    else:
        raise ValueError(f'Unknown remote action "{action}"!')

    output_directory.mkdir(parents=True, exist_ok=True)

    with (
        open(output_directory / 'stdout.log', 'a') as stdout,
        open(output_directory / 'stderr.log', 'a') as stderr,
    ):
        subprocess.run(
            [sys.executable, *command, '-i', str(input_directory), '-o', str(output_directory)],
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
