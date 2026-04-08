def process_with_ptychodus(**data: str) -> None:
    from pathlib import Path
    from ptychodus.model import ModelCore

    action = data['action']
    input_directory = Path(data['input_directory'])
    output_directory = Path(data['output_directory'])

    with ModelCore() as model:
        model.batch_mode_execute(action, input_directory, output_directory)
