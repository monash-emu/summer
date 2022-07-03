from summer import CompartmentalModel
from summer.runner import VectorizedRunner


def get_runner(m: CompartmentalModel, parameters: dict):
    runner = VectorizedRunner(m)
    runner.prepare_to_run(parameters=parameters)
    m._backend = runner
    return runner
