from tests.test_agent.example_model import build_example_model

NUM_PEOPLE = 100
INTIIAL_INFECTED = 10


def test_example_model__smoke_test_run():
    model = build_example_model(NUM_PEOPLE, INTIIAL_INFECTED)
    model.run()
