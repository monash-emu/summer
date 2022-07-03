from summer import CompartmentalModel

from summer.parameters.params import (
    Parameter,
    Time,
    Function,
    DerivedOutput,
    ComputedValue,
    find_all_parameters,
    Variable,
    model_var
)

param = Parameter
func = Function

def build_model_params() -> CompartmentalModel:
    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",param("contact_rate"),"S","I")
    m.add_transition_flow("recovery",param("recovery_rate"),"I","R")
    return m

def build_model_static(params):
    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",params['contact_rate'],"S","I")
    m.add_transition_flow("recovery",params['recovery_rate'],"I","R")
    return m

def build_model_params_func():

    def custom_rate(time, scale):
        return (time * 0.1) * scale

    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",func(custom_rate, args=(Time, param("contact_scale"),)),"S","I")
    m.add_transition_flow("recovery",param("recovery_rate"),"I","R")
    return m

def build_model_static_func(params):

    def custom_rate(time, cv):
        return (time * 0.1) * params["contact_scale"]

    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",custom_rate,"S","I")
    m.add_transition_flow("recovery",params["recovery_rate"],"I","R")
    return m

def build_model_params_strat():
    # Use our base parameterized model as above
    m = build_model_params_func()
    age_strat = Stratification(name="age", strata=['young','old'], compartments=["S", "I", "R"])

    def constant(value):
        return value
        
    # Infectiousness adjustment parameter
    age_strat.add_infectiousness_adjustments("I", {'old': None, 'young': param("young_infect_scale")})

    # Include static mixing matrix as parameter
    age_strat.set_mixing_matrix(param("mixing_matrix"))

    # Flow adjustment ModelFunction
    age_strat.set_flow_adjustments('recovery', {'young': None, 
                                                'old': func(constant, 
                                                             kwargs={"value": param("aged_recovery_scale")})
                                               })
    
    m.stratify_with(age_strat)
    
    return m
    
