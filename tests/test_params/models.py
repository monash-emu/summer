import numpy as np

from summer import CompartmentalModel, Stratification

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
    """
    Base parameterized model
    """
    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",param("contact_rate"),"S","I")
    m.add_transition_flow("recovery",param("recovery_rate"),"I","R")
    return m

def build_model_static(params):
    """
    Static model equivalent to base build_model_params
    """
    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",params['contact_rate'],"S","I")
    m.add_transition_flow("recovery",params['recovery_rate'],"I","R")
    return m

def build_model_params_func():
    """Model with parameterized Function
    """

    def custom_rate(time, scale):
        return (time * 0.1) * scale

    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",func(custom_rate, args=(Time, param("contact_scale"),)),"S","I")
    m.add_transition_flow("recovery",param("recovery_rate"),"I","R")
    return m

def build_model_static_func(params):
    """Static equivalent to build_model_params_func
    """

    def custom_rate(time, cv):
        return (time * 0.1) * params["contact_scale"]

    m = CompartmentalModel([0,100],["S","I","R"],["I"])
    m.set_initial_population(dict(S=90,I=10,R=0))
    m.add_infection_frequency_flow("infection",custom_rate,"S","I")
    m.add_transition_flow("recovery",params["recovery_rate"],"I","R")
    return m

def build_model_params_strat():
    """Stratified parameterized model
    """
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
    
def build_model_mixing_func():
    """Stratified model with a custom fucntion for mixing matrix
    """
    # Use our base parameterized model as above
    m = build_model_params_func()
    age_strat = Stratification(name="age", strata=['young','old'], compartments=["S", "I", "R"])
        
    # Infectiousness adjustment parameter
    age_strat.add_infectiousness_adjustments("I", {'old': None, 'young': param("young_infect_scale")})
    
    def scale_and_clamp(matrix, scale):
        mm = matrix * scale
        return np.clip(mm, 0.0, 1.0)
    
    # Include static mixing matrix as parameter
    age_strat.set_mixing_matrix(func(scale_and_clamp, 
                                      [param("mixing_matrix"), param("matrix_scale")]
                                     )
                               )

    # Flow adjustment ModelFunction
    age_strat.set_flow_adjustments('recovery', {'young': None, 
                                                'old': param("aged_recovery_scale")})
    
    m.stratify_with(age_strat)
    
    return m

def build_model_new_derived():
    """Model with new style derived outputs
    """
    m = build_model_mixing_func()
    m.request_output_for_compartments("total_population", ["S","I","R"])
    m.request_output_for_compartments("recovered", ["R"])
    
    def calc_prop(numerator, denominator):
            return numerator / denominator

    m.request_param_function_output("proportion_seropositive", 
                                    func(calc_prop, [DerivedOutput("recovered"),
                                           DerivedOutput("total_population")]
                                         )
                                    )
    def scale(value, scale):
        return value * scale
                                          
    m.request_param_function_output("prop_seropositive_surveyed", 
                                    func(scale, 
                                          kwargs={"value": DerivedOutput("proportion_seropositive"),
                                           "scale": param("serosurvey_scale")
                                          })
                                         )
    
    return m