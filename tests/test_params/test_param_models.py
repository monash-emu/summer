import numpy as np

from .models import *


def test_param_static_equal():
    params = {"contact_rate": 1.0, "recovery_rate": 0.01}
    
    m_param = build_model_params()
    m_param.run(parameters=params)

    m_static = build_model_static(params)

    m_static.run()

    assert( (m_param.outputs == m_static.outputs).all() )

def test_param_func_equal():
    params = {"recovery_rate": 0.02, "contact_scale": 1.0}

    m_param = build_model_params_func()
    m_param.run(parameters = params)

    m_static = build_model_static_func(params)
    m_static.run()

    assert( (m_param.outputs == m_static.outputs).all() )

def test_param_model_strat():
    m_param = build_model_params_strat()
    mm = np.array((0.1,0.2,0.1,0.5)).reshape(2,2)

    params_strat = {'recovery_rate': 0.02, 'contact_scale': 5.0, 'aged_recovery_scale': 0.5, 
                'mixing_matrix': mm, 'young_infect_scale': 5.0}

    m_param.run(parameters=params_strat)

def test_param_model_mixing_func():
    mm = np.array((0.1,0.2,0.1,0.5)).reshape(2,2)

    params_mm_func = {'recovery_rate': 0.02, 'contact_scale': 1.0, 'aged_recovery_scale': 0.5, 
                    'mixing_matrix': mm, 'young_infect_scale': 3.0, 'matrix_scale': 0.5}

    m_param = build_model_mixing_func()
    m_param.run(parameters=params_mm_func)

def test_param_model_derived_outputs():
    m_param = build_model_new_derived()

    mm = np.array((0.1,0.2,0.1,0.5)).reshape(2,2)
    params_new_do = {'recovery_rate': 0.02, 'contact_scale': 1.0, 'aged_recovery_scale': 0.5, 
                    'mixing_matrix': mm, 'young_infect_scale': 3.0, 'matrix_scale': 0.5, "serosurvey_scale": 0.5}

    m_param.run(parameters=params_new_do)
    do_df = m_param.get_derived_outputs_df()

    assert( set(do_df.columns) == set(["total_population","recovered","proportion_seropositive", "prop_seropositive_surveyed"]))