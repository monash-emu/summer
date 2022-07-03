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