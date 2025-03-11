_reg_opt_modules_ = {}
_reg_opt_workflow = None
_reg_opt_score_fn_ = None
_reg_opt_data_loader_ = None


def clear_registry():
    global _reg_opt_workflow
    global _reg_opt_score_fn_
    _reg_opt_modules_.clear()
    _reg_opt_workflow = None
    _reg_opt_score_fn_ = None
    _reg_opt_data_loader_ = None


def register_opt_module(module):
    _reg_opt_modules_[module.name] = module

def register_workflow(program):
    global _reg_opt_workflow
    _reg_opt_workflow = program
    return program


def register_evaluator(score_fn):
    global _reg_opt_score_fn_
    _reg_opt_score_fn_ = score_fn
    return score_fn


def register_data_loader(data_loader_fn):
    global _reg_opt_data_loader_
    _reg_opt_data_loader_ = data_loader_fn
    return data_loader_fn


def get_registered_opt_modules():
    return list(_reg_opt_modules_.values())


def get_registered_opt_program_entry():
    return _reg_opt_workflow


def get_registered_opt_score_fn():
    return _reg_opt_score_fn_


def get_registered_data_loader():
    return _reg_opt_data_loader_
