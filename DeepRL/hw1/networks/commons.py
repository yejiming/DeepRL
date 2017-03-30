import tensorflow as tf
slim = tf.contrib.slim


def get_model_variables(prefixes, mode="include"):
    variables = []
    flag = (mode == "include")
    for var in slim.get_model_variables():
        excluded = False
        for prefix in prefixes:
            if var.op.name.startswith(prefix):
                excluded = True
                break
        if excluded == flag:
            variables.append(var)
    return variables


def get_variable(name):
    for var in slim.get_variables():
        if var.op.name == name:
            return var
    return None


def get_variable_array(name, session):
    var = get_variable(name)
    return var.eval(session=session)