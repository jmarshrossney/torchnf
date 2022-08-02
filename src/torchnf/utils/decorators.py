from functools import wraps


def skip_if_logging_disabled(meth):
    @wraps(meth)
    def wrapper(model, *args, **kwargs):
        if getattr(model, "logger", None) is None:
            return
        return meth(model, *args, **kwargs)

    return wrapper
