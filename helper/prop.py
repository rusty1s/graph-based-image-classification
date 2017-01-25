def prop(obj, name, default=None):
    if default is None:
        return obj[name]
    else:
        return obj[name] if name in obj else default
