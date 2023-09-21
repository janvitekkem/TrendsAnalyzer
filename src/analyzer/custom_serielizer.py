from ast import literal_eval

def JSdecoded(item:dict, dict_key=False):
    if isinstance(item, list):
        return [ JSdecoded(e) for e in item ]
    elif isinstance(item, dict):
        return { literal_eval(key) : value for key, value in item.items() }
    return item

def JSencoded(item, dict_key=False):
    if isinstance(item, tuple):
        if dict_key:
            return str(item)
        else:
            return list(item)
    elif isinstance(item, list):
        return [JSencoded(e) for e in item]
    elif isinstance(item, dict):
        return { JSencoded(key, True) : JSencoded(value) for key, value in item.items() }
    elif isinstance(item, set):
        return list(item)
    return item