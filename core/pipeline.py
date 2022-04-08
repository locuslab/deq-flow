

def get_model(args):
    if args.sliced_core:
        from sliced_core import DEQFlow
    else:
        from indexing_core import DEQFlow

    return DEQFlow
