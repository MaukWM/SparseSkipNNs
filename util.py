

# List of Dicts to Dicts of Lists from: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
def ld_to_dl(ld):
    """
    Convert a list of dicts to a dict of lists
    """
    return {k: [dic[k] for dic in ld] for k in ld[0]}