"""
Jupyter notebook config
https://jupyter-notebook.readthedocs.io/en/stable/config.html
"""


def scrub_output_pre_save(model, **kwargs):
    """
    Scrub notebook output before saving
    https://gist.github.com/binaryfunt/f31a7ecc8d698dd0edbec1489f2ab055
    """
    # Only run on notebooks
    if model["type"] != "notebook":
        return
    # Only run on nbformat v4
    if model["content"]["nbformat"] != 4:
        return

    for cell in model["content"]["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        if "collapsed" in cell["metadata"]:
            cell["metadata"].pop("collapsed", 0)


c.FileContentsManager.pre_save_hook = scrub_output_pre_save
