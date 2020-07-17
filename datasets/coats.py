from typing import List
from collections import namedtuple

from cornac.utils import validate_format
from cornac.utils import cache
from cornac.data import Reader
from cornac.data.reader import read_text


Coats = namedtuple("Coats", ["url", "unzip", "path", "sep", "skip"])
COATS_DATASETS = {
    "CLOSED_LOOP": Coats(
        "file:///local/terrier/Collections/Recommendations/coats/train.csv",
        False,
        "train.csv",
        "\t",
        0,
    ),
    "OPEN_LOOP": Coats(
        "file:///local/terrier/Collections/Recommendations/coats/test.csv",
        False,
        "test.csv",
        "\t",
        0,
    ),
}


def load_feedback(variant="closed_loop", reader=None):
    """Load the user-item ratings of one of the Coats datasets

    Parameters
    ----------
    variant: str, optional, default: 'closed_loop'
        Specifies which Coats dataset to load, one of ['closed_loop', 'open_loop'].

    reader: `obj:cornac.data.Reader`, optional, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.
    """

    coat = COATS_DATASETS.get(variant.upper(), None)
    if coat is None:
        raise ValueError(
            "variant must be one of {}.".format(YAHOO_DATASETS.keys()))

    fpath = cache(url=coat.url, unzip=coat.unzip, relative_path=coat.path)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, 'UIR', sep=coat.sep, skip_lines=coat.skip)
