from typing import List
from collections import namedtuple

from cornac.utils import validate_format
from cornac.utils import cache
from cornac.data import Reader
from cornac.data.reader import read_text


Yahoo = namedtuple("Yahoo", ["url", "unzip", "path", "sep", "skip"])
YAHOO_DATASETS = {
    "CLOSED_LOOP": Yahoo(
        "file:///local/terrier/Collections/Recommendations/yahoo_ymusic_v1/ydata-ymusic-rating-study-v1_0-train.txt",
        False,
        "yahoo_ymusic_v1/ydata-ymusic-rating-study-v1_0-train.txt",
        "\t",
        0,
    ),
    "OPEN_LOOP": Yahoo(
        "file:///local/terrier/Collections/Recommendations/yahoo_ymusic_v1/ydata-ymusic-rating-study-v1_0-test.txt",
        False,
        "yahoo_ymusic_v1/ydata-ymusic-rating-study-v1_0-test.txt",
        "\t",
        0,
    ),
}


def load_feedback(variant="closed_loop", reader=None):
    """Load the user-item ratings of one of the Yahoo Music datasets

    Parameters
    ----------
    variant: str, optional, default: 'closed_loop'
        Specifies which Yahoo Music dataset to load, one of ['closed_loop', 'open_loop'].

    reader: `obj:cornac.data.Reader`, optional, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.
    """

    yah = YAHOO_DATASETS.get(variant.upper(), None)
    if yah is None:
        raise ValueError(
            "variant must be one of {}.".format(YAHOO_DATASETS.keys()))

    fpath = cache(url=yah.url, unzip=yah.unzip, relative_path=yah.path)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, 'UIR', sep=yah.sep, skip_lines=yah.skip)
