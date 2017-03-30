from tensorflow.core.framework import summary_pb2


def get_value_from_summary_string(tag, summary_str):
    """ get_value_from_summary_string.

    Retrieve a summary value from a summary string.

    Arguments:
        tag: `str`. The summary tag (name).
        summary_str: `str`. The summary string to look in.

    Returns:
        A `float`. The retrieved value.

    Raises:
        `Exception` if tag not found.

    """
    summ = summary_pb2.Summary()
    summ.ParseFromString(summary_str)

    for row in summ.value:
        if row.tag == tag:
            return float(row.simple_value)

    raise ValueError("Tag: " + tag + " cannot be found in summaries list.")