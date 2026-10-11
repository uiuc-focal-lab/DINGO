"""Join experiment records by dataset identity, independent of worker order."""


def align_results(first, second):
    def indexed(rows):
        result = {}
        for row in rows:
            if "idx" not in row:
                raise ValueError("Result comparison requires idx; regenerate legacy records")
            if row["idx"] in result:
                raise ValueError("Result comparison contains duplicate idx")
            result[row["idx"]] = row
        return result

    a, b = indexed(first), indexed(second)
    if a.keys() != b.keys():
        raise ValueError("Compared runs must contain the same dataset IDs")
    return [(a[idx], b[idx]) for idx in sorted(a)]


def better_result(first, second):
    """Offline oracle: select one real result, including its actual latency."""
    return max((first, second), key=lambda r: (r["correct"], r["parses"], -r["time"]))
