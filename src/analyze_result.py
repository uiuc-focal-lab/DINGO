import fire
import srsly

from result_utils import align_results


def main(first, second):
    first = srsly.read_jsonl(first)
    second = srsly.read_jsonl(second)

    first_only = []
    second_only = []
    both = []
    neither = []

    for one, two in align_results(first, second):
        i = one["idx"]
        if not one["correct"] and two["correct"]:
            second_only.append(i)

        elif one["correct"] and not two["correct"]:
            first_only.append(i)

        elif one["correct"] and two["correct"]:
            both.append(i)

        else:
            neither.append(i)

    print("Both (Total, ids) ", len(both), both)
    print()
    print("Neither (Total, ids) ", len(neither), neither)
    print()
    print("First Only (Total, ids) ", len(first_only), first_only)
    print()
    print("Second Only (Total, ids) ", len(second_only), second_only)


if __name__ == "__main__":
    fire.Fire(main)
