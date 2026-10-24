import argparse
import random
import sys

from . import dataset, evaluate
from .models import OllamaModel
from .task import Task


def cmd_get_data(args):
    n = dataset.export_cord(args.out, split=args.split, limit=args.limit)
    print(f"Wrote {n} examples to {args.out}")


def cmd_test(args):
    task = Task.load(args.task)
    examples = dataset.load_folder(args.data)
    if not examples:
        sys.exit(f"No labelled examples found in {args.data}")
    if args.n and args.n < len(examples):
        examples = random.Random(0).sample(examples, args.n)

    for name in args.model:
        model = OllamaModel(name, host=args.host)

        def progress(i, total, report):
            print(f"\r{name}: {i}/{total}  running accuracy {report.accuracy():.0%}", end="", flush=True)

        report = evaluate.run(model, task, examples, on_progress=progress)
        print("\r" + " " * 70 + "\r" + report.summary(task))
        print(f"  details: {report.save()}\n")


def main():
    p = argparse.ArgumentParser(prog="aiflow")
    sub = p.add_subparsers(required=True)

    g = sub.add_parser("get-data", help="download the public CORD receipt set into our folder format")
    g.add_argument("--out", default="data/cord-test")
    g.add_argument("--split", default="test", choices=["train", "validation", "test"])
    g.add_argument("--limit", type=int)
    g.set_defaults(func=cmd_get_data)

    t = sub.add_parser("test", help="the Play button: score one or more models on labelled examples")
    t.add_argument("--model", action="append", required=True, help="Ollama model name; repeat to compare")
    t.add_argument("--task", default="templates/receipt.json")
    t.add_argument("--data", default="data/cord-test")
    t.add_argument("--n", type=int, help="test on a random sample of this many examples")
    t.add_argument("--host", default="http://localhost:11434")
    t.set_defaults(func=cmd_test)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
