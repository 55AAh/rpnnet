from signal import signal, SIGINT
import argparse
import time
import importlib

from rpnnet import feedforward

terminate = [False]


def load_protocol(rel_path):
    if rel_path[-3:] == ".py":
        rel_path = rel_path[:-3]
    rel_path = rel_path.replace('/', '.'). \
        replace('\\', '.')
    try:
        mod = importlib.import_module(rel_path)
        dp = mod.DataProvider()
        for f in [
            "get_samples",
                "is_correct"]:
            if f not in dir(dp):
                raise Exception(
                    f"No '{f}' function was found in DataProvider class!")
        return dp
    except Exception as e:
        raise Exception(f"Cannot load protocol '{rel_path}': {e}")


def format_time(t, ls=False, pm=False):
    if t >= 60:
        return f"{int(t/60)}m {int(t%60)}s"
    else:
        s = f"{int((t * 1000) % 1000)}"
        if pm:
            s = " "*(3-len(s)) + s
        s += "ms"
        if t >= 1 or ls:
            s = f"{int(t)}s " + s
        return s

    s = str(int(t % 60)) + "s"
    if t >= 60:
        s = str(int(t / 60)) + "m " + s
    else:
        s += " " + str(int((t * 1000) % 1000)) + "ms"
    return s


class TaskPrinter():
    def __init__(self, text):
        self.text = text
        self.start_time = time.time()

    def __enter__(self):
        print(self.text + "...", end='', flush=True)

    def __exit__(self, _exc_type, exp_val, _exc_traceback):
        print(f" ok, {format_time(time.time()-self.start_time)}")


class AvgWindowCounter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []
        self.window_sum = 0

    def add(self, value):
        self.window_sum += value
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window_sum -= self.window[0]
            self.window = self.window[1:]
        return self.cur()

    def cur(self):
        return self.window_sum / len(self.window)

    def cnt(self):
        return len(self.window)

    def per(self):
        return self.cnt() / self.window_size


class SparsePrinter:
    def __init__(self, pps):
        self.pps = pps
        self.prev_time = time.time()

    def p(self, *args, **kwargs):
        if time.time() - self.prev_time > 1 / self.pps:
            print(*args, **kwargs)
            self.prev_time = time.time()


def prepare_net(net_file, data_file, protocol):
    with TaskPrinter(f"Reading net from '{net_file.name}'"):
        net = feedforward.Net.from_file(net_file)
    with TaskPrinter(f"Getting testing samples from '{data_file.name}' " +
                     "using protocol"):
        samples = protocol.get_samples(data_file.name)
    return net, samples


def prepare_trainer(net_file, data_file, protocol):
    with TaskPrinter(f"Reading net from '{net_file.name}'"):
        net = feedforward.Net.from_file(net_file)
    with TaskPrinter(f"Getting training samples from '{data_file.name}' " +
                     "using protocol"):
        samples = protocol.get_samples(data_file.name)
    with TaskPrinter("Creating trainer"):
        trainer = net.build_trainer()

    return trainer, samples


def print_stats(nt):
    print((f" Geometry:             [{nt.geometry[0]} |" +
           " ".join([str(g) for g in nt.geometry[1:-1]]) +
           f"| {nt.geometry[-1]}]").replace("||", "|"))
    print(f" Total samples:        {nt.total_samples}")
    print(f" Total training time:  {format_time(nt.total_time)}")


def ask(q):
    a = " "
    while len(a) == 0 or a not in "YN":
        a = input(q + " (y/n): ").upper()
    return a == "Y"


def train_ordered(net_path, trainer, samples, batchsize, gradcoeff,
                  samplescount, maxtime, protocol):
    if samplescount is not None \
            and samplescount < batchsize:
        raise Exception("Samples count must be >= that batchsize!")
    print("\n\tORDERED TRAINING\n")
    print_stats(trainer)
    print()

    i, c = 0, 0
    start_time = time.time()
    avg_cost_counter = AvgWindowCounter(
        min(max(100000 / batchsize, 10), 10000))
    p = SparsePrinter(20)
    while not terminate[0]:
        time_elapsed = time.time() - start_time
        c += 1
        if c % batchsize == 0:
            trainer.apply_training()

        cost = trainer.train_sample(samples[i][0], samples[i][1], gradcoeff)
        avg_cost = avg_cost_counter.add(cost)

        m = f" Sample {c:>7} "
        if samplescount is not None:
            m += f"({c / samplescount:.2%})"
            if c > samplescount:
                break
        m += f"     {format_time(time_elapsed, True, True):>15} "
        if maxtime is not None:
            m += f"({time_elapsed / maxtime:.2%})"
            if time_elapsed > maxtime:
                break
        m += f"   {cost:>10.5f} | {avg_cost:<10.5f}"
        p.p(m)

        i += 1
        if i >= len(samples):
            i = 0

    print()
    save_net(trainer, net_path)
    print()


def train_random(net_path, trainer, samples, batchsize, gradcoeff,
                 samplescount, passescount, maxtime,
                 analytics, cached, protocol):
    print(f"\n\tRANDOM SUBSAMPLE TRAINING{' (CACHED)' if cached else ''}\n")
    print_stats(trainer)
    print()

    if cached:
        with TaskPrinter("Building sampled trainer"):
            sampled_trainer = trainer.build_sampled(samples)

    c = 0
    start_time = time.time()
    avg_cost_counter = AvgWindowCounter(10)
    avg_corr_counter = AvgWindowCounter(10)
    p = SparsePrinter(20)
    time_elapsed = time.time() - start_time - 0.001
    while not terminate[0]:
        old_time_elapsed = time_elapsed
        time_elapsed = time.time() - start_time
        c += passescount

        if cached:
            ret = sampled_trainer.train_random(
                passescount, batchsize, gradcoeff, analytics)
        else:
            ret = trainer.train_random(
                passescount, batchsize, samples, gradcoeff, analytics)

        if analytics:
            analytics_start_time = time.time()
            cost, input_indices, outputs = ret
            correct_count = 0
            for i, o in zip(input_indices, outputs):
                if protocol.is_correct(i, o):
                    correct_count += 1
            corr = correct_count / len(input_indices)
            avg_corr = avg_corr_counter.add(corr)
            analytics_time = time.time() - analytics_start_time
            analytics_per = min(analytics_time /
                                (time_elapsed-old_time_elapsed), 0.99)
        else:
            cost = ret
        avg_cost = avg_cost_counter.add(cost)

        m = f" Sample {c:>7} "
        if samplescount is not None:
            m += f"({c / samplescount:.2%})"
            if c > samplescount:
                break
        m += f"\t{format_time(time_elapsed, True, True):>15} "
        if maxtime is not None:
            m += f"({time_elapsed / maxtime:.2%})"
            if time_elapsed > maxtime:
                break
        m += f"   {cost:>10.5f} | {avg_cost:<10.5f}"
        m += f" {format_time(time_elapsed-old_time_elapsed):>9} "
        if analytics:

            m += f"\t{format_time(analytics_time, False, True):>9}"
            m += f" ({analytics_per:.2%})"
            m += f"   {corr:>.5%} | {avg_corr:<.5%}"
        p.p(m)

    print()
    if cached:
        save_net(sampled_trainer, net_path)
    else:
        save_net(trainer, net_path)
    print()


def test(net, samples, protocol):
    print("\n\tTESTING")
    print_stats(net)
    print()
    p = SparsePrinter(1000)
    cost_sum = 0
    corr_count = 0
    for i, (inputs, desired_outputs) in enumerate(samples):
        p.p(f"\r{i+1}", end='')
        outputs = net.process(inputs)
        cost_sum += net.calc_cost(outputs, desired_outputs)
        if protocol.is_correct(i, outputs):
            corr_count += 1
    print(f"\r{len(samples)}")
    print()
    print(f" AVG COST: {cost_sum/len(samples):.5f}")
    print(f"  CORRECT: {corr_count/len(samples):.5%}")


def save_net(nts, path):
    if ask("Save trained net?"):
        while True:
            p = input(
                f"Enter filename or press Enter to overwrite '{path}':\n") \
                .strip()
            path = path if p == "" else p
            try:
                nts.save_to_file(open(path, "w"))
                print(f"Saved to '{path}'!")
                break
            except Exception as e:
                print(e)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line Python client for rpnnet.")

    parser.add_argument("net", type=argparse.FileType('r'))
    parser.add_argument("protocol", type=argparse.FileType('r'))
    command_parsers = parser.add_subparsers(dest="command", required=True)

    ordered_parser = command_parsers.add_parser("train-ordered")
    ordered_parser.add_argument("data", type=argparse.FileType('r'))
    ordered_parser.add_argument("batchsize", type=int)
    ordered_parser.add_argument("gradcoeff", type=float)
    ordered_parser.add_argument("-s", "--samplescount", type=int)
    ordered_parser.add_argument("-t", "--time", type=float)

    random_parser = command_parsers.add_parser("train-random")
    random_parser.add_argument("data", type=argparse.FileType('r'))
    random_parser.add_argument("passescount", type=int)
    random_parser.add_argument("batchsize", type=int)
    random_parser.add_argument("gradcoeff", type=float)
    random_parser.add_argument("-s", "--samplescount", type=int)
    random_parser.add_argument("-t", "--time", type=float)
    random_parser.add_argument("-a", "--analytics", action="store_true")

    random_parser = command_parsers.add_parser("train-randomcached")
    random_parser.add_argument("data", type=argparse.FileType('r'))
    random_parser.add_argument("passescount", type=int)
    random_parser.add_argument("batchsize", type=int)
    random_parser.add_argument("gradcoeff", type=float)
    random_parser.add_argument("-s", "--samplescount", type=int)
    random_parser.add_argument("-t", "--time", type=float)
    random_parser.add_argument("-a", "--analytics", action="store_true")

    test_parser = command_parsers.add_parser("test")
    test_parser.add_argument("data", type=argparse.FileType('r'))

    return parser


def main():
    args = create_parser().parse_args()

    def sigint_handler(sig, f):
        print("\n TERMINATE")
        terminate[0] = True
    signal(SIGINT, sigint_handler)

    dp = load_protocol(args.protocol.name)

    net_path = args.net.name
    if args.command == "test":
        net, samples = prepare_net(args.net, args.data, dp)
        test(net, samples, dp)
    elif args.command[:5] == "train":
        trainer, samples = prepare_trainer(args.net, args.data, dp)
        if args.command == "train-ordered":
            train_ordered(net_path, trainer, samples,
                          args.batchsize, args.gradcoeff,
                          args.samplescount, args.time, dp)
        elif args.command == "train-random":
            train_random(net_path, trainer, samples,
                         args.batchsize, args.gradcoeff,
                         args.samplescount, args.passescount,
                         args.time, args.analytics,
                         False, dp)
        elif args.command == "train-randomcached":
            train_random(net_path, trainer, samples,
                         args.batchsize, args.gradcoeff,
                         args.samplescount, args.passescount,
                         args.time, args.analytics,
                         True, dp)


if __name__ == "__main__":
    main()
