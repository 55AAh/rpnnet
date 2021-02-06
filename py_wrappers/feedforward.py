import time

from target.release.rpnnet import feedforward


class Net:
    def __init__(self, geometry, coeffs=None,
                 total_samples=0, total_time=0):
        self._net = feedforward.Net(geometry, coeffs)
        self._geometry = geometry
        self._total_samples = total_samples
        self._total_time = total_time

    @classmethod
    def _from_consumed(cls, freed_net, consumed_net):
        obj = cls.__new__(cls)
        super(Net, obj).__init__()
        obj._net = freed_net
        obj._geometry = consumed_net._geometry
        obj._total_samples = consumed_net._total_samples
        obj._total_time = consumed_net._total_time
        return obj

    @property
    def geometry(self): return self._geometry

    @property
    def total_samples(self): return self._total_samples

    @total_samples.setter
    def total_samples(self, val): self._total_samples = val

    @property
    def total_time(self): return self._total_time

    @total_time.setter
    def total_time(self, val): self._total_time = val

    @staticmethod
    def from_file(file):
        with file:
            file.seek(0)
            lines = file.readlines()

        if len(lines) == 0:
            return None

        geometry = [int(number) for number in lines[0].rstrip(',').split(',')]
        coeffs = [float(number) for number in lines[1].rstrip(',').split(',')]
        stats = lines[2].split(',')
        total_samples, total_time = int(stats[0]), float(stats[1])
        return Net(geometry, coeffs, total_samples, total_time)

    @staticmethod
    def calc_cost(outputs, desired_outputs):
        return feedforward.Net.calc_cost(outputs, desired_outputs)

    def process(self, inputs): return self._net.process(inputs)

    def build_trainer(self): return Trainer(self)

    def export(self): return (self._net.export(),
                              (self.total_samples, self.total_time))

    @staticmethod
    def _save_exported_to_file(exported, file):
        ((geometry, coeffs), (total_samples, total_time)) = exported

        data = \
            ','.join([str(g) for g in geometry]) + '\n' + \
            ','.join([str(c) for c in coeffs]) + '\n' + \
            f"{total_samples},{total_time}"

        with file:
            file.seek(0)
            file.write(data)
            file.truncate()

    def save_to_file(self, file):
        self._save_exported_to_file(self.export(), file)


class TrainSample:
    def __init__(self, trainer, samples_count):
        self.trainer = trainer
        self.samples_count = samples_count

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, _exc_type, exp_val, _exc_traceback):
        if exp_val is None:
            delta_time = time.time() - self.start_time
            self.trainer.total_samples += self.samples_count
            self.trainer.total_time += delta_time


class Trainer:
    def __init__(self, net):
        self._trainer = net._net.build_trainer()
        self._net = net

    @classmethod
    def _from_consumed(cls, freed_trainer, consumed_net):
        obj = cls.__new__(cls)
        super(Trainer, obj).__init__()
        obj._trainer = freed_trainer
        obj._net = consumed_net
        return obj

    @property
    def geometry(self): return self._net.geometry

    @property
    def total_samples(self): return self._net._total_samples

    @total_samples.setter
    def total_samples(self, val): self._net._total_samples = val

    @property
    def total_time(self): return self._net._total_time

    @total_time.setter
    def total_time(self, val): self._net._total_time = val

    def export_net(self): return (self._trainer.export_net(),
                                  (self.total_samples, self.total_time))

    def save_to_file(self, file):
        Net._save_exported_to_file(self.export_net(), file)

    def process(self, inputs): return self._trainer.process(inputs)

    def train_sample(self, inputs, desired_outputs, grad_mult_coeff):
        with TrainSample(self, 1):
            result = self._trainer.train(
                inputs, desired_outputs, grad_mult_coeff)
        return result

    def apply_training(self):
        self._trainer.apply_training()

    def train_batch(self, samples, grad_mult_coeff, get_outputs=False):
        with TrainSample(self, len(samples)):
            cost, outputs = self._trainer.train_batch(
                samples, grad_mult_coeff, get_outputs)
        if get_outputs:
            return cost, outputs
        return cost

    def train_random(self, passes_count, batch_size, samples,
                     grad_mult_coeff, get_outputs=False):
        with TrainSample(self, passes_count):
            cost, res = \
                self._trainer.train_random(passes_count,
                                           batch_size, samples,
                                           grad_mult_coeff, get_outputs)
        if get_outputs:
            input_indices, outputs = res
            return cost, input_indices, outputs
        return cost

    def build_sampled(self, samples):
        return SampledTrainer(self, samples)

    def teardown(self):
        return Net._from_consumed(self._trainer.teardown(), self._net)


class SampledTrainer:
    def __init__(self, trainer, samples):
        self._sampled = trainer._trainer.build_sampled(samples)
        self._net = trainer._net

    @property
    def geometry(self): return self._net.geometry

    @property
    def total_samples(self): return self._net._total_samples

    @total_samples.setter
    def total_samples(self, val): self._net._total_samples = val

    @property
    def total_time(self): return self._net._total_time

    @total_time.setter
    def total_time(self, val): self._net._total_time = val

    def export_net(self): return (self._sampled.export_net(),
                                  (self.total_samples, self.total_time))

    def save_to_file(self, file):
        Net._save_exported_to_file(self.export_net(), file)

    def process(self, inputs): return self._sampled.process(inputs)

    def train_random(self, passes_count, batch_size,
                     grad_mult_coeff, get_outputs=False):
        with TrainSample(self, passes_count):
            cost, res = self._sampled.train_random(
                passes_count, batch_size, grad_mult_coeff, get_outputs)
        if get_outputs:
            input_indices, outputs = res
            return cost, input_indices, outputs
        return cost

    def teardown(self):
        return Trainer._from_consumed(self._sampled.teardown(), self._net)
