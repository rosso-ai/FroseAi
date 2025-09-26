import os
import json
import csv
import torch
from torch import nn
from omegaconf import OmegaConf
from logging import getLogger
from datetime import datetime
from ..context import FroseArguments


class FedValidator:
    def __init__(self, conf: FroseArguments, test_data):
        self._conf = conf
        self._test_data = test_data
        self._metrics = {}

        dt_now = datetime.now()
        job_name = self._conf.repo_name + "_" + dt_now.strftime('%Y%m%d%H%M%S')
        log_output_path = os.path.join(self._conf.log_output_path, job_name)
        os.makedirs(log_output_path, exist_ok=True)
        OmegaConf.save(self._conf, os.path.join(str(log_output_path), "config.yml"))

        file_name = os.path.join(str(log_output_path), "metrics.csv")
        self._metrics_f = open(file_name, "w", encoding="utf-8")
        self._metrics_writer = csv.writer(self._metrics_f)
        self._log_no_header = True

        self._logger = getLogger("FroseAi-Vaidator")

    def __del__(self):
        self._metrics_f.close()

    @property
    def test_data(self):
        return self._test_data

    @property
    def metrics(self):
        return json.dumps(self._metrics)

    def test(self, model: nn.Module, round_num: int, device="cpu"):
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))
        criterion = nn.CrossEntropyLoss()

        metrics = {"accuracy": 0., "loss": 0.}

        loss_ary = []
        with torch.no_grad():
            total = 0
            correct = 0
            batch_loss = []
            model.to(device)

            for _, (x, target) in enumerate(self.test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, 1)
                c = (predicted == target).squeeze()
                for i in range(4):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                total += target.size(0)
                correct += (predicted == target).sum().item()

                batch_loss.append(loss.item())
                loss_ary.append(sum(batch_loss) / len(batch_loss))

            metrics["accuracy"] = correct / total
            metrics["loss"] = sum(loss_ary) / len(loss_ary)

            self._logger.info(" *** ROUND %d  AGGREGATE DONE  : %s"  % (round_num, str(metrics)))

        return metrics

    def write_log(self, round_num: int):
        metrics_key = ["round"]
        metrics_val = [round_num]
        for k, v in self._metrics.items():
            metrics_key.append(k)
            metrics_val.append(v)

        if self._log_no_header:
            self._metrics_writer.writerow(metrics_key)
            self._log_no_header = False

        self._metrics_writer.writerow(metrics_val)
        self._metrics_f.flush()

