import torch
import copy
from typing import Dict
from torch import nn
from ..agg_frame import FroseAiAggFrame


class FedAvgAggregator(FroseAiAggFrame):
    def aggregate(self):
        sample_num = 0
        for i in range(self.client_num):
            sample_num += self._received[i]["sample_num"]

        with torch.no_grad():
            average_params = self.model.cpu().state_dict()
            for i in range(self.client_num):

                sample_rate = 1
                if sample_num != 0:
                    sample_rate = self._received[i]["sample_num"] / sample_num

                for k in average_params.keys():
                    if i == 0:
                        average_params[k] = self._received[i]["model"][k] * sample_rate
                    else:
                        average_params[k] += self._received[i]["model"][k] * sample_rate

            self.model.load_state_dict(average_params)
            self.messages["model"] = copy.deepcopy(self.model).cpu().state_dict()

    def test(self):
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))
        criterion = nn.CrossEntropyLoss().to(self.device)

        metrics = {"accuracy": 0., "loss": 0.}

        loss_ary = []
        with torch.no_grad():
            total = 0
            correct = 0
            batch_loss = []
            self.model.to(self.device)
            for _, (x, target) in enumerate(self.test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
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

            self._logger.info(" *** ROUND %d  AGGREGATE DONE  : %s"  % (self.round, str(metrics)))

        return metrics

