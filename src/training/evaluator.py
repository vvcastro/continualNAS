from copy import deepcopy
import numpy as np
import json
import os

from torch import nn, optim
import torch

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_classification.run_manager import RunManager
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import (
    DynamicSeparableConv2d,
)

from ofa.utils.layers import ResidualBlock, MBConvLayer, set_layer_from_config

import warnings

warnings.simplefilter("ignore")

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1


class OFAEvaluator:
    def __init__(
        self,
        family="mobilenetv3",
        model_path="./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0",
        pretrained=False,
        data_classes=1000,
    ):
        match family:
            case "mobilenetv3":
                if "w1.0" in model_path or "resnet50" in model_path:
                    width_mult = 1.0
                elif "w1.2" in model_path:
                    width_mult = 1.2
                else:
                    raise ValueError

                self.engine = OFAMobileNetV3(
                    n_classes=data_classes,
                    dropout_rate=0,
                    width_mult=width_mult,
                    ks_list=[3, 5, 7],
                    expand_ratio_list=[3, 4, 6],
                    depth_list=[2, 3, 4],
                )

            case _:
                raise KeyError(f"OFA family type: '{family}' not implemented!")

        if pretrained:
            init = torch.load(model_path, map_location="cpu")["state_dict"]

            ## FIX size mismatch error #####
            init["classifier.linear.weight"] = init["classifier.linear.weight"][
                :data_classes
            ]
            init["classifier.linear.bias"] = init["classifier.linear.bias"][
                :data_classes
            ]
            ##############################

            self.engine.load_state_dict(init)

    def get_architecture_model(self, architecture=None):
        """randomly sample a sub-network"""
        if architecture is not None:
            self.engine.set_active_subnet(
                d=architecture["depths"],
                e=architecture["widths"],
                ks=architecture["ksizes"],
            )
        else:
            architecture = self.engine.sample_active_subnet()
        subnet = self.engine.get_active_subnet(preserve_weight=True)
        return subnet, architecture

    def _train_model(self, model, train_loader, epochs, optimiser: str = "adam"):
        """
        Trains the specified model using the `data_loader` object.

        ## Params
        - `train_loader`: The data loader object for the trainin data.
        - `optimiser`: used to set a different optimiser, i.e: SAM.
        - `epochs`: number of epochs to use in the training set.
        """
        criterion = nn.CrossEntropyLoss()
        optimiser = self._get_optimizer(model, optimiser, learning_rate=1e-3)

        model.train()
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                optimiser.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimiser.step()

    def _evaluate_model(self, model, val_loader):
        """
        Evaluates the model in the specific validation split dataset.

        ## Params:
        - `model`: The potentially trained model to test.
        - `val_loader`: The loader object holding the validation data split.
        """
        criterion = nn.CrossEntropyLoss()
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        evaluation_metrics = {
            "val_loss": val_loss / len(val_loader),
            "accuracy": accuracy,
        }
        return evaluation_metrics

    def _get_optimiser(self, model, name, learning_rate=1e-3):
        """
        Optimiser initialisation for the model weights.
        - The idea is to include several class, specially interest is SAM.
        """
        match name:
            case "sgd":
                return optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=1e-4,
                )

            case "adam":
                return optim.Adam(model.parameters(), lr=learning_rate)

            case "sam":
                return optim.Adam(model.parameters(), lr=learning_rate)

            case _:
                raise ValueError(f"Optimiser '{name}' is not supported")

    @staticmethod
    def save_net_config(path, net, config_name="net.config"):
        """dump run_config and net_config to the model_folder"""
        net_save_path = os.path.join(path, config_name)
        json.dump(net.config, open(net_save_path, "w"), indent=4)
        print("Network configs dump to %s" % net_save_path)

    @staticmethod
    def save_net(path, net, model_name):
        """dump net weight as checkpoint"""
        if isinstance(net, torch.nn.DataParallel):
            checkpoint = {"state_dict": net.module.state_dict()}
        else:
            checkpoint = {"state_dict": net.state_dict()}
        model_path = os.path.join(path, model_name)
        torch.save(checkpoint, model_path)
        print("Network model dump to %s" % model_path)

    @staticmethod
    def eval(
        subnet,
        data_path,
        dataset="imagenet",
        n_epochs=0,
        resolution=(224, 224),
        trn_batch_size=128,
        vld_batch_size=250,
        num_workers=4,
        valid_size=None,
        is_test=True,
        log_dir=".tmp/eval",
        measure_latency=None,
        no_logs=False,
        reset_running_statistics=True,
        pmax=2,
        fmax=100,
        amax=5,
        wp=1,
        wf=1 / 40,
        wa=1,
        penalty=10**10,
    ):
        lut = {"cpu": "data/i7-8700K_lut.yaml"}

        info = get_net_info(  ## compute net info (params, etc..)
            subnet,
            (3, resolution, resolution),
            measure_latency=measure_latency,
            print_info=False,
            clean=True,
            lut=lut,
            pmax=pmax,
            fmax=fmax,
            amax=amax,
            wp=wp,
            wf=wf,
            wa=wa,
            penalty=penalty,
        )

        run_config = get_run_config(
            dataset=dataset,
            data_path=data_path,
            image_size=resolution,
            n_epochs=n_epochs,
            train_batch_size=trn_batch_size,
            test_batch_size=vld_batch_size,
            n_worker=num_workers,
            valid_size=valid_size,
        )

        # set the image size. You can set any image size from 192 to 256 here
        run_config.data_provider.assign_active_img_size(resolution)

        if n_epochs > 0:
            # for datasets other than the one supernet was trained on (ImageNet)
            # a few epochs of training need to be applied
            """ these lines are commented to avoid AttributeError: 'MobileNetV3' object has no attribute 'reset_classifier'
            subnet.reset_classifier(
                last_channel=subnet.classifier.in_features,
                n_classes=run_config.data_provider.n_classes, dropout_rate=cfgs.drop_rate)
            """

        run_manager = RunManager(log_dir, subnet, run_config, init=False)

        if reset_running_statistics:
            # run_manager.reset_running_statistics(net=subnet, batch_size=vld_batch_size)
            run_manager.reset_running_statistics(net=subnet)

        if n_epochs > 0:
            cfgs.subnet = subnet
            subnet = run_manager.train(cfgs)

        loss, top1, top5 = run_manager.validate(
            net=subnet, is_test=is_test, no_logs=no_logs
        )

        info["loss"], info["top1"], info["top5"] = loss, top1, top5

        save_path = (
            os.path.join(log_dir, "net.stats") if cfgs.save is None else cfgs.save
        )
        if cfgs.save_config:
            OFAEvaluator.save_net_config(log_dir, subnet, "net.config")
            OFAEvaluator.save_net(log_dir, subnet, "net.init")
        with open(save_path, "w") as handle:
            json.dump(info, handle)

        print(info)
