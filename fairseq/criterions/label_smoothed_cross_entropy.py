# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True,
        lprobs_old=None, lprobs_mle=None, config=None, sample=None):

    from fairseq_cli.train import model_old, model_mle

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    # Entropy of the current policy
    lprob_ent = lprobs.clone().detach()
    entropy = -torch.sum(lprob_ent * torch.exp(lprob_ent), dim=-1)
    if entropy.dim() < target.dim():
        entropy = entropy.unsqueeze(-1)

    batch_size = sample['target'].shape[0]

    if lprobs_old is None or lprobs_mle is None:
        weight_theta_hat = 1.0
    else:
        weight_theta_hat = lprobs_old.gather(dim=-1, index=target)
        weight_mle = lprobs_mle.gather(dim=-1, index=target)


    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        if lprobs_old is not None or lprobs_mle is not None:
            weight_theta_hat.masked_fill_(pad_mask, 1.0)
        entropy.masked_fill_(pad_mask, 1.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        raise NotImplementedError

    assert entropy.size() == weight_mle.size()

    if lprobs_old is not None or lprobs_mle is not None:
        with torch.no_grad():
            if config.suffix_num > 0:
                def obtain_suffix_weights_kk(weight_fn, kk, entropy):
                    fn_weight_original = weight_fn.clone()

                    if config.reward_type == "sump_ent":
                        fn_weight_original += config.ent_alpha * entropy.clone()
                    elif config.reward_type == "ent":
                        fn_weight_original = config.ent_alpha * entropy.clone()

                    if kk == 0:
                        fn_weight_nextk = fn_weight_original
                    else:
                        fn_weight_nextk = fn_weight_original.clone()
                        fn_weight_nextk = fn_weight_nextk.reshape(batch_size, -1)
                        fn_weight_original = fn_weight_original.reshape(batch_size, -1)

                        fn_weight_nextk[:, :-kk] = fn_weight_original[:, kk:].clone()
                        for aa in range(1, kk+1):
                            if aa <= fn_weight_nextk.shape[1]:
                                fn_weight_nextk[:, -aa].fill_(1.0)
                    
                    if config.reward_type == 'sump':
                        fn_weight_nextk = fn_weight_nextk - config.q_baseline
                    elif config.reward_type == 'logp':
                        fn_weight_nextk = torch.log(fn_weight_nextk+1e-10) - config.q_baseline
                    elif config.reward_type == 'sum_entp':
                        fn_weight_nextk = fn_weight_nextk - config.q_baseline 
                    elif config.reward_type == 'expp':
                        fn_weight_nextk = torch.exp(fn_weight_nextk) - config.q_baseline
                    elif config.reward_type == "ent":
                        fn_weight_nextk = fn_weight_nextk - config.q_baseline

                    fn_weight_nextk = torch.clamp(fn_weight_nextk, min=config.trunc_min)
                    return fn_weight_nextk.reshape(-1, 1)

                if config.suffix_num == 5:
                    try:
                        weight_suffix = obtain_suffix_weights_kk(weight_mle, 0, entropy) + \
                            (config.gamma ** 1) * obtain_suffix_weights_kk(weight_mle, 1, entropy) + \
                            (config.gamma ** 2) * obtain_suffix_weights_kk(weight_mle, 2, entropy) + \
                            (config.gamma ** 3) * obtain_suffix_weights_kk(weight_mle, 3, entropy) + \
                            (config.gamma ** 4) * obtain_suffix_weights_kk(weight_mle, 4, entropy) + \
                            (config.gamma ** 5) * obtain_suffix_weights_kk(weight_mle, 5, entropy)
                    except:  # check sequence length
                        weight_suffix = obtain_suffix_weights_kk(weight_mle, 0, entropy) + \
                            (config.gamma ** 1) * obtain_suffix_weights_kk(weight_mle, 1, entropy) + \
                            (config.gamma ** 2) * obtain_suffix_weights_kk(weight_mle, 2, entropy) + \
                            (config.gamma ** 3) * obtain_suffix_weights_kk(weight_mle, 3, entropy)                     
                else:
                    # Can implement much more elegantly for longer suffix_num!
                    raise NotImplementedError(config.suffix_num)

                b1 = torch.clamp(weight_theta_hat, min=config.iw_min, max=1.0)  # warning
                b2 = weight_suffix
    else: 
        b1 = 1.0
        b2 = 1.0

    nll_loss_new = (b1 * b2) * nll_loss

    if reduce:
        nll_loss = nll_loss.sum()
        nll_loss_new = nll_loss_new.sum()
        smooth_loss = smooth_loss.sum()
        
    eps_i = epsilon / (lprobs.size(-1))
    loss = (1.0 - eps_i) * nll_loss_new + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        from fairseq_cli.train import model_old, model_mle

        model_old.eval()
        model_mle.eval()

        with torch.no_grad():
            net_output_old = model_old(**sample["net_input"])
            net_output_mle = model_mle(**sample["net_input"])
        net_output = model(**sample["net_input"])
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # GOLD loss
        if model.cfg.use_is_obj:
            loss, nll_loss = self.compute_loss(
                model, net_output, sample, reduce=reduce,
                output_old=net_output_old, output_mle=net_output_mle
            )
        else:
            loss, nll_loss = self.compute_loss(
                model, net_output, sample, reduce=reduce,
                output_old=None, output_mle=net_output_mle,
            )

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, output_old=None, output_mle=None):
        from fairseq_cli.train import model_old, model_mle

        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        with torch.no_grad():
            if output_old is not None:
                lprobs_old = model_old.get_normalized_probs(output_old, log_probs=False)
                lprobs_old = lprobs_old.view(-1, lprobs.size(-1))
                lprobs_mle = model_mle.get_normalized_probs(output_mle, log_probs=False)
                lprobs_mle = lprobs_mle.view(-1, lprobs.size(-1))
            else:
                lprobs_old = None
                lprobs_mle = None


        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            lprobs_old=lprobs_old,
            lprobs_mle=lprobs_mle,
            config=model.cfg,
            sample=sample
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
