"""The output head: an adaptive softmax tied to the input tables."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .configuration_budgie import BudgieConfig
from .embeddings import AdaptiveInput

LOSS_CHUNK = 2048  # tokens whose vocabulary logits exist at the same time when computing the loss


class AdaptiveSoftmaxHead(nn.Module):
    """Adaptive softmax over frequency ranks, using the weights of an AdaptiveInput table.

    The first cluster's tokens are scored against that cluster's embedding rows, together with one
    extra "cluster" logit per tail cluster (this module's only parameters, `cluster_vectors`). A
    tail cluster's tokens are then scored inside the cluster, after projecting the hidden state
    down through the same Linear that projects that cluster's embeddings up. P(token) is
    P(cluster) * P(token | cluster). Logits are soft-capped at config.logit_softcap. With no
    adaptive_cutoffs there is one cluster and this is an ordinary tied full softmax.

    The table is passed to each call, not stored, so its weights are not registered twice."""

    def __init__(self, config: BudgieConfig):
        super().__init__()
        self.cuts, self.softcap = config.adaptive_cuts, config.logit_softcap
        n_tail = len(config.adaptive_cutoffs)
        self.cluster_vectors = nn.Parameter(torch.zeros(n_tail, config.hidden_size)) if n_tail else None

    def _cap(self, x):
        x = x.float()
        return self.softcap * torch.tanh(x / self.softcap) if self.softcap else x

    def _logits(self, h, table: AdaptiveInput):
        """h [N, d] -> (first-cluster + cluster-choice logits [N, c0 + n_tail], [logits inside each tail cluster])."""
        head = h @ table.tok[0].weight.t()
        if self.cluster_vectors is not None:
            head = torch.cat([head, h @ self.cluster_vectors.t()], -1)
        tails = [(h @ table.proj[i - 1].weight) @ table.tok[i].weight.t() for i in range(1, len(table.tok))]
        return head, tails

    def _token_nll(self, h, rank, table):
        """Per-token negative log-likelihood [N] (0 where ignored) and the valid mask, for hidden
        states h and target ranks (-1 = ignore)."""
        cuts = self.cuts
        valid, tgt = rank >= 0, rank.clamp_min(0)
        head, tails = self._logits(h, table)
        head = self._cap(head)
        if tails:
            cluster = torch.bucketize(tgt, torch.tensor(cuts[1:-1], device=tgt.device), right=True)
            head_idx = torch.where(cluster == 0, tgt, cuts[1] + cluster - 1)
        else:
            cluster, head_idx = torch.zeros_like(tgt), tgt
        nll = torch.logsumexp(head, -1) - head.gather(1, head_idx[:, None])[:, 0]
        for i, logits in enumerate(tails, start=1):
            logits = self._cap(logits)
            local = (tgt - cuts[i]).clamp(0, cuts[i + 1] - cuts[i] - 1)
            inside = torch.logsumexp(logits, -1) - logits.gather(1, local[:, None])[:, 0]
            nll = nll + torch.where(cluster == i, inside, torch.zeros_like(inside))
        return torch.where(valid, nll, torch.zeros_like(nll)), valid

    def _nll(self, h, rank, table):
        """Summed negative log-likelihood and token count."""
        nll, valid = self._token_nll(h, rank, table)
        return nll.sum(), valid.sum()

    @torch.no_grad()
    def token_nll(self, hidden, labels, table: AdaptiveInput):
        """For evaluation: the negative log-likelihood of every next-token target and its frequency
        rank, both [N]; rank -1 marks an ignored target. Lets a caller split the loss by how
        frequent the token is (e.g. to see what an adaptive softmax costs the rare tokens)."""
        h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        tgt = labels[:, 1:].reshape(-1).to(h.device)
        rank = torch.where(tgt >= 0, table.rank_of[tgt.clamp_min(0)], torch.full_like(tgt, -1))
        nll = torch.cat([self._token_nll(h[s:s + LOSS_CHUNK], rank[s:s + LOSS_CHUNK], table)[0] for s in range(0, h.shape[0], LOSS_CHUNK)])
        return nll, rank

    def loss(self, hidden, labels, table: AdaptiveInput):
        """Mean next-token negative log-likelihood; labels < 0 (-100) are ignored. Token chunks are
        recomputed in backward instead of keeping their vocabulary logits."""
        h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        tgt = labels[:, 1:].reshape(-1).to(h.device)
        rank = torch.where(tgt >= 0, table.rank_of[tgt.clamp_min(0)], torch.full_like(tgt, -1))
        total, count = h.new_zeros((), dtype=torch.float32), rank.new_zeros(())
        for start in range(0, h.shape[0], LOSS_CHUNK):
            args = (h[start:start + LOSS_CHUNK], rank[start:start + LOSS_CHUNK])
            if torch.is_grad_enabled() and h.requires_grad:
                nll, n = checkpoint(lambda a, b: self._nll(a, b, table), *args, use_reentrant=False)
            else:
                nll, n = self._nll(*args, table)
            total, count = total + nll, count + n
        return total / count.clamp_min(1)

    def log_probs(self, hidden, table: AdaptiveInput):
        """[..., d] -> [..., vocab] log-probabilities, indexed by token id."""
        shape = hidden.shape[:-1]
        head, tails = self._logits(hidden.reshape(-1, hidden.shape[-1]), table)
        lp_head = torch.log_softmax(self._cap(head), -1)
        parts = [lp_head[:, : self.cuts[1]]]
        for i, logits in enumerate(tails, start=1):
            parts.append(torch.log_softmax(self._cap(logits), -1) + lp_head[:, self.cuts[1] + i - 1 : self.cuts[1] + i])
        return torch.cat(parts, -1).index_select(-1, table.rank_of).reshape(*shape, -1)
