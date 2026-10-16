"""Fixed global windows for budget-matched Xenium/Visium continuation."""
from __future__ import annotations
import hashlib
import json
import numpy as np


class FixedDomainBatchSampler:
    """Draw a deterministic window before sharding across 1/2/4 GPU ranks.

    Each update has 64 centers. Visium centers are balanced across physical
    samples, and domain totals are exact per update. Sampling is with
    replacement across updates; this is an update budget, not a corpus epoch.
    A seek does not replay a mutable RNG or depend on DataLoader prefetch.
    """

    def __init__(self, xenium_indices, visium_indices_by_sample, *,
                 updates, visium_centers=16, world=1, rank=0, seed=42):
        self.xenium = np.asarray(xenium_indices, dtype=np.int64)
        self.visium = {str(k): np.asarray(v, dtype=np.int64)
                       for k, v in sorted(visium_indices_by_sample.items())}
        self.updates = int(updates)
        self.visium_centers = int(visium_centers)
        self.world, self.rank, self.seed = int(world), int(rank), int(seed)
        if self.world not in (1,2,4) or not 0 <= self.rank < self.world:
            raise ValueError("Use one, two or four valid ranks")
        self.local_batch, self.start = 64 // self.world, 0
        if self.updates < 1 or not 0 <= self.visium_centers < 64 or self.seed < 0:
            raise ValueError("Invalid fixed update budget or domain ratio")
        if not len(self.xenium) or len(np.unique(self.xenium)) != len(self.xenium):
            raise ValueError("Xenium center IDs must be nonempty and unique")
        if self.visium_centers and (not self.visium or any(not len(v) for v in self.visium.values())):
            raise ValueError("Mixed training requires at least one usable Visium sample")
        pools = list(self.visium.values())
        all_hd = np.concatenate(pools) if pools else np.empty(0,dtype=np.int64)
        if len(np.unique(all_hd)) != len(all_hd) or np.intersect1d(self.xenium,all_hd).size:
            raise ValueError("Domain and physical-sample index pools must be disjoint")
        self.contract = {"updates":self.updates,"global_batch":64,"visium_centers_per_update":self.visium_centers,
                         "seed":self.seed,"sampling":"fixed update windows; replacement across updates; balanced Visium physical samples",
                         "xenium_pool_sha256":hashlib.sha256(self.xenium.astype("<i8").tobytes()).hexdigest(),
                         "visium_pools":{k:hashlib.sha256(v.astype("<i8").tobytes()).hexdigest() for k,v in self.visium.items()}}
        self.fingerprint = hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()

    def seek(self, epoch, update):
        if epoch != 0 or not 0 <= int(update) <= self.updates:
            raise ValueError("Fixed-budget continuation uses epoch zero and a valid update cursor")
        self.start = int(update)

    def global_window(self, update):
        if not 0 <= int(update) < self.updates:
            raise IndexError(update)
        update = int(update)
        xrng = np.random.default_rng(np.random.SeedSequence([self.seed,update,4101]))
        x = xrng.choice(self.xenium,64-self.visium_centers,
                        replace=len(self.xenium)<64-self.visium_centers)
        selected = [x]
        if self.visium_centers:
            names = list(self.visium)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,update,4102]))
            # Rotate the extra slots, so 16 centers across 5 samples balance
            # over 5 updates rather than always favoring the first sample.
            assignments = (np.arange(self.visium_centers)+update) % len(names)
            for i, name in enumerate(names):
                count = int((assignments==i).sum())
                if count:
                    pool = self.visium[name]
                    selected.append(rng.choice(pool,count,replace=len(pool)<count))
        window = np.concatenate(selected).astype(np.int64,copy=False)
        np.random.default_rng(np.random.SeedSequence([self.seed,update,4103])).shuffle(window)
        if len(window) != 64:
            raise RuntimeError("Global window lost centers")
        return window

    def __len__(self):
        return self.updates-self.start

    def __iter__(self):
        for update in range(self.start,self.updates):
            window = self.global_window(update)
            lo = self.rank*self.local_batch
            yield window[lo:lo+self.local_batch].tolist()
