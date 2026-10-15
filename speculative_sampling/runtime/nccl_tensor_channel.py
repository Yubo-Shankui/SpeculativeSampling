"""A dedicated two-rank NCCL channel for DiT-to-media GPU tensors.

This deliberately does not join the DiT model-parallel default process group.
The DiT rank-0 process and the media process rendezvous through a separate
TCPStore and own a separate ProcessGroupNCCL.
"""
from __future__ import annotations

from datetime import timedelta


DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


def tensor_descriptor(tensor):
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in DTYPES:
        raise ValueError(f"unsupported NCCL tensor dtype: {tensor.dtype}")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("NCCL media tensors must be contiguous CUDA tensors")
    return {"shape": list(tensor.shape), "dtype": name}


class NcclTensorChannel:
    """Independent rank-0/rank-1 ProcessGroupNCCL with ordered P2P sends."""

    def __init__(self, host, port, *, rank, device, timeout_seconds=600):
        import torch
        import torch.distributed as dist

        if rank not in (0, 1):
            raise ValueError("NCCL tensor channel rank must be 0 or 1")
        self.torch = torch
        self.rank = rank
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("NCCL tensor channel requires a CUDA device")
        timeout = timedelta(seconds=float(timeout_seconds))
        self.store = dist.TCPStore(
            str(host), int(port), 2, rank == 0, timeout, wait_for_workers=True)
        # Kept stable for wire compatibility with already deployed LTX peers;
        # the channel itself is model-agnostic.
        group_name = "ltx-media-p2p"
        group_store = dist.PrefixStore(f"{group_name}/", self.store)
        backend_store = dist.PrefixStore("cuda/", group_store)
        options = dist.ProcessGroupNCCL.Options()
        options._timeout = timeout
        # _new_process_group_helper normally supplies these identities. Since
        # this channel intentionally coexists with an unrelated default DiT
        # group, construct the backend directly but preserve the same setup.
        options.global_ranks_in_group = [0, 1]
        options.group_name = group_name
        self.group = dist.ProcessGroupNCCL(backend_store, rank, 2, options)
        # ProcessGroupNCCL initializes P2P communicators lazily. Pay that
        # one-time cost while workers are still starting, never on the first
        # browser proposal.
        probe = torch.zeros(1, dtype=torch.uint8, device=self.device)
        if rank == 0:
            self.group.send([probe], 1, 0).wait()
        else:
            self.group.recv([probe], 0, 0).wait()
        torch.cuda.synchronize(self.device)

    def describe(self, video, audio):
        descriptors = {
            "video": tensor_descriptor(video),
            "audio": tensor_descriptor(audio),
        }
        if video.dtype != audio.dtype:
            raise ValueError("video and audio must share one NCCL payload dtype")
        return descriptors

    def send(self, video, audio):
        if self.rank != 0:
            raise RuntimeError("only tensor-channel rank 0 can send")
        descriptors = self.describe(video, audio)
        # One fused send avoids ordering/communicator-init hazards from issuing
        # multiple unbatched NCCL P2P operations on a fresh process group.
        payload = self.torch.cat((video.reshape(-1), audio.reshape(-1)))
        work = self.group.send([payload], 1, 0)
        return descriptors, (work, payload)

    def receive(self, descriptors):
        if self.rank != 1:
            raise RuntimeError("only tensor-channel rank 1 can receive")
        layouts = []
        for name in ("video", "audio"):
            descriptor = descriptors.get(name)
            if not isinstance(descriptor, dict) or set(descriptor) != {"shape", "dtype"}:
                raise ValueError(f"invalid {name} tensor descriptor")
            shape = descriptor["shape"]
            dtype = descriptor["dtype"]
            if (not isinstance(shape, list) or not shape
                    or any(type(value) is not int or value <= 0 for value in shape)
                    or dtype not in DTYPES):
                raise ValueError(f"invalid {name} tensor descriptor")
            layouts.append((shape, dtype))
        if layouts[0][1] != layouts[1][1]:
            raise ValueError("video and audio must share one NCCL payload dtype")
        counts = []
        for shape, _ in layouts:
            count = 1
            for value in shape:
                count *= value
            counts.append(count)
        payload = self.torch.empty(
            sum(counts), dtype=getattr(self.torch, DTYPES[layouts[0][1]]),
            device=self.device)
        self.group.recv([payload], 0, 0).wait()
        video = payload[:counts[0]].view(layouts[0][0])
        audio = payload[counts[0]:].view(layouts[1][0])
        return video, audio

    @staticmethod
    def wait(pending):
        work, payload = pending
        work.wait()
        # Keep the fused payload alive until NCCL has consumed it.
        del payload
