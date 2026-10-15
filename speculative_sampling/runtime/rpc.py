"""Authenticated, bounded tensor RPC for dedicated text/media workers.

Wire values are CPU tensors and primitive containers, not GPU pointers. torch's
weights-only loader rejects executable pickle objects. Every response carries a
worker epoch; restart invalidates outstanding remote handles. This transport is
intended for a trusted private network (or an SSH tunnel), not public HTTP.
"""
from __future__ import annotations

import hmac
import io
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen


MAX_BYTES = 512 * 1024 * 1024


class RemoteWorkerRestarted(RuntimeError):
    """A response came from a new process epoch.

    The client deliberately keeps its old epoch until the caller explicitly
    accepts this transition and reconstructs every remote handle it needs.
    """

    def __init__(self, previous_epoch, observed_epoch):
        super().__init__("remote worker restarted: outstanding handles are invalid")
        self.previous_epoch = previous_epoch
        self.observed_epoch = observed_epoch


class RemoteCallError(RuntimeError):
    """The contacted worker ran the request and rejected or failed it."""


def pack(value):
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    data = stream.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError("RPC payload exceeds configured bound")
    return data


def unpack(data):
    import torch
    if len(data) > MAX_BYTES:
        raise ValueError("RPC payload exceeds configured bound")
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


class TensorClient:
    def __init__(self, endpoint, token, *, timeout=180):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.epoch = None
        self.lock = threading.Lock()

    def call(self, operation, **arguments):
        with self.lock:
            request = Request(self.endpoint + "/rpc", data=pack({
                "operation": operation, "arguments": arguments, "epoch": self.epoch,
            }), headers={"Authorization": "Bearer " + self.token,
                         "Content-Type": "application/octet-stream"}, method="POST")
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read(MAX_BYTES + 1)
            result = unpack(data)
            if self.epoch is not None and result["epoch"] != self.epoch:
                raise RemoteWorkerRestarted(self.epoch, result["epoch"])
            self.epoch = result["epoch"]
            if result.get("error"):
                raise RemoteCallError(result["error"])
            return result["value"]

    def accept_restart(self, restart):
        """Adopt one observed epoch transition; handles remain caller-owned."""
        if not isinstance(restart, RemoteWorkerRestarted):
            raise TypeError("accept_restart requires RemoteWorkerRestarted")
        with self.lock:
            if self.epoch != restart.previous_epoch:
                raise RuntimeError("client epoch changed before restart was accepted")
            self.epoch = restart.observed_epoch


def serve(service, bind, port, token):
    if not token or len(token) < 24:
        raise ValueError("use a random service token of at least 24 characters")
    epoch = uuid.uuid4().hex
    lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.path != "/rpc" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            ):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    self.send_error(413)
                    return
                self.connection.settimeout(180)
                request = unpack(self.rfile.read(length))
                if request.get("epoch") not in (None, epoch):
                    raise RuntimeError("stale worker epoch")
                operation = request["operation"]
                if operation not in service.operations:
                    raise ValueError("unsupported service operation")
                with lock:
                    value = getattr(service, operation)(**request["arguments"])
                result = {"epoch": epoch, "value": value}
            except Exception as exc:
                result = {"epoch": epoch, "error": f"{type(exc).__name__}: {exc}"}
            data = pack(result)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer((bind, port), Handler)
    server.daemon_threads = True
    return server
