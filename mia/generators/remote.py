"""Run a generator in another Python environment, behind the same interface.

TabPFN, Opacus 1.x and DPSynth need Python >= 3.12 and torch >= 2.6; the rest of
this project (and every recorded result) lives on Python 3.9 / torch 1.13 /
numpy 1.23, which the Private-PGM stack pins.  Rather than fork the pipeline,
`base.build` returns a `RemoteGenerator` whenever the requested generator's
`requires` are not met by the running interpreter.  The proxy starts

    <env python> -m mia.generators.remote <reply_fd>

in the environment the generator names, and forwards `fit`, `sample`, `save`
and `load` to it.  That worker imports the very same generator class and runs
it natively, so there is one implementation per generator and the target
pipeline, zoo and attacks cannot tell the difference.

Arrays cross as .npy files (the format is stable across numpy 1 and 2, pickles
are not) and control messages as JSON lines: commands on the worker's stdin,
replies on a dedicated pipe, so the worker's stdout stays free for training
logs.  The worker inherits the caller's environment, including
CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import Generator


@dataclass
class RemoteGenerator(Generator):
    remote_name: str = ""
    remote_env: str = "sota"
    remote_params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = self.remote_name
        self._proc = None
        self._tmp = None
        self._resolved = None
        self._report = {}

    # ── Worker lifecycle ────────────────────────────────────────────────────

    def _start(self):
        if self._proc is not None:
            return
        from .. import paths
        python = paths.ENV_PYTHON[self.remote_env]
        if not python.exists():
            raise FileNotFoundError(
                f"Generator {self.remote_name!r} runs in the {self.remote_env!r} "
                f"environment, but {python} does not exist.  Create it from "
                f"environment-{self.remote_env}.yml or point "
                f"CAMDA_{self.remote_env.upper()}_PYTHON at its interpreter.")
        self._tmp = tempfile.TemporaryDirectory(prefix="mia_remote_")
        reply_r, reply_w = os.pipe()
        self._proc = subprocess.Popen(
            [str(python), "-W", "ignore", "-m", "mia.generators.remote", str(reply_w)],
            stdin=subprocess.PIPE, pass_fds=(reply_w,), cwd=str(paths.PROJECT_ROOT),
            text=True)
        os.close(reply_w)
        self._replies = os.fdopen(reply_r, "r")
        self._call("build", name=self.remote_name,
                   params={**self.remote_params, "seed": self.seed,
                           "device": self.device, "verbose": self.verbose})

    def _call(self, cmd: str, **kw) -> dict:
        self._proc.stdin.write(json.dumps({"cmd": cmd, **kw}) + "\n")
        self._proc.stdin.flush()
        line = self._replies.readline()
        if not line:
            code = self._proc.wait()
            self._proc = None
            raise RuntimeError(f"{self.remote_name} worker died (exit {code}) "
                               f"during {cmd!r}; see its output above")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"{self.remote_name} worker failed in {cmd!r}:\n"
                               f"{reply['error']}")
        self._resolved = reply.get("resolved", self._resolved)
        self._report = reply.get("report", self._report)
        return reply

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __del__(self):
        self.close()

    # ── Generator interface ─────────────────────────────────────────────────

    def fit(self, X, y, n_classes):
        self._start()
        d = Path(self._tmp.name)
        np.save(d / "X.npy", np.asarray(X, dtype=np.float32))
        np.save(d / "y.npy", np.asarray(y, dtype=np.int64))
        self._call("fit", X=str(d / "X.npy"), y=str(d / "y.npy"), n_classes=int(n_classes))
        return self

    def sample(self, n):
        d = Path(self._tmp.name)
        self._call("sample", n=int(n), X=str(d / "Xs.npy"), y=str(d / "ys.npy"))
        return (np.load(d / "Xs.npy").astype(np.float32),
                np.load(d / "ys.npy").astype(np.int64))

    def save(self, path):
        self._start()
        if not self._call("save", path=str(path)).get("ok"):
            raise NotImplementedError(f"{self.remote_name} does not support save()")

    def load(self, path):
        self._start()
        self._call("load", path=str(path))
        return self

    def resolved_params(self):
        self._start()
        return dict(self._resolved or {})

    def report(self):
        return dict(self._report or {})

    def params(self):
        return {**self.resolved_params(), "generator": self.remote_name}


# ─────────────────────────────────────────────────────────────────────────────
# Worker side
# ─────────────────────────────────────────────────────────────────────────────

def _serve(reply_fd: int) -> None:
    import traceback
    from . import base

    replies = os.fdopen(reply_fd, "w")
    gen = None
    for line in sys.stdin:
        msg = json.loads(line)
        cmd = msg.pop("cmd")
        try:
            out = {"ok": True}
            if cmd == "build":
                cls = base.REGISTRY[msg["name"]]
                if not cls.available():
                    raise RuntimeError(f"{msg['name']} needs {cls.requires}, which "
                                       f"{sys.executable} does not provide")
                gen = cls(**msg["params"])
            elif cmd == "fit":
                gen.fit(np.load(msg["X"]), np.load(msg["y"]), msg["n_classes"])
            elif cmd == "sample":
                X, y = gen.sample(msg["n"])
                np.save(msg["X"], np.asarray(X, dtype=np.float32))
                np.save(msg["y"], np.asarray(y, dtype=np.int64))
            elif cmd == "save":
                try:
                    gen.save(Path(msg["path"]))
                except NotImplementedError:
                    out = {"ok": False}
            elif cmd == "load":
                gen.load(Path(msg["path"]))
            else:
                raise ValueError(f"unknown command {cmd!r}")
            out["resolved"] = gen.resolved_params()
            out["report"] = gen.report()
        except Exception:
            out = {"error": traceback.format_exc()}
        replies.write(json.dumps(out, default=str) + "\n")
        replies.flush()


if __name__ == "__main__":
    _serve(int(sys.argv[1]))
