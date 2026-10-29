"""Demo-path test: export two tiny graph layers to ONNX for Unity Sentis.

Run in the rift-train environment (NOT rift-teacher), from the repo root:

    python scripts/export_sentis_test.py

Both models compute the same graph layer on the same small graph:
each cell sums its neighbours' features, multiplies by a weight matrix, then ReLU.

  Model A ("dense"):   out = relu(adjacency @ x @ W)
                       plain matrix multiplies; the safe fallback
  Model B ("scatter"): out = relu(index_add(dst, (x @ W)[src]))
                       gather + scatter-add, the way graph libraries
                       (PyTorch Geometric) actually pass messages

Sentis' docs say its GPUCompute backend does not support scatter with a
reduction, so model B may fail there. That is exactly what this test checks.

Outputs in outputs/sentis_test/ (gitignored; anyone can regenerate them):
    model_a_dense.onnx, model_b_scatter.onnx
    expected_output.txt   the correct answer, one number per line (6 x 8 values)
Copy these three files plus unity/SentisTest/SentisSmokeTest.cs into a Unity
project's Assets folder. The Unity script rebuilds the same inputs and compares.
"""

from pathlib import Path

import numpy as np
import torch

N, F, H = 6, 4, 8  # cells, input features per cell, output features per cell
# Small test graph: a ring of 6 cells plus one chord (0-3), links in both directions.
UNDIRECTED = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
OUT_DIR = Path("outputs/sentis_test")


def test_inputs():
    """Same formulas as SentisSmokeTest.cs, so Unity can rebuild identical inputs."""
    x = np.array([[np.sin(0.5 * (i * F + j)) for j in range(F)] for i in range(N)], np.float32)
    src = np.array([a for a, b in UNDIRECTED] + [b for a, b in UNDIRECTED], np.int64)
    dst = np.array([b for a, b in UNDIRECTED] + [a for a, b in UNDIRECTED], np.int64)
    adj = np.zeros((N, N), np.float32)
    adj[dst, src] = 1.0  # row = receiving cell, column = sending cell
    return x, src, dst, adj


def weights():
    return torch.tensor([[np.cos(0.3 * (j * H + k)) for k in range(H)] for j in range(F)],
                        dtype=torch.float32)


class DenseLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(weights())

    def forward(self, x, adj):
        return torch.relu(adj @ x @ self.W)


class ScatterLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(weights())

    def forward(self, x, src, dst):
        messages = (x @ self.W)[src]  # one message per link, from its sending cell
        total = torch.zeros(x.shape[0], self.W.shape[1], dtype=x.dtype)
        return torch.relu(total.index_add(0, dst, messages))  # add up messages per cell


def export(model, args, names, path, dynamic):
    # external_data=False keeps the weights inside the .onnx file (Unity wants one file).
    torch.onnx.export(model, args, str(path), input_names=names, output_names=["out"],
                      dynamic_shapes=dynamic, opset_version=18, dynamo=True,
                      external_data=False)


def main():
    import onnx

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, src, dst, adj = test_inputs()
    tx, tsrc, tdst, tadj = map(torch.from_numpy, (x, src, dst, adj))

    a, b = DenseLayer().eval(), ScatterLayer().eval()
    with torch.no_grad():
        out_a, out_b = a(tx, tadj), b(tx, tsrc, tdst)
    assert torch.allclose(out_a, out_b, atol=1e-5), "the two layers should agree"

    # Cell and link counts may change between shapes, so those sizes stay dynamic.
    cells, links = torch.export.Dim("cells"), torch.export.Dim("links")
    export(a, (tx, tadj), ["x", "adj"], OUT_DIR / "model_a_dense.onnx",
           {"x": {0: cells}, "adj": {0: cells, 1: cells}})
    export(b, (tx, tsrc, tdst), ["x", "src", "dst"], OUT_DIR / "model_b_scatter.onnx",
           {"x": {0: cells}, "src": {0: links}, "dst": {0: links}})
    np.savetxt(OUT_DIR / "expected_output.txt", out_a.numpy().ravel(), fmt="%.6f")

    for name in ("model_a_dense.onnx", "model_b_scatter.onnx"):
        m = onnx.load(OUT_DIR / name)
        onnx.checker.check_model(m)
        ops = []
        for n in m.graph.node:
            reduction = [a.s.decode() for a in n.attribute if a.name == "reduction"]
            ops.append(n.op_type + (f"(reduction={reduction[0]})" if reduction else ""))
        print(f"{name}: opset {m.opset_import[0].version}, operators: {', '.join(sorted(set(ops)))}")
    print(f"Saved models and expected_output.txt in {OUT_DIR}")


if __name__ == "__main__":
    main()
