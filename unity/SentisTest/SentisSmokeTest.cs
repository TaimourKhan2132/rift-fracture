// RIFT demo-path test (week-1 kill test).
//
// Runs the two test models from scripts/export_sentis_test.py on every Sentis
// backend (CPU, GPUCompute, GPUPixel) and compares each result with
// expected_output.txt. Both models compute the same graph layer:
//   Model A: plain matrix multiplies (the safe fallback)
//   Model B: gather + scatter-add, the way graph libraries pass messages
// Sentis' docs say GPUCompute lacks scatter reductions, so Model B may fail
// there. That is what this test is for.
//
// Setup:
//   1. Run scripts/export_sentis_test.py (rift-train environment).
//   2. In the Unity project, make a folder Assets/SentisTest and copy in:
//      outputs/sentis_test/model_a_dense.onnx, model_b_scatter.onnx,
//      expected_output.txt, and this script.
//   3. In the scene, create an empty GameObject, add this component, and drag
//      the two models and the text file onto its three slots in the Inspector.
//   4. Press Play and read the Console: one line per model and backend.

using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class SentisSmokeTest : MonoBehaviour
{
    public ModelAsset modelA;         // model_a_dense.onnx
    public ModelAsset modelB;         // model_b_scatter.onnx
    public TextAsset expectedOutput;  // expected_output.txt

    // Same test graph and formulas as test_inputs() in export_sentis_test.py.
    const int N = 6, F = 4;
    static readonly int[,] Undirected = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 0 }, { 0, 3 } };

    float[] x, adj;
    int[] src, dst;

    void Start()
    {
        float[] expected = expectedOutput.text
            .Split(new[] { '\n', '\r', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => float.Parse(s, CultureInfo.InvariantCulture))
            .ToArray();

        x = new float[N * F];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < F; j++)
                x[i * F + j] = Mathf.Sin(0.5f * (i * F + j));

        int half = Undirected.GetLength(0), links = 2 * half;
        src = new int[links];
        dst = new int[links];
        for (int k = 0; k < half; k++)
        {
            src[k] = Undirected[k, 0]; dst[k] = Undirected[k, 1];
            src[k + half] = Undirected[k, 1]; dst[k + half] = Undirected[k, 0];
        }
        adj = new float[N * N];
        for (int k = 0; k < links; k++)
            adj[dst[k] * N + src[k]] = 1f;  // row = receiving cell, column = sending cell

        Debug.Log($"[RIFT] Sentis smoke test: {expected.Length} expected values, {links} links");
        int passed = 0, total = 0;
        foreach (var backend in new[] { BackendType.CPU, BackendType.GPUCompute, BackendType.GPUPixel })
        {
            total += 2;
            if (Check("Model A (dense)  ", modelA, backend, expected, worker =>
                {
                    var tx = new Tensor<float>(new TensorShape(N, F), x);
                    var tadj = new Tensor<float>(new TensorShape(N, N), adj);
                    worker.SetInput("x", tx);
                    worker.SetInput("adj", tadj);
                    return new Tensor[] { tx, tadj };
                })) passed++;
            if (Check("Model B (scatter)", modelB, backend, expected, worker =>
                {
                    var tx = new Tensor<float>(new TensorShape(N, F), x);
                    var tsrc = new Tensor<int>(new TensorShape(links), src);
                    var tdst = new Tensor<int>(new TensorShape(links), dst);
                    worker.SetInput("x", tx);
                    worker.SetInput("src", tsrc);
                    worker.SetInput("dst", tdst);
                    return new Tensor[] { tx, tsrc, tdst };
                })) passed++;
        }
        Debug.Log($"[RIFT] Done: {passed} of {total} model/backend combinations passed.");
    }

    // Loads the model on one backend, runs it 21 times (the first run warms up),
    // and compares the last result with the expected numbers.
    // setInputs creates and sets the input tensors and returns them so they can be freed.
    bool Check(string label, ModelAsset asset, BackendType backend, float[] expected,
               Func<Worker, Tensor[]> setInputs)
    {
        try
        {
            Model model = ModelLoader.Load(asset);
            using var worker = new Worker(model, backend);
            float[] result = null;
            var timer = new Stopwatch();
            for (int run = 0; run < 21; run++)
            {
                if (run == 1) timer.Start();
                Tensor[] inputs = setInputs(worker);
                worker.Schedule();
                var output = worker.PeekOutput() as Tensor<float>;  // owned by the worker
                result = output.DownloadToArray();  // waits for the result, so timing is honest
                foreach (var t in inputs) t.Dispose();  // free input memory every run
            }
            timer.Stop();

            if (result.Length != expected.Length)
            {
                Debug.LogError($"[RIFT] {label} | {backend,-10} | FAIL | got {result.Length} values, expected {expected.Length}");
                return false;
            }
            float maxError = expected.Zip(result, (e, r) => Mathf.Abs(e - r)).Max();
            bool pass = maxError < 1e-4f;
            string line = $"[RIFT] {label} | {backend,-10} | {(pass ? "PASS" : "FAIL")} | " +
                          $"max error {maxError:E1} | {timer.Elapsed.TotalMilliseconds / 20:F3} ms per run";
            if (pass) Debug.Log(line); else Debug.LogError(line);
            return pass;
        }
        catch (Exception ex)
        {
            Debug.LogError($"[RIFT] {label} | {backend,-10} | FAIL | {ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }
}
