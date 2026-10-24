import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isSklearnPickle, parseSklearn } from "../src/formats/sklearn";

// Fixtures: iris-fitted models from fixtures/scripts/make_sklearn.py (sklearn 1.7.2).
const fx = (f: string) => readFileSync(fileURLToPath(new URL(`./fixtures/sklearn/${f}`, import.meta.url)));
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe("scikit-learn Pipeline(StandardScaler, LogisticRegression)", () => {
  for (const [file, container] of [
    ["pipeline_logreg.pkl", "pickle"],
    ["pipeline_logreg_p5.pkl", "pickle"],
    ["pipeline_logreg.joblib", "joblib"],
    ["pipeline_logreg_z.joblib", "joblib (zlib)"],
  ] as const) {
    it(`parses ${file}`, () => {
      const buf = fx(file);
      expect(isSklearnPickle(buf)).toBe(true);
      const m = parseSklearn(buf, file);
      expect(m.format).toBe(`scikit-learn (${container})`);
      expect(m.producer).toBe("Pipeline, sklearn 1.7.2");
      expect(m.layers.map((l) => l.type)).toEqual(["Input", "StandardScaler", "LogisticRegression"]);
      expect(m.edges).toEqual([[0, 1], [1, 2]]);
      expect(m.inputShape).toEqual([1, 4]);
      expect(m.outputShape).toEqual([1, 3]);
      const [, scaler, clf] = m.layers;
      expect(scaler.params).toBe(8); // mean_ + scale_
      expect(scaler.w?.mean).toBeCloseTo((5.84333333 + 3.05733333 + 3.758 + 1.19933333) / 4, 3);
      expect(clf.params).toBe(12 + 3); // coef_ 3×4 + intercept_
      expect(clf.macs).toBe(12);
      expect(clf.w?.shape).toEqual([3, 4]);
      expect(clf.w?.min).toBeCloseTo(-1.92814, 3);
      expect(clf.w?.max).toBeCloseTo(2.63928, 3);
      expect(clf.attr.C).toBe(1);
      expect(clf.attr.max_iter).toBe(500);
      expect(clf.attr.coef_).toBe("float64[3×4]");
      expect(clf.group).toBe("output");
    });
  }
});

describe("scikit-learn MLPClassifier", () => {
  for (const file of ["mlp.pkl", "mlp.joblib"]) {
    it(`expands coefs_ into dense layers (${file})`, () => {
      const m = parseSklearn(fx(file), file);
      expect(m.layers.map((l) => l.type)).toEqual(["Input", "Dense+relu", "Dense+relu", "Dense+softmax"]);
      expect(m.layers.map((l) => l.shape)).toEqual(["4", "16", "8", "3"]);
      expect(m.edges).toEqual([[0, 1], [1, 2], [2, 3]]);
      // (4·16+16) + (16·8+8) + (8·3+3)
      expect(sum(m.layers.map((l) => l.params))).toBe(80 + 136 + 27);
      expect(sum(m.layers.map((l) => l.macs))).toBe(64 + 128 + 24);
      const w = m.layers[1].w!;
      expect(w.shape).toEqual([4, 16]);
      expect(w.min).toBeCloseTo(-0.7017, 3);
      expect(w.mean).toBeCloseTo(0.02685, 3);
      expect(w.std).toBeCloseTo(0.30263, 3);
    });
  }
});

describe("scikit-learn RandomForestClassifier", () => {
  for (const file of ["forest.pkl", "forest.joblib"]) {
    it(`builds parallel trees with node counts (${file})`, () => {
      const m = parseSklearn(fx(file), file);
      expect(m.layers.map((l) => l.type)).toEqual([
        "Input", "DecisionTreeClassifier", "DecisionTreeClassifier", "DecisionTreeClassifier", "RandomForestClassifier",
      ]);
      expect(m.edges).toEqual([[0, 1], [0, 2], [0, 3], [1, 4], [2, 4], [3, 4]]);
      expect(m.layers.slice(1, 4).map((l) => l.params)).toEqual([9, 9, 13]); // tree_.node_count
      expect(m.layers[1].flops).toBe(3); // max_depth comparisons
      expect(m.layers[4].attr.n_estimators).toBe(3);
      expect(m.layers[4].attr.total_member_params).toBe(31);
      expect(m.outputShape).toEqual([1, 3]);
    });
  }
});

describe("scikit-learn ColumnTransformer pipeline", () => {
  for (const file of ["column_pipeline.pkl", "column_pipeline.joblib"]) {
    it(`renders parallel branches and a concat (${file})`, () => {
      const m = parseSklearn(fx(file), file);
      expect(m.layers.map((l) => l.type)).toEqual(["Input", "ColumnTransformer", "StandardScaler", "PCA", "hstack", "LogisticRegression"]);
      expect(m.edges).toEqual([[0, 1], [1, 2], [1, 3], [2, 4], [3, 4], [4, 5]]);
      expect(m.layers[2].shape).toBe("2");
      expect(m.layers[3].params).toBe(4 + 2); // components_ 2×2 + mean_
      expect(m.layers[4].shape).toBe("4");
      expect(m.layers[2].attr.columns).toEqual([0, 1]);
    });
  }
});

describe("scikit-learn fallbacks", () => {
  it("reports unsupported joblib compression as metadata", () => {
    const xz = new Uint8Array([0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00, 0, 0, 0, 0]);
    const m = parseSklearn(xz, "m.joblib");
    expect(m.format).toBe("scikit-learn (joblib (lzma/xz))");
    expect(m.layers).toHaveLength(1);
    expect(m.doc).toMatch(/not supported/);
  });

  it("reports non-sklearn pickles without inventing a graph", () => {
    // pickle.dumps({"a": 1}, protocol=4)
    const p = new Uint8Array([0x80, 0x04, 0x95, 0x0a, 0, 0, 0, 0, 0, 0, 0, 0x7d, 0x94, 0x8c, 0x01, 0x61, 0x94, 0x4b, 0x01, 0x73, 0x2e]);
    expect(isSklearnPickle(p)).toBe(false);
    const m = parseSklearn(p, "d.pkl");
    expect(m.format).toBe("Pickle");
    expect(m.doc).toContain("dict");
  });
});
