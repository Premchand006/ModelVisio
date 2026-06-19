var An = Object.defineProperty;
var xn = (t, e, r) => e in t ? An(t, e, { enumerable: !0, configurable: !0, writable: !0, value: r }) : t[e] = r;
var Pe = (t, e, r) => xn(t, typeof e != "symbol" ? e + "" : e, r);
function qe(t) {
  return t == null ? 0 : typeof t == "number" ? t : typeof t.toNumber == "function" ? t.toNumber() : Number(t);
}
function Ie(t) {
  return (t ?? []).map(qe);
}
const xt = {
  0: "undefined",
  1: "float32",
  2: "uint8",
  3: "int8",
  4: "uint16",
  5: "int16",
  6: "int32",
  7: "int64",
  8: "string",
  9: "bool",
  10: "float16",
  11: "float64",
  12: "uint32",
  13: "uint64",
  14: "complex64",
  15: "complex128",
  16: "bfloat16"
}, Hr = {
  1: 4,
  // float32
  2: 1,
  // uint8
  3: 1,
  // int8
  4: 2,
  // uint16
  5: 2,
  // int16
  6: 4,
  // int32
  7: 8,
  // int64
  9: 1,
  // bool
  10: 2,
  // float16
  11: 8,
  // float64
  12: 4,
  // uint32
  13: 8,
  // uint64
  16: 2
  // bfloat16
}, Yt = new TextDecoder();
function Tn(t) {
  const e = {};
  for (const r of t.attribute ?? [])
    r.name && (r.ints && r.ints.length ? e[r.name] = Ie(r.ints) : r.floats && r.floats.length ? e[r.name] = r.floats : r.strings && r.strings.length ? e[r.name] = r.strings.map((o) => Yt.decode(o)) : r.s && r.s.length ? e[r.name] = Yt.decode(r.s) : r.i != null && r.type === 2 ? e[r.name] = qe(r.i) : r.f != null && r.type === 1 ? e[r.name] = r.f : r.i != null ? e[r.name] = qe(r.i) : r.f != null && (e[r.name] = r.f));
  return e;
}
function ve(t, e) {
  const r = t[e];
  return Array.isArray(r) ? r : void 0;
}
function Fe(t, e, r) {
  const o = t[e];
  return typeof o == "number" ? o : r;
}
function er(t, e) {
  const r = t[e];
  return typeof r == "string" ? r : void 0;
}
const Q = -1;
function Fn(t) {
  var r, o, s;
  const e = (s = (o = (r = t.type) == null ? void 0 : r.tensor_type) == null ? void 0 : o.shape) == null ? void 0 : s.dim;
  return e ? e.map((a) => {
    if (a.dim_value != null) {
      const i = typeof a.dim_value == "number" ? a.dim_value : a.dim_value.toNumber();
      return i > 0 ? i : Q;
    }
    return Q;
  }) : null;
}
function In(t) {
  const e = /* @__PURE__ */ new Map(), r = (o) => {
    for (const s of o ?? []) {
      if (!s.name) continue;
      const a = Fn(s);
      a && e.set(s.name, a);
    }
  };
  r(t.input), r(t.value_info), r(t.output);
  for (const o of t.initializer ?? [])
    o.name && e.set(o.name, Ie(o.dims));
  return e;
}
const ze = (t) => t.reduce((e, r) => r === Q ? e : e * r, 1);
function We(t, e, r, o, s, a, i) {
  if (t === Q) return Q;
  if (i === "SAME_UPPER" || i === "SAME_LOWER")
    return Math.ceil(t / r);
  const f = a * (e - 1) + 1;
  return Math.floor((t + o + s - f) / r) + 1;
}
function Rn(t, e) {
  const r = [], o = Math.max(t.length, e.length);
  for (let s = 0; s < o; s++) {
    const a = t[t.length - 1 - s] ?? 1, i = e[e.length - 1 - s] ?? 1;
    a === Q || i === Q ? r.unshift(Q) : r.unshift(Math.max(a, i));
  }
  return r;
}
function Ln(t, e, r) {
  const o = r[0], s = r[1];
  switch (t) {
    case "Relu":
    case "LeakyRelu":
    case "Sigmoid":
    case "Tanh":
    case "Clip":
    case "Elu":
    case "Softmax":
    case "LogSoftmax":
    case "BatchNormalization":
    case "InstanceNormalization":
    case "LRN":
    case "Dropout":
    case "Identity":
    case "Erf":
    case "Gelu":
      return o ? [...o] : null;
    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Pow":
    case "Max":
    case "Min":
      return o && s ? Rn(o, s) : o ? [...o] : null;
    case "Conv":
    case "ConvInteger": {
      if (!o || !s || o.length < 3) return null;
      const [a, , i, f] = o, n = s[0], u = ve(e, "kernel_shape") ?? [s[2], s[3] ?? s[2]], l = ve(e, "strides") ?? [1, 1], c = ve(e, "dilations") ?? [1, 1], h = ve(e, "pads") ?? [0, 0, 0, 0], p = er(e, "auto_pad"), g = We(i, u[0], l[0], h[0], h[2], c[0], p), v = We(f, u[1], l[1] ?? l[0], h[1] ?? h[0], h[3] ?? h[2], c[1] ?? c[0], p);
      return [a, n, g, v];
    }
    case "MaxPool":
    case "AveragePool": {
      if (!o || o.length < 4) return null;
      const [a, i, f, n] = o, u = ve(e, "kernel_shape") ?? [1, 1], l = ve(e, "strides") ?? u, c = ve(e, "pads") ?? [0, 0, 0, 0], h = er(e, "auto_pad"), p = We(f, u[0], l[0], c[0], c[2], 1, h), g = We(n, u[1], l[1] ?? l[0], c[1] ?? c[0], c[3] ?? c[2], 1, h);
      return [a, i, p, g];
    }
    case "GlobalAveragePool":
    case "GlobalMaxPool":
      return !o || o.length < 2 ? null : [o[0], o[1], ...o.slice(2).map(() => 1)];
    case "Gemm": {
      if (!o || !s) return null;
      const a = Fe(e, "transA", 0), i = Fe(e, "transB", 0), f = a ? o[1] : o[0], n = i ? s[0] : s[1];
      return [f, n];
    }
    case "MatMul": {
      if (!o || !s) return null;
      const a = o[o.length - 2] ?? Q, i = s[s.length - 1] ?? Q;
      return [...o.slice(0, -2), a, i];
    }
    case "Flatten": {
      if (!o) return null;
      const a = Fe(e, "axis", 1), i = a < 0 ? o.length + a : a, f = o.slice(0, i), n = o.slice(i), u = (l) => l.some((c) => c === Q);
      return [u(f) ? Q : ze(f), u(n) ? Q : ze(n)];
    }
    case "Concat": {
      const a = Fe(e, "axis", 0), i = r.filter((l) => !!l);
      if (i.length === 0) return null;
      const f = [...i[0]], n = a < 0 ? f.length + a : a;
      let u = 0;
      for (const l of i) {
        if (l[n] === Q) {
          u = Q;
          break;
        }
        u += l[n];
      }
      return f[n] = u, f;
    }
    case "GlobalLpPool":
      return o ? [o[0], o[1], 1, 1] : null;
    default:
      return null;
  }
}
function tr(t) {
  return !t || t.length === 0 ? "?" : t.map((e) => e === Q ? "?" : String(e)).join("×");
}
const Be = { flops: 0, macs: 0 };
function Pn(t, e, r, o) {
  const s = r[0], a = r[1], i = o ? ze(o) : 0;
  switch (t) {
    case "Conv":
    case "ConvInteger": {
      if (!s || !a || !o) return Be;
      const f = s[1] === Q ? 1 : s[1], n = Fe(e, "group", 1), l = (ve(e, "kernel_shape") ?? [a[2] ?? 1, a[3] ?? 1]).reduce((h, p) => h * p, 1), c = i * (f / n) * l;
      return { macs: c, flops: 2 * c };
    }
    case "Gemm": {
      if (!s || !o) return Be;
      const n = (Fe(e, "transA", 0) ? s[0] : s[1]) ?? Q, l = i * (n === Q ? 1 : n);
      return { macs: l, flops: 2 * l };
    }
    case "MatMul": {
      if (!s || !o) return Be;
      const f = s[s.length - 1] === Q ? 1 : s[s.length - 1], n = i * f;
      return { macs: n, flops: 2 * n };
    }
    case "MaxPool":
    case "AveragePool": {
      if (!o) return Be;
      const n = (ve(e, "kernel_shape") ?? [1, 1]).reduce((u, l) => u * l, 1);
      return { macs: 0, flops: i * n };
    }
    case "GlobalAveragePool":
    case "GlobalMaxPool":
      return { macs: 0, flops: s ? ze(s) : 0 };
    case "BatchNormalization":
    case "InstanceNormalization":
    case "LayerNormalization":
      return { macs: 0, flops: 2 * i };
    case "Add":
    case "Sub":
    case "Mul":
    case "Div":
    case "Relu":
    case "LeakyRelu":
    case "Sigmoid":
    case "Tanh":
    case "Clip":
    case "Elu":
    case "Erf":
    case "Gelu":
    case "Pow":
      return { macs: 0, flops: i };
    case "Softmax":
    case "LogSoftmax":
      return { macs: 0, flops: 3 * i };
    default:
      return Be;
  }
}
const Bn = {
  Conv: { math: "y = W ∗ x + b", insight: "Convolution — usually the FLOP hotspot; quantization-sensitive.", qSens: 0.7, group: "backbone" },
  ConvTranspose: { math: "y = Wᵀ ∗ x", insight: "Transposed conv (upsampling). Watch output-shape rounding across compilers.", qSens: 0.7, group: "backbone" },
  Gemm: { math: "y = α·(A·B) + β·C", insight: "Dense/linear layer. Largest weight tensors; the classifier head.", qSens: 0.6, group: "head" },
  MatMul: { math: "y = A · B", insight: "Matrix multiply — attention/projection cost center in transformers.", qSens: 0.6, group: "head" },
  BatchNormalization: { math: "y = γ·(x−μ)/√(σ²+ε) + β", insight: "Fold into the preceding Conv at inference for best INT8 accuracy.", qSens: 0.35, group: "backbone" },
  Relu: { math: "y = max(0, x)", insight: "Activation; fuses into the preceding op on every edge runtime.", qSens: 0.15, group: "backbone" },
  LeakyRelu: { math: "y = max(αx, x)", insight: "Leaky activation; fuses on most runtimes.", qSens: 0.15, group: "backbone" },
  Sigmoid: { math: "y = 1/(1+e^{-x})", insight: "Saturating activation; quantize with care near the tails.", qSens: 0.3, group: "backbone" },
  Tanh: { math: "y = tanh(x)", insight: "Saturating activation.", qSens: 0.3, group: "backbone" },
  MaxPool: { math: "y = max over window", insight: "Spatial downsample, no params.", qSens: 0.1, group: "backbone" },
  AveragePool: { math: "y = mean over window", insight: "Spatial downsample, no params.", qSens: 0.1, group: "backbone" },
  GlobalAveragePool: { math: "y_c = mean_{h,w} x", insight: "Collapses spatial dims — bridge from backbone to head.", qSens: 0.2, group: "neck" },
  GlobalMaxPool: { math: "y_c = max_{h,w} x", insight: "Global pooling bridge.", qSens: 0.2, group: "neck" },
  Flatten: { math: "reshape to 2-D", insight: "No-op on most runtimes; pure reshape.", qSens: 0, group: "neck" },
  Reshape: { math: "reshape", insight: "Metadata-only on most runtimes.", qSens: 0, group: "neck" },
  Transpose: { math: "permute axes", insight: "Layout change; can be costly if it blocks fusion.", qSens: 0, group: "neck" },
  Concat: { math: "concat along axis", insight: "Joins feature maps; common in skip/neck connections.", qSens: 0.1, group: "neck" },
  Add: { math: "y = a + b", insight: "Elementwise add — residual connections.", qSens: 0.2, group: "backbone" },
  Mul: { math: "y = a · b", insight: "Elementwise multiply.", qSens: 0.2, group: "backbone" },
  Softmax: { math: "y_i = e^{x_i}/Σe^{x_j}", insight: "Normalizes logits to probabilities. Verify axis after conversion.", qSens: 0.3, group: "output" },
  LogSoftmax: { math: "log softmax", insight: "Log-probabilities.", qSens: 0.3, group: "output" }
}, Dn = { math: "—", insight: "", qSens: 0.2, group: "backbone" };
function $n(t) {
  return Bn[t] ?? { ...Dn, insight: `${t} operator.` };
}
var be = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
function Mn(t) {
  return t && t.__esModule && Object.prototype.hasOwnProperty.call(t, "default") ? t.default : t;
}
function Cn(t) {
  if (t.__esModule) return t;
  var e = t.default;
  if (typeof e == "function") {
    var r = function o() {
      return this instanceof o ? Reflect.construct(e, arguments, this.constructor) : e.apply(this, arguments);
    };
    r.prototype = e.prototype;
  } else r = {};
  return Object.defineProperty(r, "__esModule", { value: !0 }), Object.keys(t).forEach(function(o) {
    var s = Object.getOwnPropertyDescriptor(t, o);
    Object.defineProperty(r, o, s.get ? s : {
      enumerable: !0,
      get: function() {
        return t[o];
      }
    });
  }), r;
}
var Zr = { exports: {} }, Xr = { exports: {} }, Qr = {}, nt = {}, Kr = Un;
function Un(t, e) {
  for (var r = new Array(arguments.length - 1), o = 0, s = 2, a = !0; s < arguments.length; )
    r[o++] = arguments[s++];
  return new Promise(function(f, n) {
    r[o] = function(l) {
      if (a)
        if (a = !1, l)
          n(l);
        else {
          for (var c = new Array(arguments.length - 1), h = 0; h < c.length; )
            c[h++] = arguments[h];
          f.apply(null, c);
        }
    };
    try {
      t.apply(e || null, r);
    } catch (u) {
      a && (a = !1, n(u));
    }
  });
}
var Yr = {};
(function(t) {
  var e = t;
  e.length = function(f) {
    var n = f.length;
    if (!n)
      return 0;
    for (var u = 0; --n % 4 > 1 && f.charAt(n) === "="; )
      ++u;
    return Math.ceil(f.length * 3) / 4 - u;
  };
  for (var r = new Array(64), o = new Array(123), s = 0; s < 64; )
    o[r[s] = s < 26 ? s + 65 : s < 52 ? s + 71 : s < 62 ? s - 4 : s - 59 | 43] = s++;
  e.encode = function(f, n, u) {
    for (var l = null, c = [], h = 0, p = 0, g; n < u; ) {
      var v = f[n++];
      switch (p) {
        case 0:
          c[h++] = r[v >> 2], g = (v & 3) << 4, p = 1;
          break;
        case 1:
          c[h++] = r[g | v >> 4], g = (v & 15) << 2, p = 2;
          break;
        case 2:
          c[h++] = r[g | v >> 6], c[h++] = r[v & 63], p = 0;
          break;
      }
      h > 8191 && ((l || (l = [])).push(String.fromCharCode.apply(String, c)), h = 0);
    }
    return p && (c[h++] = r[g], c[h++] = 61, p === 1 && (c[h++] = 61)), l ? (h && l.push(String.fromCharCode.apply(String, c.slice(0, h))), l.join("")) : String.fromCharCode.apply(String, c.slice(0, h));
  };
  var a = "invalid encoding";
  e.decode = function(f, n, u) {
    for (var l = u, c = 0, h, p = 0; p < f.length; ) {
      var g = f.charCodeAt(p++);
      if (g === 61 && c > 1)
        break;
      if ((g = o[g]) === void 0)
        throw Error(a);
      switch (c) {
        case 0:
          h = g, c = 1;
          break;
        case 1:
          n[u++] = h << 2 | (g & 48) >> 4, h = g, c = 2;
          break;
        case 2:
          n[u++] = (h & 15) << 4 | (g & 60) >> 2, h = g, c = 3;
          break;
        case 3:
          n[u++] = (h & 3) << 6 | g, c = 0;
          break;
      }
    }
    if (c === 1)
      throw Error(a);
    return u - l;
  }, e.test = function(f) {
    return /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(f);
  };
})(Yr);
var qn = Qe;
function Qe() {
  this._listeners = /* @__PURE__ */ Object.create(null);
}
Qe.prototype.on = function(e, r, o) {
  return (this._listeners[e] || (this._listeners[e] = [])).push({
    fn: r,
    ctx: o || this
  }), this;
};
Qe.prototype.off = function(e, r) {
  if (e === void 0)
    this._listeners = /* @__PURE__ */ Object.create(null);
  else if (r === void 0)
    this._listeners[e] = [];
  else {
    var o = this._listeners[e];
    if (!o)
      return this;
    for (var s = 0; s < o.length; )
      o[s].fn === r ? o.splice(s, 1) : ++s;
  }
  return this;
};
Qe.prototype.emit = function(e) {
  var r = this._listeners[e];
  if (r) {
    for (var o = [], s = 1; s < arguments.length; )
      o.push(arguments[s++]);
    for (s = 0; s < r.length; )
      r[s].fn.apply(r[s++].ctx, o);
  }
  return this;
};
var zn = rr(rr);
function rr(t) {
  return typeof Float32Array < "u" ? function() {
    var e = new Float32Array([-0]), r = new Uint8Array(e.buffer), o = r[3] === 128;
    function s(n, u, l) {
      e[0] = n, u[l] = r[0], u[l + 1] = r[1], u[l + 2] = r[2], u[l + 3] = r[3];
    }
    function a(n, u, l) {
      e[0] = n, u[l] = r[3], u[l + 1] = r[2], u[l + 2] = r[1], u[l + 3] = r[0];
    }
    t.writeFloatLE = o ? s : a, t.writeFloatBE = o ? a : s;
    function i(n, u) {
      return r[0] = n[u], r[1] = n[u + 1], r[2] = n[u + 2], r[3] = n[u + 3], e[0];
    }
    function f(n, u) {
      return r[3] = n[u], r[2] = n[u + 1], r[1] = n[u + 2], r[0] = n[u + 3], e[0];
    }
    t.readFloatLE = o ? i : f, t.readFloatBE = o ? f : i;
  }() : function() {
    function e(o, s, a, i) {
      var f = s < 0 ? 1 : 0;
      if (f && (s = -s), s === 0)
        o(1 / s > 0 ? (
          /* positive */
          0
        ) : (
          /* negative 0 */
          2147483648
        ), a, i);
      else if (isNaN(s))
        o(2143289344, a, i);
      else if (s > 34028234663852886e22)
        o((f << 31 | 2139095040) >>> 0, a, i);
      else if (s < 11754943508222875e-54)
        o((f << 31 | Math.round(s / 1401298464324817e-60)) >>> 0, a, i);
      else {
        var n = Math.floor(Math.log(s) / Math.LN2), u = Math.round(s * Math.pow(2, -n) * 8388608) & 8388607;
        o((f << 31 | n + 127 << 23 | u) >>> 0, a, i);
      }
    }
    t.writeFloatLE = e.bind(null, nr), t.writeFloatBE = e.bind(null, ir);
    function r(o, s, a) {
      var i = o(s, a), f = (i >> 31) * 2 + 1, n = i >>> 23 & 255, u = i & 8388607;
      return n === 255 ? u ? NaN : f * (1 / 0) : n === 0 ? f * 1401298464324817e-60 * u : f * Math.pow(2, n - 150) * (u + 8388608);
    }
    t.readFloatLE = r.bind(null, sr), t.readFloatBE = r.bind(null, or);
  }(), typeof Float64Array < "u" ? function() {
    var e = new Float64Array([-0]), r = new Uint8Array(e.buffer), o = r[7] === 128;
    function s(n, u, l) {
      e[0] = n, u[l] = r[0], u[l + 1] = r[1], u[l + 2] = r[2], u[l + 3] = r[3], u[l + 4] = r[4], u[l + 5] = r[5], u[l + 6] = r[6], u[l + 7] = r[7];
    }
    function a(n, u, l) {
      e[0] = n, u[l] = r[7], u[l + 1] = r[6], u[l + 2] = r[5], u[l + 3] = r[4], u[l + 4] = r[3], u[l + 5] = r[2], u[l + 6] = r[1], u[l + 7] = r[0];
    }
    t.writeDoubleLE = o ? s : a, t.writeDoubleBE = o ? a : s;
    function i(n, u) {
      return r[0] = n[u], r[1] = n[u + 1], r[2] = n[u + 2], r[3] = n[u + 3], r[4] = n[u + 4], r[5] = n[u + 5], r[6] = n[u + 6], r[7] = n[u + 7], e[0];
    }
    function f(n, u) {
      return r[7] = n[u], r[6] = n[u + 1], r[5] = n[u + 2], r[4] = n[u + 3], r[3] = n[u + 4], r[2] = n[u + 5], r[1] = n[u + 6], r[0] = n[u + 7], e[0];
    }
    t.readDoubleLE = o ? i : f, t.readDoubleBE = o ? f : i;
  }() : function() {
    function e(o, s, a, i, f, n) {
      var u = i < 0 ? 1 : 0;
      if (u && (i = -i), i === 0)
        o(0, f, n + s), o(1 / i > 0 ? (
          /* positive */
          0
        ) : (
          /* negative 0 */
          2147483648
        ), f, n + a);
      else if (isNaN(i))
        o(0, f, n + s), o(2146959360, f, n + a);
      else if (i > 17976931348623157e292)
        o(0, f, n + s), o((u << 31 | 2146435072) >>> 0, f, n + a);
      else {
        var l;
        if (i < 22250738585072014e-324)
          l = i / 5e-324, o(l >>> 0, f, n + s), o((u << 31 | l / 4294967296) >>> 0, f, n + a);
        else {
          var c = Math.floor(Math.log(i) / Math.LN2);
          c === 1024 && (c = 1023), l = i * Math.pow(2, -c), o(l * 4503599627370496 >>> 0, f, n + s), o((u << 31 | c + 1023 << 20 | l * 1048576 & 1048575) >>> 0, f, n + a);
        }
      }
    }
    t.writeDoubleLE = e.bind(null, nr, 0, 4), t.writeDoubleBE = e.bind(null, ir, 4, 0);
    function r(o, s, a, i, f) {
      var n = o(i, f + s), u = o(i, f + a), l = (u >> 31) * 2 + 1, c = u >>> 20 & 2047, h = 4294967296 * (u & 1048575) + n;
      return c === 2047 ? h ? NaN : l * (1 / 0) : c === 0 ? l * 5e-324 * h : l * Math.pow(2, c - 1075) * (h + 4503599627370496);
    }
    t.readDoubleLE = r.bind(null, sr, 0, 4), t.readDoubleBE = r.bind(null, or, 4, 0);
  }(), t;
}
function nr(t, e, r) {
  e[r] = t & 255, e[r + 1] = t >>> 8 & 255, e[r + 2] = t >>> 16 & 255, e[r + 3] = t >>> 24;
}
function ir(t, e, r) {
  e[r] = t >>> 24, e[r + 1] = t >>> 16 & 255, e[r + 2] = t >>> 8 & 255, e[r + 3] = t & 255;
}
function sr(t, e) {
  return (t[e] | t[e + 1] << 8 | t[e + 2] << 16 | t[e + 3] << 24) >>> 0;
}
function or(t, e) {
  return (t[e] << 24 | t[e + 1] << 16 | t[e + 2] << 8 | t[e + 3]) >>> 0;
}
var en = {};
(function(t) {
  var e = t, r = "�";
  e.length = function(s) {
    for (var a = 0, i = 0, f = 0; f < s.length; ++f)
      i = s.charCodeAt(f), i < 128 ? a += 1 : i < 2048 ? a += 2 : (i & 64512) === 55296 && (s.charCodeAt(f + 1) & 64512) === 56320 ? (++f, a += 4) : a += 3;
    return a;
  }, e.read = function(s, a, i) {
    if (i - a < 1)
      return "";
    for (var f = "", n = a; n < i; ) {
      var u = s[n++];
      if (u <= 127)
        f += String.fromCharCode(u);
      else if (u >= 192 && u < 224) {
        var l = (u & 31) << 6 | s[n++] & 63;
        f += l >= 128 ? String.fromCharCode(l) : r;
      } else if (u >= 224 && u < 240) {
        var c = (u & 15) << 12 | (s[n++] & 63) << 6 | s[n++] & 63;
        f += c >= 2048 ? String.fromCharCode(c) : r;
      } else if (u >= 240) {
        var h = (u & 7) << 18 | (s[n++] & 63) << 12 | (s[n++] & 63) << 6 | s[n++] & 63;
        h < 65536 || h > 1114111 ? f += r : (h -= 65536, f += String.fromCharCode(55296 + (h >> 10)), f += String.fromCharCode(56320 + (h & 1023)));
      }
    }
    return f;
  }, e.write = function(s, a, i) {
    for (var f = i, n, u, l = 0; l < s.length; ++l)
      n = s.charCodeAt(l), n < 128 ? a[i++] = n : n < 2048 ? (a[i++] = n >> 6 | 192, a[i++] = n & 63 | 128) : (n & 64512) === 55296 && ((u = s.charCodeAt(l + 1)) & 64512) === 56320 ? (n = 65536 + ((n & 1023) << 10) + (u & 1023), ++l, a[i++] = n >> 18 | 240, a[i++] = n >> 12 & 63 | 128, a[i++] = n >> 6 & 63 | 128, a[i++] = n & 63 | 128) : (a[i++] = n >> 12 | 224, a[i++] = n >> 6 & 63 | 128, a[i++] = n & 63 | 128);
    return i - f;
  };
})(en);
var jn = Vn;
function Vn(t, e, r) {
  var o = r || 8192, s = o >>> 1, a = null, i = o;
  return function(n) {
    if (n < 1 || n > s)
      return t(n);
    i + n > o && (a = t(o), i = 0);
    var u = e.call(a, i, i += n);
    return i & 7 && (i = (i | 7) + 1), u;
  };
}
var it, ar;
function Jn() {
  if (ar) return it;
  ar = 1, it = e;
  var t = me();
  function e(a, i) {
    this.lo = a >>> 0, this.hi = i >>> 0;
  }
  var r = e.zero = new e(0, 0);
  r.toNumber = function() {
    return 0;
  }, r.zzEncode = r.zzDecode = function() {
    return this;
  }, r.length = function() {
    return 1;
  };
  var o = e.zeroHash = "\0\0\0\0\0\0\0\0";
  e.fromNumber = function(i) {
    if (i === 0)
      return r;
    var f = i < 0;
    f && (i = -i);
    var n = i >>> 0, u = (i - n) / 4294967296 >>> 0;
    return f && (u = ~u >>> 0, n = ~n >>> 0, ++n > 4294967295 && (n = 0, ++u > 4294967295 && (u = 0))), new e(n, u);
  }, e.from = function(i) {
    if (typeof i == "number")
      return e.fromNumber(i);
    if (t.isString(i))
      if (t.Long)
        i = t.Long.fromString(i);
      else
        return e.fromNumber(parseInt(i, 10));
    return i.low || i.high ? new e(i.low >>> 0, i.high >>> 0) : r;
  }, e.prototype.toNumber = function(i) {
    if (!i && this.hi >>> 31) {
      var f = ~this.lo + 1 >>> 0, n = ~this.hi >>> 0;
      return f || (n = n + 1 >>> 0), -(f + n * 4294967296);
    }
    return this.lo + this.hi * 4294967296;
  }, e.prototype.toLong = function(i) {
    return t.Long ? new t.Long(this.lo | 0, this.hi | 0, !!i) : { low: this.lo | 0, high: this.hi | 0, unsigned: !!i };
  };
  var s = String.prototype.charCodeAt;
  return e.fromHash = function(i) {
    return i === o ? r : new e(
      (s.call(i, 0) | s.call(i, 1) << 8 | s.call(i, 2) << 16 | s.call(i, 3) << 24) >>> 0,
      (s.call(i, 4) | s.call(i, 5) << 8 | s.call(i, 6) << 16 | s.call(i, 7) << 24) >>> 0
    );
  }, e.prototype.toHash = function() {
    return String.fromCharCode(
      this.lo & 255,
      this.lo >>> 8 & 255,
      this.lo >>> 16 & 255,
      this.lo >>> 24,
      this.hi & 255,
      this.hi >>> 8 & 255,
      this.hi >>> 16 & 255,
      this.hi >>> 24
    );
  }, e.prototype.zzEncode = function() {
    var i = this.hi >> 31;
    return this.hi = ((this.hi << 1 | this.lo >>> 31) ^ i) >>> 0, this.lo = (this.lo << 1 ^ i) >>> 0, this;
  }, e.prototype.zzDecode = function() {
    var i = -(this.lo & 1);
    return this.lo = ((this.lo >>> 1 | this.hi << 31) ^ i) >>> 0, this.hi = (this.hi >>> 1 ^ i) >>> 0, this;
  }, e.prototype.length = function() {
    var i = this.lo, f = (this.lo >>> 28 | this.hi << 4) >>> 0, n = this.hi >>> 24;
    return n === 0 ? f === 0 ? i < 16384 ? i < 128 ? 1 : 2 : i < 2097152 ? 3 : 4 : f < 16384 ? f < 128 ? 5 : 6 : f < 2097152 ? 7 : 8 : n < 128 ? 9 : 10;
  }, it;
}
var He = { exports: {} }, ur;
function Gn() {
  return ur || (ur = 1, function(t, e) {
    (function(r, o) {
      function s(a) {
        return a.default || a;
      }
      o(e), t.exports = s(e);
    })(
      typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : be,
      function(r) {
        Object.defineProperty(r, "__esModule", {
          value: !0
        }), r.default = void 0;
        /**
         * @license
         * Copyright 2009 The Closure Library Authors
         * Copyright 2020 Daniel Wirtz / The long.js Authors.
         *
         * Licensed under the Apache License, Version 2.0 (the "License");
         * you may not use this file except in compliance with the License.
         * You may obtain a copy of the License at
         *
         *     http://www.apache.org/licenses/LICENSE-2.0
         *
         * Unless required by applicable law or agreed to in writing, software
         * distributed under the License is distributed on an "AS IS" BASIS,
         * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
         * See the License for the specific language governing permissions and
         * limitations under the License.
         *
         * SPDX-License-Identifier: Apache-2.0
         */
        var o = null;
        try {
          o = new WebAssembly.Instance(
            new WebAssembly.Module(
              new Uint8Array([
                // \0asm
                0,
                97,
                115,
                109,
                // version 1
                1,
                0,
                0,
                0,
                // section "type"
                1,
                13,
                2,
                // 0, () => i32
                96,
                0,
                1,
                127,
                // 1, (i32, i32, i32, i32) => i32
                96,
                4,
                127,
                127,
                127,
                127,
                1,
                127,
                // section "function"
                3,
                7,
                6,
                // 0, type 0
                0,
                // 1, type 1
                1,
                // 2, type 1
                1,
                // 3, type 1
                1,
                // 4, type 1
                1,
                // 5, type 1
                1,
                // section "global"
                6,
                6,
                1,
                // 0, "high", mutable i32
                127,
                1,
                65,
                0,
                11,
                // section "export"
                7,
                50,
                6,
                // 0, "mul"
                3,
                109,
                117,
                108,
                0,
                1,
                // 1, "div_s"
                5,
                100,
                105,
                118,
                95,
                115,
                0,
                2,
                // 2, "div_u"
                5,
                100,
                105,
                118,
                95,
                117,
                0,
                3,
                // 3, "rem_s"
                5,
                114,
                101,
                109,
                95,
                115,
                0,
                4,
                // 4, "rem_u"
                5,
                114,
                101,
                109,
                95,
                117,
                0,
                5,
                // 5, "get_high"
                8,
                103,
                101,
                116,
                95,
                104,
                105,
                103,
                104,
                0,
                0,
                // section "code"
                10,
                191,
                1,
                6,
                // 0, "get_high"
                4,
                0,
                35,
                0,
                11,
                // 1, "mul"
                36,
                1,
                1,
                126,
                32,
                0,
                173,
                32,
                1,
                173,
                66,
                32,
                134,
                132,
                32,
                2,
                173,
                32,
                3,
                173,
                66,
                32,
                134,
                132,
                126,
                34,
                4,
                66,
                32,
                135,
                167,
                36,
                0,
                32,
                4,
                167,
                11,
                // 2, "div_s"
                36,
                1,
                1,
                126,
                32,
                0,
                173,
                32,
                1,
                173,
                66,
                32,
                134,
                132,
                32,
                2,
                173,
                32,
                3,
                173,
                66,
                32,
                134,
                132,
                127,
                34,
                4,
                66,
                32,
                135,
                167,
                36,
                0,
                32,
                4,
                167,
                11,
                // 3, "div_u"
                36,
                1,
                1,
                126,
                32,
                0,
                173,
                32,
                1,
                173,
                66,
                32,
                134,
                132,
                32,
                2,
                173,
                32,
                3,
                173,
                66,
                32,
                134,
                132,
                128,
                34,
                4,
                66,
                32,
                135,
                167,
                36,
                0,
                32,
                4,
                167,
                11,
                // 4, "rem_s"
                36,
                1,
                1,
                126,
                32,
                0,
                173,
                32,
                1,
                173,
                66,
                32,
                134,
                132,
                32,
                2,
                173,
                32,
                3,
                173,
                66,
                32,
                134,
                132,
                129,
                34,
                4,
                66,
                32,
                135,
                167,
                36,
                0,
                32,
                4,
                167,
                11,
                // 5, "rem_u"
                36,
                1,
                1,
                126,
                32,
                0,
                173,
                32,
                1,
                173,
                66,
                32,
                134,
                132,
                32,
                2,
                173,
                32,
                3,
                173,
                66,
                32,
                134,
                132,
                130,
                34,
                4,
                66,
                32,
                135,
                167,
                36,
                0,
                32,
                4,
                167,
                11
              ])
            ),
            {}
          ).exports;
        } catch {
        }
        function s(_, d, I) {
          this.low = _ | 0, this.high = d | 0, this.unsigned = !!I;
        }
        s.prototype.__isLong__, Object.defineProperty(s.prototype, "__isLong__", {
          value: !0
        });
        function a(_) {
          return (_ && _.__isLong__) === !0;
        }
        function i(_) {
          var d = Math.clz32(_ & -_);
          return _ ? 31 - d : d;
        }
        s.isLong = a;
        var f = {}, n = {};
        function u(_, d) {
          var I, D, V;
          return d ? (_ >>>= 0, (V = 0 <= _ && _ < 256) && (D = n[_], D) ? D : (I = c(_, 0, !0), V && (n[_] = I), I)) : (_ |= 0, (V = -128 <= _ && _ < 128) && (D = f[_], D) ? D : (I = c(_, _ < 0 ? -1 : 0, !1), V && (f[_] = I), I));
        }
        s.fromInt = u;
        function l(_, d) {
          if (isNaN(_)) return d ? E : b;
          if (d) {
            if (_ < 0) return E;
            if (_ >= m) return B;
          } else {
            if (_ <= -w) return $;
            if (_ + 1 >= w) return L;
          }
          return _ < 0 ? l(-_, d).neg() : c(
            _ % N | 0,
            _ / N | 0,
            d
          );
        }
        s.fromNumber = l;
        function c(_, d, I) {
          return new s(_, d, I);
        }
        s.fromBits = c;
        var h = Math.pow;
        function p(_, d, I) {
          if (_.length === 0) throw Error("empty string");
          if (typeof d == "number" ? (I = d, d = !1) : d = !!d, _ === "NaN" || _ === "Infinity" || _ === "+Infinity" || _ === "-Infinity")
            return d ? E : b;
          if (I = I || 10, I < 2 || 36 < I) throw RangeError("radix");
          var D;
          if ((D = _.indexOf("-")) > 0) throw Error("interior hyphen");
          if (D === 0)
            return p(_.substring(1), d, I).neg();
          for (var V = l(h(I, 8)), C = b, G = 0; G < _.length; G += 8) {
            var Z = Math.min(8, _.length - G), ie = parseInt(_.substring(G, G + Z), I);
            if (Z < 8) {
              var J = l(h(I, Z));
              C = C.mul(J).add(l(ie));
            } else
              C = C.mul(V), C = C.add(l(ie));
          }
          return C.unsigned = d, C;
        }
        s.fromString = p;
        function g(_, d) {
          return typeof _ == "number" ? l(_, d) : typeof _ == "string" ? p(_, d) : c(
            _.low,
            _.high,
            typeof d == "boolean" ? d : _.unsigned
          );
        }
        s.fromValue = g;
        var v = 65536, T = 1 << 24, N = v * v, m = N * N, w = m / 2, y = u(T), b = u(0);
        s.ZERO = b;
        var E = u(0, !0);
        s.UZERO = E;
        var R = u(1);
        s.ONE = R;
        var U = u(1, !0);
        s.UONE = U;
        var A = u(-1);
        s.NEG_ONE = A;
        var L = c(-1, 2147483647, !1);
        s.MAX_VALUE = L;
        var B = c(-1, -1, !0);
        s.MAX_UNSIGNED_VALUE = B;
        var $ = c(0, -2147483648, !1);
        s.MIN_VALUE = $;
        var O = s.prototype;
        O.toInt = function() {
          return this.unsigned ? this.low >>> 0 : this.low;
        }, O.toNumber = function() {
          return this.unsigned ? (this.high >>> 0) * N + (this.low >>> 0) : this.high * N + (this.low >>> 0);
        }, O.toString = function(d) {
          if (d = d || 10, d < 2 || 36 < d) throw RangeError("radix");
          if (this.isZero()) return "0";
          if (this.isNegative())
            if (this.eq($)) {
              var I = l(d), D = this.div(I), V = D.mul(I).sub(this);
              return D.toString(d) + V.toInt().toString(d);
            } else return "-" + this.neg().toString(d);
          for (var C = l(h(d, 6), this.unsigned), G = this, Z = ""; ; ) {
            var ie = G.div(C), J = G.sub(ie.mul(C)).toInt() >>> 0, W = J.toString(d);
            if (G = ie, G.isZero()) return W + Z;
            for (; W.length < 6; ) W = "0" + W;
            Z = "" + W + Z;
          }
        }, O.getHighBits = function() {
          return this.high;
        }, O.getHighBitsUnsigned = function() {
          return this.high >>> 0;
        }, O.getLowBits = function() {
          return this.low;
        }, O.getLowBitsUnsigned = function() {
          return this.low >>> 0;
        }, O.getNumBitsAbs = function() {
          if (this.isNegative())
            return this.eq($) ? 64 : this.neg().getNumBitsAbs();
          for (var d = this.high != 0 ? this.high : this.low, I = 31; I > 0 && !(d & 1 << I); I--) ;
          return this.high != 0 ? I + 33 : I + 1;
        }, O.isSafeInteger = function() {
          var d = this.high >> 21;
          return d ? this.unsigned ? !1 : d === -1 && !(this.low === 0 && this.high === -2097152) : !0;
        }, O.isZero = function() {
          return this.high === 0 && this.low === 0;
        }, O.eqz = O.isZero, O.isNegative = function() {
          return !this.unsigned && this.high < 0;
        }, O.isPositive = function() {
          return this.unsigned || this.high >= 0;
        }, O.isOdd = function() {
          return (this.low & 1) === 1;
        }, O.isEven = function() {
          return (this.low & 1) === 0;
        }, O.equals = function(d) {
          return a(d) || (d = g(d)), this.unsigned !== d.unsigned && this.high >>> 31 === 1 && d.high >>> 31 === 1 ? !1 : this.high === d.high && this.low === d.low;
        }, O.eq = O.equals, O.notEquals = function(d) {
          return !this.eq(
            /* validates */
            d
          );
        }, O.neq = O.notEquals, O.ne = O.notEquals, O.lessThan = function(d) {
          return this.comp(
            /* validates */
            d
          ) < 0;
        }, O.lt = O.lessThan, O.lessThanOrEqual = function(d) {
          return this.comp(
            /* validates */
            d
          ) <= 0;
        }, O.lte = O.lessThanOrEqual, O.le = O.lessThanOrEqual, O.greaterThan = function(d) {
          return this.comp(
            /* validates */
            d
          ) > 0;
        }, O.gt = O.greaterThan, O.greaterThanOrEqual = function(d) {
          return this.comp(
            /* validates */
            d
          ) >= 0;
        }, O.gte = O.greaterThanOrEqual, O.ge = O.greaterThanOrEqual, O.compare = function(d) {
          if (a(d) || (d = g(d)), this.eq(d)) return 0;
          var I = this.isNegative(), D = d.isNegative();
          return I && !D ? -1 : !I && D ? 1 : this.unsigned ? d.high >>> 0 > this.high >>> 0 || d.high === this.high && d.low >>> 0 > this.low >>> 0 ? -1 : 1 : this.sub(d).isNegative() ? -1 : 1;
        }, O.comp = O.compare, O.negate = function() {
          return !this.unsigned && this.eq($) ? $ : this.not().add(R);
        }, O.neg = O.negate, O.add = function(d) {
          a(d) || (d = g(d));
          var I = this.high >>> 16, D = this.high & 65535, V = this.low >>> 16, C = this.low & 65535, G = d.high >>> 16, Z = d.high & 65535, ie = d.low >>> 16, J = d.low & 65535, W = 0, re = 0, X = 0, Y = 0;
          return Y += C + J, X += Y >>> 16, Y &= 65535, X += V + ie, re += X >>> 16, X &= 65535, re += D + Z, W += re >>> 16, re &= 65535, W += I + G, W &= 65535, c(X << 16 | Y, W << 16 | re, this.unsigned);
        }, O.subtract = function(d) {
          return a(d) || (d = g(d)), this.add(d.neg());
        }, O.sub = O.subtract, O.multiply = function(d) {
          if (this.isZero()) return this;
          if (a(d) || (d = g(d)), o) {
            var I = o.mul(
              this.low,
              this.high,
              d.low,
              d.high
            );
            return c(I, o.get_high(), this.unsigned);
          }
          if (d.isZero()) return this.unsigned ? E : b;
          if (this.eq($)) return d.isOdd() ? $ : b;
          if (d.eq($)) return this.isOdd() ? $ : b;
          if (this.isNegative())
            return d.isNegative() ? this.neg().mul(d.neg()) : this.neg().mul(d).neg();
          if (d.isNegative())
            return this.mul(d.neg()).neg();
          if (this.lt(y) && d.lt(y))
            return l(
              this.toNumber() * d.toNumber(),
              this.unsigned
            );
          var D = this.high >>> 16, V = this.high & 65535, C = this.low >>> 16, G = this.low & 65535, Z = d.high >>> 16, ie = d.high & 65535, J = d.low >>> 16, W = d.low & 65535, re = 0, X = 0, Y = 0, Ae = 0;
          return Ae += G * W, Y += Ae >>> 16, Ae &= 65535, Y += C * W, X += Y >>> 16, Y &= 65535, Y += G * J, X += Y >>> 16, Y &= 65535, X += V * W, re += X >>> 16, X &= 65535, X += C * J, re += X >>> 16, X &= 65535, X += G * ie, re += X >>> 16, X &= 65535, re += D * W + V * J + C * ie + G * Z, re &= 65535, c(Y << 16 | Ae, re << 16 | X, this.unsigned);
        }, O.mul = O.multiply, O.divide = function(d) {
          if (a(d) || (d = g(d)), d.isZero()) throw Error("division by zero");
          if (o) {
            if (!this.unsigned && this.high === -2147483648 && d.low === -1 && d.high === -1)
              return this;
            var I = (this.unsigned ? o.div_u : o.div_s)(
              this.low,
              this.high,
              d.low,
              d.high
            );
            return c(I, o.get_high(), this.unsigned);
          }
          if (this.isZero()) return this.unsigned ? E : b;
          var D, V, C;
          if (this.unsigned) {
            if (d.unsigned || (d = d.toUnsigned()), d.gt(this)) return E;
            if (d.gt(this.shru(1)))
              return U;
            C = E;
          } else {
            if (this.eq($)) {
              if (d.eq(R) || d.eq(A))
                return $;
              if (d.eq($)) return R;
              var G = this.shr(1);
              return D = G.div(d).shl(1), D.eq(b) ? d.isNegative() ? R : A : (V = this.sub(d.mul(D)), C = D.add(V.div(d)), C);
            } else if (d.eq($)) return this.unsigned ? E : b;
            if (this.isNegative())
              return d.isNegative() ? this.neg().div(d.neg()) : this.neg().div(d).neg();
            if (d.isNegative()) return this.div(d.neg()).neg();
            C = b;
          }
          for (V = this; V.gte(d); ) {
            D = Math.max(1, Math.floor(V.toNumber() / d.toNumber()));
            for (var Z = Math.ceil(Math.log(D) / Math.LN2), ie = Z <= 48 ? 1 : h(2, Z - 48), J = l(D), W = J.mul(d); W.isNegative() || W.gt(V); )
              D -= ie, J = l(D, this.unsigned), W = J.mul(d);
            J.isZero() && (J = R), C = C.add(J), V = V.sub(W);
          }
          return C;
        }, O.div = O.divide, O.modulo = function(d) {
          if (a(d) || (d = g(d)), o) {
            var I = (this.unsigned ? o.rem_u : o.rem_s)(
              this.low,
              this.high,
              d.low,
              d.high
            );
            return c(I, o.get_high(), this.unsigned);
          }
          return this.sub(this.div(d).mul(d));
        }, O.mod = O.modulo, O.rem = O.modulo, O.not = function() {
          return c(~this.low, ~this.high, this.unsigned);
        }, O.countLeadingZeros = function() {
          return this.high ? Math.clz32(this.high) : Math.clz32(this.low) + 32;
        }, O.clz = O.countLeadingZeros, O.countTrailingZeros = function() {
          return this.low ? i(this.low) : i(this.high) + 32;
        }, O.ctz = O.countTrailingZeros, O.and = function(d) {
          return a(d) || (d = g(d)), c(
            this.low & d.low,
            this.high & d.high,
            this.unsigned
          );
        }, O.or = function(d) {
          return a(d) || (d = g(d)), c(
            this.low | d.low,
            this.high | d.high,
            this.unsigned
          );
        }, O.xor = function(d) {
          return a(d) || (d = g(d)), c(
            this.low ^ d.low,
            this.high ^ d.high,
            this.unsigned
          );
        }, O.shiftLeft = function(d) {
          return a(d) && (d = d.toInt()), (d &= 63) === 0 ? this : d < 32 ? c(
            this.low << d,
            this.high << d | this.low >>> 32 - d,
            this.unsigned
          ) : c(0, this.low << d - 32, this.unsigned);
        }, O.shl = O.shiftLeft, O.shiftRight = function(d) {
          return a(d) && (d = d.toInt()), (d &= 63) === 0 ? this : d < 32 ? c(
            this.low >>> d | this.high << 32 - d,
            this.high >> d,
            this.unsigned
          ) : c(
            this.high >> d - 32,
            this.high >= 0 ? 0 : -1,
            this.unsigned
          );
        }, O.shr = O.shiftRight, O.shiftRightUnsigned = function(d) {
          return a(d) && (d = d.toInt()), (d &= 63) === 0 ? this : d < 32 ? c(
            this.low >>> d | this.high << 32 - d,
            this.high >>> d,
            this.unsigned
          ) : d === 32 ? c(this.high, 0, this.unsigned) : c(this.high >>> d - 32, 0, this.unsigned);
        }, O.shru = O.shiftRightUnsigned, O.shr_u = O.shiftRightUnsigned, O.rotateLeft = function(d) {
          var I;
          return a(d) && (d = d.toInt()), (d &= 63) === 0 ? this : d === 32 ? c(this.high, this.low, this.unsigned) : d < 32 ? (I = 32 - d, c(
            this.low << d | this.high >>> I,
            this.high << d | this.low >>> I,
            this.unsigned
          )) : (d -= 32, I = 32 - d, c(
            this.high << d | this.low >>> I,
            this.low << d | this.high >>> I,
            this.unsigned
          ));
        }, O.rotl = O.rotateLeft, O.rotateRight = function(d) {
          var I;
          return a(d) && (d = d.toInt()), (d &= 63) === 0 ? this : d === 32 ? c(this.high, this.low, this.unsigned) : d < 32 ? (I = 32 - d, c(
            this.high << I | this.low >>> d,
            this.low << I | this.high >>> d,
            this.unsigned
          )) : (d -= 32, I = 32 - d, c(
            this.low << I | this.high >>> d,
            this.high << I | this.low >>> d,
            this.unsigned
          ));
        }, O.rotr = O.rotateRight, O.toSigned = function() {
          return this.unsigned ? c(this.low, this.high, !1) : this;
        }, O.toUnsigned = function() {
          return this.unsigned ? this : c(this.low, this.high, !0);
        }, O.toBytes = function(d) {
          return d ? this.toBytesLE() : this.toBytesBE();
        }, O.toBytesLE = function() {
          var d = this.high, I = this.low;
          return [
            I & 255,
            I >>> 8 & 255,
            I >>> 16 & 255,
            I >>> 24,
            d & 255,
            d >>> 8 & 255,
            d >>> 16 & 255,
            d >>> 24
          ];
        }, O.toBytesBE = function() {
          var d = this.high, I = this.low;
          return [
            d >>> 24,
            d >>> 16 & 255,
            d >>> 8 & 255,
            d & 255,
            I >>> 24,
            I >>> 16 & 255,
            I >>> 8 & 255,
            I & 255
          ];
        }, s.fromBytes = function(d, I, D) {
          return D ? s.fromBytesLE(d, I) : s.fromBytesBE(d, I);
        }, s.fromBytesLE = function(d, I) {
          return new s(
            d[0] | d[1] << 8 | d[2] << 16 | d[3] << 24,
            d[4] | d[5] << 8 | d[6] << 16 | d[7] << 24,
            I
          );
        }, s.fromBytesBE = function(d, I) {
          return new s(
            d[4] << 24 | d[5] << 16 | d[6] << 8 | d[7],
            d[0] << 24 | d[1] << 16 | d[2] << 8 | d[3],
            I
          );
        }, typeof BigInt == "function" && (s.fromBigInt = function(d, I) {
          var D = Number(BigInt.asIntN(32, d)), V = Number(BigInt.asIntN(32, d >> BigInt(32)));
          return c(D, V, I);
        }, s.fromValue = function(d, I) {
          return typeof d == "bigint" ? s.fromBigInt(d, I) : g(d, I);
        }, O.toBigInt = function() {
          var d = BigInt(this.low >>> 0), I = BigInt(this.unsigned ? this.high >>> 0 : this.high);
          return I << BigInt(32) | d;
        }), r.default = s;
      }
    );
  }(He, He.exports)), He.exports;
}
var fr;
function me() {
  return fr || (fr = 1, function(t) {
    var e = t;
    e.asPromise = Kr, e.base64 = Yr, e.EventEmitter = qn, e.float = zn, e.utf8 = en, e.pool = jn, e.LongBits = Jn();
    function r(a) {
      return a === "__proto__" || a === "prototype" || a === "constructor";
    }
    e.isUnsafeProperty = r, e.isNode = !!(typeof be < "u" && be && be.process && be.process.versions && be.process.versions.node), e.global = e.isNode && be || typeof window < "u" && window || typeof self < "u" && self || be, e.emptyArray = Object.freeze ? Object.freeze([]) : (
      /* istanbul ignore next */
      []
    ), e.emptyObject = Object.freeze ? Object.freeze({}) : (
      /* istanbul ignore next */
      {}
    ), e.isInteger = Number.isInteger || /* istanbul ignore next */
    function(i) {
      return typeof i == "number" && isFinite(i) && Math.floor(i) === i;
    }, e.isString = function(i) {
      return typeof i == "string" || i instanceof String;
    }, e.isObject = function(i) {
      return i && typeof i == "object";
    }, e.isset = /**
     * Checks if a property on a message is considered to be present.
     * @param {Object} obj Plain object or message instance
     * @param {string} prop Property name
     * @returns {boolean} `true` if considered to be present, otherwise `false`
     */
    e.isSet = function(i, f) {
      var n = i[f];
      return n != null && Object.hasOwnProperty.call(i, f) ? typeof n != "object" || (Array.isArray(n) ? n.length : Object.keys(n).length) > 0 : !1;
    }, e.Buffer = function() {
      try {
        var a = e.global.Buffer;
        return a.prototype.utf8Write ? a : (
          /* istanbul ignore next */
          null
        );
      } catch {
        return null;
      }
    }(), e._Buffer_from = null, e._Buffer_allocUnsafe = null, e.newBuffer = function(i) {
      return typeof i == "number" ? e.Buffer ? e._Buffer_allocUnsafe(i) : new e.Array(i) : e.Buffer ? e._Buffer_from(i) : typeof Uint8Array > "u" ? i : new Uint8Array(i);
    }, e.Array = typeof Uint8Array < "u" ? Uint8Array : Array, e.Long = /* istanbul ignore next */
    e.global.dcodeIO && /* istanbul ignore next */
    e.global.dcodeIO.Long || /* istanbul ignore next */
    e.global.Long || function() {
      try {
        var a = Gn();
        return a && a.isLong ? a : null;
      } catch {
        return null;
      }
    }(), e.key2Re = /^true|false|0|1$/, e.key32Re = /^-?(?:0|[1-9][0-9]*)$/, e.key64Re = /^(?:[\\x00-\\xff]{8}|-?(?:0|[1-9][0-9]*))$/, e.longToHash = function(i) {
      return i ? e.LongBits.from(i).toHash() : e.LongBits.zeroHash;
    }, e.longFromHash = function(i, f) {
      var n = e.LongBits.fromHash(i);
      return e.Long ? e.Long.fromBits(n.lo, n.hi, f) : n.toNumber(!!f);
    };
    function o(a) {
      var i = typeof arguments[arguments.length - 1] == "boolean", f = i ? arguments.length - 1 : arguments.length;
      i = i && arguments[arguments.length - 1];
      for (var n = 1; n < f; ++n) {
        var u = arguments[n];
        if (u)
          for (var l = Object.keys(u), c = 0; c < l.length; ++c)
            !r(l[c]) && (a[l[c]] === void 0 || !i) && (a[l[c]] = u[l[c]]);
      }
      return a;
    }
    e.merge = o, e.nestingLimit = 32, e.recursionLimit = 100, e.makeProp = function(i, f) {
      Object.defineProperty(i, f, {
        enumerable: !0,
        configurable: !0,
        writable: !0
      });
    }, e.lcFirst = function(i) {
      return i.charAt(0).toLowerCase() + i.substring(1);
    };
    function s(a) {
      function i(f, n) {
        if (!(this instanceof i))
          return new i(f, n);
        Object.defineProperty(this, "message", { get: function() {
          return f;
        } }), Error.captureStackTrace ? Error.captureStackTrace(this, i) : Object.defineProperty(this, "stack", { value: new Error().stack || "" }), n && o(this, n);
      }
      return i.prototype = Object.create(Error.prototype, {
        constructor: {
          value: i,
          writable: !0,
          enumerable: !1,
          configurable: !0
        },
        name: {
          get: function() {
            return a;
          },
          set: void 0,
          enumerable: !1,
          // configurable: false would accurately preserve the behavior of
          // the original, but I'm guessing that was not intentional.
          // For an actual error subclass, this property would
          // be configurable.
          configurable: !0
        },
        toString: {
          value: function() {
            return this.name + ": " + this.message;
          },
          writable: !0,
          enumerable: !1,
          configurable: !0
        }
      }), i;
    }
    e.newError = s, e.ProtocolError = s("ProtocolError"), e.oneOfGetter = function(i) {
      for (var f = {}, n = 0; n < i.length; ++n)
        f[i[n]] = 1;
      return function() {
        for (var u = Object.keys(this), l = u.length - 1; l > -1; --l)
          if (f[u[l]] === 1 && this[u[l]] !== void 0 && this[u[l]] !== null)
            return u[l];
      };
    }, e.oneOfSetter = function(i) {
      return function(f) {
        for (var n = 0; n < i.length; ++n)
          i[n] !== f && delete this[i[n]];
      };
    }, e.toJSONOptions = {
      longs: String,
      enums: String,
      bytes: String,
      json: !0
    }, e._configure = function() {
      var a = e.Buffer;
      if (!a) {
        e._Buffer_from = e._Buffer_allocUnsafe = null;
        return;
      }
      e._Buffer_from = a.from !== Uint8Array.from && a.from || /* istanbul ignore next */
      function(f, n) {
        return new a(f, n);
      }, e._Buffer_allocUnsafe = a.allocUnsafe || /* istanbul ignore next */
      function(f) {
        return new a(f);
      };
    };
  }(nt)), nt;
}
var Bt = j, oe = me(), Tt, Ke = oe.LongBits, lr = oe.base64, cr = oe.utf8;
function Je(t, e, r) {
  this.fn = t, this.len = e, this.next = void 0, this.val = r;
}
function Dt() {
}
function Wn(t) {
  this.head = t.head, this.tail = t.tail, this.len = t.len, this.next = t.states;
}
function j() {
  this.len = 0, this.head = new Je(Dt, 0, 0), this.tail = this.head, this.states = null;
}
var tn = function() {
  return oe.Buffer ? function() {
    return (j.create = function() {
      return new Tt();
    })();
  } : function() {
    return new j();
  };
};
j.create = tn();
j.alloc = function(e) {
  return new oe.Array(e);
};
oe.Array !== Array && (j.alloc = oe.pool(j.alloc, oe.Array.prototype.subarray));
j.prototype._push = function(e, r, o) {
  return this.tail = this.tail.next = new Je(e, r, o), this.len += r, this;
};
function $t(t, e, r) {
  e[r] = t & 255;
}
function Hn(t, e, r) {
  for (; t > 127; )
    e[r++] = t & 127 | 128, t >>>= 7;
  e[r] = t;
}
function Mt(t, e) {
  this.len = t, this.next = void 0, this.val = e;
}
Mt.prototype = Object.create(Je.prototype);
Mt.prototype.fn = Hn;
j.prototype.uint32 = function(e) {
  return this.len += (this.tail = this.tail.next = new Mt(
    (e = e >>> 0) < 128 ? 1 : e < 16384 ? 2 : e < 2097152 ? 3 : e < 268435456 ? 4 : 5,
    e
  )).len, this;
};
j.prototype.int32 = function(e) {
  return (e |= 0) < 0 ? this._push(Ct, 10, Ke.fromNumber(e)) : this.uint32(e);
};
j.prototype.sint32 = function(e) {
  return this.uint32((e << 1 ^ e >> 31) >>> 0);
};
function Ct(t, e, r) {
  for (var o = t.lo, s = t.hi; s; )
    e[r++] = o & 127 | 128, o = (o >>> 7 | s << 25) >>> 0, s >>>= 7;
  for (; o > 127; )
    e[r++] = o & 127 | 128, o = o >>> 7;
  e[r++] = o;
}
j.prototype.uint64 = function(e) {
  var r = Ke.from(e);
  return this._push(Ct, r.length(), r);
};
j.prototype.int64 = j.prototype.uint64;
j.prototype.sint64 = function(e) {
  var r = Ke.from(e).zzEncode();
  return this._push(Ct, r.length(), r);
};
j.prototype.bool = function(e) {
  return this._push($t, 1, e ? 1 : 0);
};
function Ft(t, e, r) {
  e[r] = t & 255, e[r + 1] = t >>> 8 & 255, e[r + 2] = t >>> 16 & 255, e[r + 3] = t >>> 24;
}
j.prototype.fixed32 = function(e) {
  return this._push(Ft, 4, e >>> 0);
};
j.prototype.sfixed32 = j.prototype.fixed32;
j.prototype.fixed64 = function(e) {
  var r = Ke.from(e);
  return this._push(Ft, 4, r.lo)._push(Ft, 4, r.hi);
};
j.prototype.sfixed64 = j.prototype.fixed64;
j.prototype.float = function(e) {
  return this._push(oe.float.writeFloatLE, 4, e);
};
j.prototype.double = function(e) {
  return this._push(oe.float.writeDoubleLE, 8, e);
};
var Zn = oe.Array.prototype.set ? function(e, r, o) {
  r.set(e, o);
} : function(e, r, o) {
  for (var s = 0; s < e.length; ++s)
    r[o + s] = e[s];
};
j.prototype.bytes = function(e) {
  var r = e.length >>> 0;
  if (!r)
    return this._push($t, 1, 0);
  if (oe.isString(e)) {
    var o = j.alloc(r = lr.length(e));
    lr.decode(e, o, 0), e = o;
  }
  return this.uint32(r)._push(Zn, r, e);
};
j.prototype.string = function(e) {
  var r = cr.length(e);
  return r ? this.uint32(r)._push(cr.write, r, e) : this._push($t, 1, 0);
};
j.prototype.fork = function() {
  return this.states = new Wn(this), this.head = this.tail = new Je(Dt, 0, 0), this.len = 0, this;
};
j.prototype.reset = function() {
  return this.states ? (this.head = this.states.head, this.tail = this.states.tail, this.len = this.states.len, this.states = this.states.next) : (this.head = this.tail = new Je(Dt, 0, 0), this.len = 0), this;
};
j.prototype.ldelim = function() {
  var e = this.head, r = this.tail, o = this.len;
  return this.reset().uint32(o), o && (this.tail.next = e.next, this.tail = r, this.len += o), this;
};
j.prototype.finish = function() {
  for (var e = this.head.next, r = this.constructor.alloc(this.len), o = 0; e; )
    e.fn(e.val, r, o), o += e.len, e = e.next;
  return r;
};
j._configure = function(t) {
  Tt = t, j.create = tn(), Tt._configure();
};
var Xn = de, rn = Bt;
(de.prototype = Object.create(rn.prototype)).constructor = de;
var Oe = me();
function de() {
  rn.call(this);
}
de._configure = function() {
  de.alloc = Oe._Buffer_allocUnsafe, de.writeBytesBuffer = Oe.Buffer && Oe.Buffer.prototype instanceof Uint8Array && Oe.Buffer.prototype.set.name === "set" ? function(e, r, o) {
    r.set(e, o);
  } : function(e, r, o) {
    if (e.copy)
      e.copy(r, o, 0, e.length);
    else for (var s = 0; s < e.length; )
      r[o++] = e[s++];
  };
};
de.prototype.bytes = function(e) {
  Oe.isString(e) && (e = Oe._Buffer_from(e, "base64"));
  var r = e.length >>> 0;
  return this.uint32(r), r && this._push(de.writeBytesBuffer, r, e), this;
};
function Qn(t, e, r) {
  t.length < 40 ? Oe.utf8.write(t, e, r) : e.utf8Write ? e.utf8Write(t, r) : e.write(t, r);
}
de.prototype.string = function(e) {
  var r = Oe.Buffer.byteLength(e);
  return this.uint32(r), r && this._push(Qn, r, e), this;
};
de._configure();
var Ut = H, ae = me(), It, nn = ae.LongBits, Kn = ae.utf8;
function ue(t, e) {
  return RangeError("index out of range: " + t.pos + " + " + (e || 1) + " > " + t.len);
}
function H(t) {
  this.buf = t, this.pos = 0, this.len = t.length;
}
var hr = typeof Uint8Array < "u" ? function(e) {
  if (e instanceof Uint8Array || Array.isArray(e))
    return new H(e);
  throw Error("illegal buffer");
} : function(e) {
  if (Array.isArray(e))
    return new H(e);
  throw Error("illegal buffer");
}, sn = function() {
  return ae.Buffer ? function(r) {
    return (H.create = function(s) {
      return ae.Buffer.isBuffer(s) ? new It(s) : hr(s);
    })(r);
  } : hr;
};
H.create = sn();
H.prototype._slice = ae.Array.prototype.subarray || /* istanbul ignore next */
ae.Array.prototype.slice;
H.prototype.uint32 = /* @__PURE__ */ function() {
  var e = 4294967295;
  return function() {
    if (e = (this.buf[this.pos] & 127) >>> 0, this.buf[this.pos++] < 128 || (e = (e | (this.buf[this.pos] & 127) << 7) >>> 0, this.buf[this.pos++] < 128) || (e = (e | (this.buf[this.pos] & 127) << 14) >>> 0, this.buf[this.pos++] < 128) || (e = (e | (this.buf[this.pos] & 127) << 21) >>> 0, this.buf[this.pos++] < 128) || (e = (e | (this.buf[this.pos] & 15) << 28) >>> 0, this.buf[this.pos++] < 128)) return e;
    if ((this.pos += 5) > this.len)
      throw this.pos = this.len, ue(this, 10);
    return e;
  };
}();
H.prototype.int32 = function() {
  return this.uint32() | 0;
};
H.prototype.sint32 = function() {
  var e = this.uint32();
  return e >>> 1 ^ -(e & 1) | 0;
};
function st() {
  var t = new nn(0, 0), e = 0;
  if (this.len - this.pos > 4) {
    for (; e < 4; ++e)
      if (t.lo = (t.lo | (this.buf[this.pos] & 127) << e * 7) >>> 0, this.buf[this.pos++] < 128)
        return t;
    if (t.lo = (t.lo | (this.buf[this.pos] & 127) << 28) >>> 0, t.hi = (t.hi | (this.buf[this.pos] & 127) >> 4) >>> 0, this.buf[this.pos++] < 128)
      return t;
    e = 0;
  } else {
    for (; e < 3; ++e) {
      if (this.pos >= this.len)
        throw ue(this);
      if (t.lo = (t.lo | (this.buf[this.pos] & 127) << e * 7) >>> 0, this.buf[this.pos++] < 128)
        return t;
    }
    return t.lo = (t.lo | (this.buf[this.pos++] & 127) << e * 7) >>> 0, t;
  }
  if (this.len - this.pos > 4) {
    for (; e < 5; ++e)
      if (t.hi = (t.hi | (this.buf[this.pos] & 127) << e * 7 + 3) >>> 0, this.buf[this.pos++] < 128)
        return t;
  } else
    for (; e < 5; ++e) {
      if (this.pos >= this.len)
        throw ue(this);
      if (t.hi = (t.hi | (this.buf[this.pos] & 127) << e * 7 + 3) >>> 0, this.buf[this.pos++] < 128)
        return t;
    }
  throw Error("invalid varint encoding");
}
H.prototype.bool = function() {
  return this.uint32() !== 0;
};
function Xe(t, e) {
  return (t[e - 4] | t[e - 3] << 8 | t[e - 2] << 16 | t[e - 1] << 24) >>> 0;
}
H.prototype.fixed32 = function() {
  if (this.pos + 4 > this.len)
    throw ue(this, 4);
  return Xe(this.buf, this.pos += 4);
};
H.prototype.sfixed32 = function() {
  if (this.pos + 4 > this.len)
    throw ue(this, 4);
  return Xe(this.buf, this.pos += 4) | 0;
};
function dr() {
  if (this.pos + 8 > this.len)
    throw ue(this, 8);
  return new nn(Xe(this.buf, this.pos += 4), Xe(this.buf, this.pos += 4));
}
H.prototype.float = function() {
  if (this.pos + 4 > this.len)
    throw ue(this, 4);
  var e = ae.float.readFloatLE(this.buf, this.pos);
  return this.pos += 4, e;
};
H.prototype.double = function() {
  if (this.pos + 8 > this.len)
    throw ue(this, 4);
  var e = ae.float.readDoubleLE(this.buf, this.pos);
  return this.pos += 8, e;
};
H.prototype.bytes = function() {
  var e = this.uint32(), r = this.pos, o = this.pos + e;
  if (o > this.len)
    throw ue(this, e);
  if (this.pos += e, Array.isArray(this.buf))
    return this.buf.slice(r, o);
  if (r === o) {
    var s = ae.Buffer;
    return s ? s.alloc(0) : new this.buf.constructor(0);
  }
  return this._slice.call(this.buf, r, o);
};
H.prototype.string = function() {
  var e = this.bytes();
  return Kn.read(e, 0, e.length);
};
H.prototype.skip = function(e) {
  if (typeof e == "number") {
    if (this.pos + e > this.len)
      throw ue(this, e);
    this.pos += e;
  } else
    do
      if (this.pos >= this.len)
        throw ue(this);
    while (this.buf[this.pos++] & 128);
  return this;
};
H.recursionLimit = ae.recursionLimit;
H.prototype.skipType = function(t, e) {
  if (e === void 0 && (e = 0), e > H.recursionLimit)
    throw Error("maximum nesting depth exceeded");
  switch (t) {
    case 0:
      this.skip();
      break;
    case 1:
      this.skip(8);
      break;
    case 2:
      this.skip(this.uint32());
      break;
    case 3:
      for (; (t = this.uint32() & 7) !== 4; )
        this.skipType(t, e + 1);
      break;
    case 5:
      this.skip(4);
      break;
    default:
      throw Error("invalid wire type " + t + " at offset " + this.pos);
  }
  return this;
};
H._configure = function(t) {
  It = t, H.create = sn(), It._configure();
  var e = ae.Long ? "toLong" : (
    /* istanbul ignore next */
    "toNumber"
  );
  ae.merge(H.prototype, {
    int64: function() {
      return st.call(this)[e](!1);
    },
    uint64: function() {
      return st.call(this)[e](!0);
    },
    sint64: function() {
      return st.call(this).zzDecode()[e](!1);
    },
    fixed64: function() {
      return dr.call(this)[e](!0);
    },
    sfixed64: function() {
      return dr.call(this)[e](!1);
    }
  });
};
var Yn = ke, on = Ut;
(ke.prototype = Object.create(on.prototype)).constructor = ke;
var pr = me();
function ke(t) {
  on.call(this, t);
}
ke._configure = function() {
  pr.Buffer && (ke.prototype._slice = pr.Buffer.prototype.slice);
};
ke.prototype.string = function() {
  var e = this.uint32();
  return this.buf.utf8Slice ? this.buf.utf8Slice(this.pos, this.pos = Math.min(this.pos + e, this.len)) : this.buf.toString("utf-8", this.pos, this.pos = Math.min(this.pos + e, this.len));
};
ke._configure();
var qt = {}, ei = je, zt = me();
(je.prototype = Object.create(zt.EventEmitter.prototype)).constructor = je;
function je(t, e, r) {
  if (typeof t != "function")
    throw TypeError("rpcImpl must be a function");
  zt.EventEmitter.call(this), this.rpcImpl = t, this.requestDelimited = !!e, this.responseDelimited = !!r;
}
je.prototype.rpcCall = function t(e, r, o, s, a) {
  if (!s)
    throw TypeError("request must be specified");
  var i = this;
  if (!a)
    return zt.asPromise(t, i, e, r, o, s);
  if (!i.rpcImpl) {
    setTimeout(function() {
      a(Error("already ended"));
    }, 0);
    return;
  }
  try {
    return i.rpcImpl(
      e,
      r[i.requestDelimited ? "encodeDelimited" : "encode"](s).finish(),
      function(n, u) {
        if (n)
          return i.emit("error", n, e), a(n);
        if (u === null) {
          i.end(
            /* endedByRPC */
            !0
          );
          return;
        }
        if (!(u instanceof o))
          try {
            u = o[i.responseDelimited ? "decodeDelimited" : "decode"](u);
          } catch (l) {
            return i.emit("error", l, e), a(l);
          }
        return i.emit("data", u, e), a(null, u);
      }
    );
  } catch (f) {
    i.emit("error", f, e), setTimeout(function() {
      a(f);
    }, 0);
    return;
  }
};
je.prototype.end = function(e) {
  return this.rpcImpl && (e || this.rpcImpl(null, null, null), this.rpcImpl = null, this.emit("end").off()), this;
};
(function(t) {
  var e = t;
  e.Service = ei;
})(qt);
var an = /* @__PURE__ */ Object.create(null);
(function(t) {
  var e = t;
  e.build = "minimal", e.Writer = Bt, e.BufferWriter = Xn, e.Reader = Ut, e.BufferReader = Yn, e.util = me(), e.rpc = qt, e.roots = an, e.configure = r;
  function r() {
    e.util._configure(), e.Writer._configure(e.BufferWriter), e.Reader._configure(e.BufferReader);
  }
  r();
})(Qr);
var ot = {}, at = { exports: {} }, ti = jt, ri = /^(?:do|if|in|for|let|new|try|var|case|else|enum|eval|false|null|this|true|void|with|break|catch|class|const|super|throw|while|yield|delete|export|import|public|return|static|switch|typeof|default|extends|finally|package|private|continue|debugger|function|arguments|interface|protected|implements|instanceof)$/;
function jt(t, e) {
  typeof t == "string" && (e = t, t = void 0);
  var r = [];
  function o(a) {
    if (typeof a != "string") {
      var i = s();
      if (jt.verbose && console.log("codegen: " + i), i = "return " + i, a) {
        for (var f = Object.keys(a), n = new Array(f.length + 1), u = new Array(f.length), l = 0; l < f.length; )
          n[l] = f[l], u[l] = a[f[l++]];
        return n[l] = i, Function.apply(null, n).apply(null, u);
      }
      return Function(i)();
    }
    for (var c = new Array(arguments.length - 1), h = 0; h < c.length; )
      c[h] = arguments[++h];
    if (h = 0, a = a.replace(/%([%dfijs])/g, function(g, v) {
      var T = c[h++];
      switch (v) {
        case "d":
        case "f":
          return String(Number(T));
        case "i":
          return String(Math.floor(T));
        case "j":
          return JSON.stringify(T);
        case "s":
          return String(T);
      }
      return "%";
    }), h !== c.length)
      throw Error("parameter count mismatch");
    return r.push(a), o;
  }
  function s(a) {
    return "function " + ni(a || e) + "(" + (t && t.join(",") || "") + `){
  ` + r.join(`
  `) + `
}`;
  }
  return o.toString = s, o;
}
jt.verbose = !1;
function ni(t) {
  return !t || (t = String(t).replace(/[^\w$]/g, ""), !t) ? "" : (/^\d/.test(t) && (t = "_" + t), ri.test(t) ? t + "_" : t);
}
var ii = {}, si = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  default: ii
}), un = /* @__PURE__ */ Cn(si), xe = null;
try {
  xe = un, (!xe || !xe.readFile || !xe.readFileSync) && (xe = null);
} catch {
}
var oi = xe, ai = Me, ui = Kr, ut = oi;
function Me(t, e, r) {
  return typeof e == "function" ? (r = e, e = {}) : e || (e = {}), r ? !e.xhr && ut && ut.readFile ? ut.readFile(t, function(s, a) {
    return s && typeof XMLHttpRequest < "u" ? Me.xhr(t, e, r) : s ? r(s) : r(null, e.binary ? a : a.toString("utf8"));
  }) : Me.xhr(t, e, r) : ui(Me, this, t, e);
}
Me.xhr = function(e, r, o) {
  var s = new XMLHttpRequest();
  s.onreadystatechange = function() {
    if (s.readyState === 4) {
      if (s.status !== 0 && s.status !== 200)
        return o(Error("status " + s.status));
      if (r.binary) {
        var i = s.response;
        if (!i) {
          i = [];
          for (var f = 0; f < s.responseText.length; ++f)
            i.push(s.responseText.charCodeAt(f) & 255);
        }
        return o(null, typeof Uint8Array < "u" ? new Uint8Array(i) : i);
      }
      return o(null, s.responseText);
    }
  }, r.binary && ("overrideMimeType" in s && s.overrideMimeType("text/plain; charset=x-user-defined"), s.responseType = "arraybuffer"), s.open("GET", e), s.send();
};
var fn = {};
(function(t) {
  var e = t, r = (
    /**
     * Tests if the specified path is absolute.
     * @param {string} path Path to test
     * @returns {boolean} `true` if path is absolute
     */
    e.isAbsolute = function(a) {
      return /^(?:\/|\w+:)/.test(a);
    }
  ), o = (
    /**
     * Normalizes the specified path.
     * @param {string} path Path to normalize
     * @returns {string} Normalized path
     */
    e.normalize = function(a) {
      a = a.replace(/\\/g, "/").replace(/\/{2,}/g, "/");
      var i = a.split("/"), f = r(a), n = "";
      f && (n = i.shift() + "/");
      for (var u = 0; u < i.length; )
        i[u] === ".." ? u > 0 && i[u - 1] !== ".." ? i.splice(--u, 2) : f ? i.splice(u, 1) : ++u : i[u] === "." ? i.splice(u, 1) : ++u;
      return n + i.join("/");
    }
  );
  e.resolve = function(a, i, f) {
    return f || (i = o(i)), r(i) ? i : (f || (a = o(a)), (a = a.replace(/(?:\/|^)[^/]+$/, "")).length ? o(a + "/" + i) : i);
  };
})(fn);
var ln = {};
(function(t) {
  var e = t;
  e.numberRe = /^(?![eE])[0-9]*(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?$/, e.typeRefRe = /^(?:\.?[a-zA-Z_][a-zA-Z_0-9]*)(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*$/, e.reservedRe = /^(?:do|if|in|for|let|new|try|var|case|else|enum|eval|false|null|this|true|void|with|break|catch|class|const|super|throw|while|yield|delete|export|import|public|return|static|switch|typeof|default|extends|finally|package|private|continue|debugger|function|arguments|interface|protected|implements|instanceof)$/;
})(ln);
var Te = null;
try {
  Te = un, (!Te || !Te.readFile || !Te.readFileSync) && (Te = null);
} catch {
}
var fi = Te, ft, mr;
function Ge() {
  if (mr) return ft;
  mr = 1, ft = n;
  var t = Se();
  ((n.prototype = Object.create(t.prototype)).constructor = n).className = "Namespace";
  var e = Ne(), r = te(), o = Le(), s, a, i;
  n.fromJSON = function(c, h, p) {
    return p = r.checkDepth(p), new n(c, h.options).addJSON(h.nested, p);
  };
  function f(l, c) {
    if (l && l.length) {
      for (var h = {}, p = 0; p < l.length; ++p)
        h[l[p].name] = l[p].toJSON(c);
      return h;
    }
  }
  n.arrayToJSON = f, n.isReservedId = function(c, h) {
    if (c) {
      for (var p = 0; p < c.length; ++p)
        if (typeof c[p] != "string" && c[p][0] <= h && c[p][1] > h)
          return !0;
    }
    return !1;
  }, n.isReservedName = function(c, h) {
    if (c) {
      for (var p = 0; p < c.length; ++p)
        if (c[p] === h)
          return !0;
    }
    return !1;
  };
  function n(l, c) {
    t.call(this, l, c), this.nested = void 0, this._nestedArray = null, this._lookupCache = /* @__PURE__ */ Object.create(null), this._needsRecursiveFeatureResolution = !0, this._needsRecursiveResolve = !0;
  }
  function u(l) {
    l._nestedArray = null, l._lookupCache = /* @__PURE__ */ Object.create(null);
    for (var c = l; c = c.parent; )
      c._lookupCache = /* @__PURE__ */ Object.create(null);
    return l;
  }
  return Object.defineProperty(n.prototype, "nestedArray", {
    get: function() {
      return this._nestedArray || (this._nestedArray = r.toArray(this.nested));
    }
  }), n.prototype.toJSON = function(c) {
    return r.toObject([
      "options",
      this.options,
      "nested",
      f(this.nestedArray, c)
    ]);
  }, n.prototype.addJSON = function(c, h) {
    h = r.checkDepth(h);
    var p = this;
    if (c)
      for (var g = Object.keys(c), v = 0, T; v < g.length; ++v)
        T = c[g[v]], p.add(
          // most to least likely
          (T.fields !== void 0 ? s.fromJSON : T.values !== void 0 ? i.fromJSON : T.methods !== void 0 ? a.fromJSON : T.id !== void 0 ? e.fromJSON : n.fromJSON)(g[v], T, h + 1)
        );
    return this;
  }, n.prototype.get = function(c) {
    return this.nested && Object.prototype.hasOwnProperty.call(this.nested, c) ? this.nested[c] : null;
  }, n.prototype.getEnum = function(c) {
    if (this.nested && Object.prototype.hasOwnProperty.call(this.nested, c) && this.nested[c] instanceof i)
      return this.nested[c].values;
    throw Error("no such enum: " + c);
  }, n.prototype.add = function(c) {
    if (!(c instanceof e && c.extend !== void 0 || c instanceof s || c instanceof o || c instanceof i || c instanceof a || c instanceof n))
      throw TypeError("object must be a valid nested object");
    if (c.name === "__proto__")
      return this;
    if (!this.nested)
      this.nested = {};
    else {
      var h = this.get(c.name);
      if (h)
        if (h instanceof n && c instanceof n && !(h instanceof s || h instanceof a)) {
          for (var p = h.nestedArray, g = 0; g < p.length; ++g)
            c.add(p[g]);
          this.remove(h), this.nested || (this.nested = {}), c.setOptions(h.options, !0);
        } else
          throw Error("duplicate name '" + c.name + "' in " + this);
    }
    this.nested[c.name] = c, this instanceof s || this instanceof a || this instanceof i || this instanceof e || c._edition || (c._edition = c._defaultEdition), this._needsRecursiveFeatureResolution = !0, this._needsRecursiveResolve = !0;
    for (var v = this; v = v.parent; )
      v._needsRecursiveFeatureResolution = !0, v._needsRecursiveResolve = !0;
    return c.onAdd(this), u(this);
  }, n.prototype.remove = function(c) {
    if (!(c instanceof t))
      throw TypeError("object must be a ReflectionObject");
    if (c.parent !== this)
      throw Error(c + " is not a member of " + this);
    return delete this.nested[c.name], Object.keys(this.nested).length || (this.nested = void 0), c.onRemove(this), u(this);
  }, n.prototype.define = function(c, h) {
    if (r.isString(c))
      c = c.split(".");
    else if (!Array.isArray(c))
      throw TypeError("illegal path");
    if (c && c.length && c[0] === "")
      throw Error("path must be relative");
    if (c.length > r.recursionLimit)
      throw Error("max depth exceeded");
    for (var p = this; c.length > 0; ) {
      var g = c.shift();
      if (p.nested && p.nested[g]) {
        if (p = p.nested[g], !(p instanceof n))
          throw Error("path conflicts with non-namespace objects");
      } else
        p.add(p = new n(g));
    }
    return h && p.addJSON(h), p;
  }, n.prototype.resolveAll = function() {
    if (!this._needsRecursiveResolve) return this;
    this._resolveFeaturesRecursive(this._edition);
    var c = this.nestedArray, h = 0;
    for (this.resolve(); h < c.length; )
      c[h] instanceof n ? c[h++].resolveAll() : c[h++].resolve();
    return this._needsRecursiveResolve = !1, this;
  }, n.prototype._resolveFeaturesRecursive = function(c) {
    return this._needsRecursiveFeatureResolution ? (this._needsRecursiveFeatureResolution = !1, c = this._edition || c, t.prototype._resolveFeaturesRecursive.call(this, c), this.nestedArray.forEach((h) => {
      h._resolveFeaturesRecursive(c);
    }), this) : this;
  }, n.prototype.lookup = function(c, h, p) {
    if (typeof h == "boolean" ? (p = h, h = void 0) : h && !Array.isArray(h) && (h = [h]), r.isString(c) && c.length) {
      if (c === ".")
        return this.root;
      c = c.split(".");
    } else if (!c.length)
      return this;
    var g = c.join(".");
    if (c[0] === "")
      return this.root.lookup(c.slice(1), h);
    var v = this.root._fullyQualifiedObjects && this.root._fullyQualifiedObjects["." + g];
    if (v && (!h || h.indexOf(v.constructor) > -1) || (v = this._lookupImpl(c, g), v && (!h || h.indexOf(v.constructor) > -1)))
      return v;
    if (p)
      return null;
    for (var T = this; T.parent; ) {
      if (v = T.parent._lookupImpl(c, g), v && (!h || h.indexOf(v.constructor) > -1))
        return v;
      T = T.parent;
    }
    return null;
  }, n.prototype._lookupImpl = function(c, h) {
    if (Object.prototype.hasOwnProperty.call(this._lookupCache, h))
      return this._lookupCache[h];
    var p = this.get(c[0]), g = null;
    if (p)
      c.length === 1 ? g = p : p instanceof n && (c = c.slice(1), g = p._lookupImpl(c, c.join(".")));
    else
      for (var v = 0; v < this.nestedArray.length; ++v)
        if (this._nestedArray[v] instanceof n && (p = this._nestedArray[v]._lookupImpl(c, h))) {
          g = p;
          break;
        }
    return this._lookupCache[h] = g, g;
  }, n.prototype.lookupType = function(c) {
    var h = this.lookup(c, [s]);
    if (!h)
      throw Error("no such type: " + c);
    return h;
  }, n.prototype.lookupEnum = function(c) {
    var h = this.lookup(c, [i]);
    if (!h)
      throw Error("no such Enum '" + c + "' in " + this);
    return h;
  }, n.prototype.lookupTypeOrEnum = function(c) {
    var h = this.lookup(c, [s, i]);
    if (!h)
      throw Error("no such Type or Enum '" + c + "' in " + this);
    return h;
  }, n.prototype.lookupService = function(c) {
    var h = this.lookup(c, [a]);
    if (!h)
      throw Error("no such Service '" + c + "' in " + this);
    return h;
  }, n._configure = function(l, c, h) {
    s = l, a = c, i = h;
  }, ft;
}
var lt, gr;
function Vt() {
  if (gr) return lt;
  gr = 1, lt = o;
  var t = Ne();
  ((o.prototype = Object.create(t.prototype)).constructor = o).className = "MapField";
  var e = Re(), r = te();
  function o(s, a, i, f, n, u) {
    if (t.call(this, s, a, f, void 0, void 0, n, u), !r.isString(i))
      throw TypeError("keyType must be a string");
    this.keyType = i, this.resolvedKeyType = null, this.map = !0;
  }
  return o.fromJSON = function(a, i) {
    return new o(a, i.id, i.keyType, i.type, i.options, i.comment);
  }, o.prototype.toJSON = function(a) {
    var i = a ? !!a.keepComments : !1;
    return r.toObject([
      "keyType",
      this.keyType,
      "type",
      this.type,
      "id",
      this.id,
      "extend",
      this.extend,
      "options",
      this.options,
      "comment",
      i ? this.comment : void 0
    ]);
  }, o.prototype.resolve = function() {
    if (this.resolved)
      return this;
    if (e.mapKey[this.keyType] === void 0)
      throw Error("invalid key type: " + this.keyType);
    return t.prototype.resolve.call(this);
  }, o.d = function(a, i, f) {
    return typeof f == "function" ? f = r.decorateType(f).name : f && typeof f == "object" && (f = r.decorateEnum(f).name), function(u, l) {
      r.decorateType(u.constructor).add(new o(l, a, i, f));
    };
  }, lt;
}
var ct, yr;
function Jt() {
  if (yr) return ct;
  yr = 1, ct = r;
  var t = Se();
  ((r.prototype = Object.create(t.prototype)).constructor = r).className = "Method";
  var e = te();
  function r(o, s, a, i, f, n, u, l, c) {
    if (e.isObject(f) ? (u = f, f = n = void 0) : e.isObject(n) && (u = n, n = void 0), !(s === void 0 || e.isString(s)))
      throw TypeError("type must be a string");
    if (!e.isString(a))
      throw TypeError("requestType must be a string");
    if (!e.isString(i))
      throw TypeError("responseType must be a string");
    t.call(this, o, u), this.type = s || "rpc", this.requestType = a, this.requestStream = f ? !0 : void 0, this.responseType = i, this.responseStream = n ? !0 : void 0, this.resolvedRequestType = null, this.resolvedResponseType = null, this.comment = l, this.parsedOptions = c;
  }
  return r.fromJSON = function(s, a) {
    return new r(s, a.type, a.requestType, a.responseType, a.requestStream, a.responseStream, a.options, a.comment, a.parsedOptions);
  }, r.prototype.toJSON = function(s) {
    var a = s ? !!s.keepComments : !1;
    return e.toObject([
      "type",
      this.type !== "rpc" && /* istanbul ignore next */
      this.type || void 0,
      "requestType",
      this.requestType,
      "requestStream",
      this.requestStream,
      "responseType",
      this.responseType,
      "responseStream",
      this.responseStream,
      "options",
      this.options,
      "comment",
      a ? this.comment : void 0,
      "parsedOptions",
      this.parsedOptions
    ]);
  }, r.prototype.resolve = function() {
    return this.resolved ? this : (this.resolvedRequestType = this.parent.lookupType(this.requestType), this.resolvedResponseType = this.parent.lookupType(this.responseType), t.prototype.resolve.call(this));
  }, ct;
}
var ht, vr;
function Gt() {
  if (vr) return ht;
  vr = 1, ht = s;
  var t = Ge();
  ((s.prototype = Object.create(t.prototype)).constructor = s).className = "Service";
  var e = Jt(), r = te(), o = qt;
  function s(i, f) {
    t.call(this, i, f), this.methods = {}, this._methodsArray = null;
  }
  s.fromJSON = function(f, n, u) {
    u = r.checkDepth(u);
    var l = new s(f, n.options);
    if (n.methods)
      for (var c = Object.keys(n.methods), h = 0; h < c.length; ++h)
        l.add(e.fromJSON(c[h], n.methods[c[h]]));
    return n.nested && l.addJSON(n.nested, u), n.edition && (l._edition = n.edition), l.comment = n.comment, l._defaultEdition = "proto3", l;
  }, s.prototype.toJSON = function(f) {
    var n = t.prototype.toJSON.call(this, f), u = f ? !!f.keepComments : !1;
    return r.toObject([
      "edition",
      this._editionToJSON(),
      "options",
      n && n.options || void 0,
      "methods",
      t.arrayToJSON(this.methodsArray, f) || /* istanbul ignore next */
      {},
      "nested",
      n && n.nested || void 0,
      "comment",
      u ? this.comment : void 0
    ]);
  }, Object.defineProperty(s.prototype, "methodsArray", {
    get: function() {
      return this._methodsArray || (this._methodsArray = r.toArray(this.methods));
    }
  });
  function a(i) {
    return i._methodsArray = null, i;
  }
  return s.prototype.get = function(f) {
    return Object.prototype.hasOwnProperty.call(this.methods, f) ? this.methods[f] : t.prototype.get.call(this, f);
  }, s.prototype.resolveAll = function() {
    if (!this._needsRecursiveResolve) return this;
    t.prototype.resolve.call(this);
    for (var f = this.methodsArray, n = 0; n < f.length; ++n)
      f[n].resolve();
    return this;
  }, s.prototype._resolveFeaturesRecursive = function(f) {
    return this._needsRecursiveFeatureResolution ? (f = this._edition || f, t.prototype._resolveFeaturesRecursive.call(this, f), this.methodsArray.forEach((n) => {
      n._resolveFeaturesRecursive(f);
    }), this) : this;
  }, s.prototype.add = function(f) {
    if (this.get(f.name))
      throw Error("duplicate name '" + f.name + "' in " + this);
    return f instanceof e ? f.name === "__proto__" ? this : (this.methods[f.name] = f, f.parent = this, a(this)) : t.prototype.add.call(this, f);
  }, s.prototype.remove = function(f) {
    if (f instanceof e) {
      if (this.methods[f.name] !== f)
        throw Error(f + " is not a member of " + this);
      return delete this.methods[f.name], f.parent = null, a(this);
    }
    return t.prototype.remove.call(this, f);
  }, s.prototype.create = function(f, n, u) {
    for (var l = new o.Service(f, n, u), c = 0, h; c < /* initializes */
    this.methodsArray.length; ++c) {
      var p = r.lcFirst((h = this._methodsArray[c]).resolve().name).replace(/[^$\w_]/g, "");
      l[p] = /* @__PURE__ */ function(g, v, T) {
        return function(m, w) {
          return o.Service.prototype.rpcCall.call(this, g, v, T, m, w);
        };
      }(h, h.resolvedRequestType.ctor, h.resolvedResponseType.ctor);
    }
    return l;
  }, ht;
}
var Wt = ge, li = me();
function ge(t) {
  if (t)
    for (var e = Object.keys(t), r = 0; r < e.length; ++r) {
      var o = e[r];
      o !== "__proto__" && (this[o] = t[o]);
    }
}
ge.create = function(e) {
  return this.$type.create(e);
};
ge.encode = function(e, r) {
  return this.$type.encode(e, r);
};
ge.encodeDelimited = function(e, r) {
  return this.$type.encodeDelimited(e, r);
};
ge.decode = function(e) {
  return this.$type.decode(e);
};
ge.decodeDelimited = function(e) {
  return this.$type.decodeDelimited(e);
};
ge.verify = function(e) {
  return this.$type.verify(e);
};
ge.fromObject = function(e) {
  return this.$type.fromObject(e);
};
ge.toObject = function(e, r) {
  return this.$type.toObject(e, r);
};
ge.prototype.toJSON = function() {
  return this.$type.toObject(this, li.toJSONOptions);
};
var dt, wr;
function cn() {
  if (wr) return dt;
  wr = 1, dt = s;
  var t = ye(), e = Re(), r = te();
  function o(a) {
    return "missing required '" + a.name + "'";
  }
  function s(a) {
    for (var i = r.codegen(["r", "l", "e", "n"], a.name + "$decode")("if(!(r instanceof Reader))")("r=Reader.create(r)")("if(n===undefined)n=0")("if(n>Reader.recursionLimit)")('throw Error("maximum nesting depth exceeded")')("var c=l===undefined?r.len:r.pos+l,m=new this.ctor" + (a.fieldsArray.filter(function(h) {
      return h.map;
    }).length ? ",k,value" : ""))("while(r.pos<c){")("var t=r.uint32()")("if(t===e)")("break")("switch(t>>>3){"), f = 0; f < /* initializes */
    a.fieldsArray.length; ++f) {
      var n = a._fieldsArray[f].resolve(), u = n.resolvedType instanceof t ? "int32" : n.type, l = "m" + r.safeProp(n.name);
      i("case %i: {", n.id), n.map ? (i("if(%s===util.emptyObject)", l)("%s={}", l)("var c2 = r.uint32()+r.pos"), e.defaults[n.keyType] !== void 0 ? i("k=%j", e.defaults[n.keyType]) : i("k=null"), e.defaults[u] !== void 0 ? i("value=%j", e.defaults[u]) : i("value=null"), i("while(r.pos<c2){")("var tag2=r.uint32()")("switch(tag2>>>3){")("case 1: k=r.%s(); break", n.keyType)("case 2:"), e.basic[u] === void 0 ? i("value=types[%i].decode(r,r.uint32(),undefined,n+1)", f) : i("value=r.%s()", u), i("break")("default:")("r.skipType(tag2&7,n)")("break")("}")("}"), e.long[n.keyType] !== void 0 ? i('%s[typeof k==="object"?util.longToHash(k):k]=value', l) : (n.keyType === "string" && i('if(k==="__proto__")')("util.makeProp(%s,k)", l), i("%s[k]=value", l))) : n.repeated ? (i("if(!(%s&&%s.length))", l, l)("%s=[]", l), e.packed[u] !== void 0 && i("if((t&7)===2){")("var c2=r.uint32()+r.pos")("while(r.pos<c2)")("%s.push(r.%s())", l, u)("}else"), e.basic[u] === void 0 ? i(n.delimited ? "%s.push(types[%i].decode(r,undefined,((t&~7)|4),n+1))" : "%s.push(types[%i].decode(r,r.uint32(),undefined,n+1))", l, f) : i("%s.push(r.%s())", l, u)) : e.basic[u] === void 0 ? i(n.delimited ? "%s=types[%i].decode(r,undefined,((t&~7)|4),n+1)" : "%s=types[%i].decode(r,r.uint32(),undefined,n+1)", l, f) : i("%s=r.%s()", l, u), i("break")("}");
    }
    for (i("default:")("r.skipType(t&7,n)")("break")("}")("}"), f = 0; f < a._fieldsArray.length; ++f) {
      var c = a._fieldsArray[f];
      c.required && i("if(!Object.hasOwnProperty.call(m,%j))", c.name)("throw util.ProtocolError(%j,{instance:m})", o(c));
    }
    return i("return m");
  }
  return dt;
}
var pt, _r;
function hn() {
  if (_r) return pt;
  _r = 1, pt = a;
  var t = ye(), e = te();
  function r(i, f) {
    return i.name + ": " + f + (i.repeated && f !== "array" ? "[]" : i.map && f !== "object" ? "{k:" + i.keyType + "}" : "") + " expected";
  }
  function o(i, f, n, u) {
    if (f.resolvedType)
      if (f.resolvedType instanceof t) {
        i("switch(%s){", u)("default:")("return%j", r(f, "enum value"));
        for (var l = Object.keys(f.resolvedType.values), c = 0; c < l.length; ++c) i("case %i:", f.resolvedType.values[l[c]]);
        i("break")("}");
      } else
        i("{")("var e=types[%i].verify(%s,n+1);", n, u)("if(e)")("return%j+e", f.name + ".")("}");
    else
      switch (f.type) {
        case "int32":
        case "uint32":
        case "sint32":
        case "fixed32":
        case "sfixed32":
          i("if(!util.isInteger(%s))", u)("return%j", r(f, "integer"));
          break;
        case "int64":
        case "uint64":
        case "sint64":
        case "fixed64":
        case "sfixed64":
          i("if(!util.isInteger(%s)&&!(%s&&util.isInteger(%s.low)&&util.isInteger(%s.high)))", u, u, u, u)("return%j", r(f, "integer|Long"));
          break;
        case "float":
        case "double":
          i('if(typeof %s!=="number")', u)("return%j", r(f, "number"));
          break;
        case "bool":
          i('if(typeof %s!=="boolean")', u)("return%j", r(f, "boolean"));
          break;
        case "string":
          i("if(!util.isString(%s))", u)("return%j", r(f, "string"));
          break;
        case "bytes":
          i('if(!(%s&&typeof %s.length==="number"||util.isString(%s)))', u, u, u)("return%j", r(f, "buffer"));
          break;
      }
    return i;
  }
  function s(i, f, n) {
    switch (f.keyType) {
      case "int32":
      case "uint32":
      case "sint32":
      case "fixed32":
      case "sfixed32":
        i("if(!util.key32Re.test(%s))", n)("return%j", r(f, "integer key"));
        break;
      case "int64":
      case "uint64":
      case "sint64":
      case "fixed64":
      case "sfixed64":
        i("if(!util.key64Re.test(%s))", n)("return%j", r(f, "integer|Long key"));
        break;
      case "bool":
        i("if(!util.key2Re.test(%s))", n)("return%j", r(f, "boolean key"));
        break;
    }
    return i;
  }
  function a(i) {
    var f = e.codegen(["m", "n"], i.name + "$verify")('if(typeof m!=="object"||m===null)')("return%j", "object expected")("if(n===undefined)n=0")("if(n>util.recursionLimit)")("return%j", "maximum nesting depth exceeded"), n = i.oneofsArray, u = {};
    n.length && f("var p={}");
    for (var l = 0; l < /* initializes */
    i.fieldsArray.length; ++l) {
      var c = i._fieldsArray[l].resolve(), h = "m" + e.safeProp(c.name);
      if (c.optional && f("if(%s!=null&&Object.hasOwnProperty.call(m,%j)){", h, c.name), c.map)
        f("if(!util.isObject(%s))", h)("return%j", r(c, "object"))("var k=Object.keys(%s)", h)("for(var i=0;i<k.length;++i){"), s(f, c, "k[i]"), o(f, c, l, h + "[k[i]]")("}");
      else if (c.repeated)
        f("if(!Array.isArray(%s))", h)("return%j", r(c, "array"))("for(var i=0;i<%s.length;++i){", h), o(f, c, l, h + "[i]")("}");
      else {
        if (c.partOf) {
          var p = e.safeProp(c.partOf.name);
          u[c.partOf.name] === 1 && f("if(p%s===1)", p)("return%j", c.partOf.name + ": multiple values"), u[c.partOf.name] = 1, f("p%s=1", p);
        }
        o(f, c, l, h);
      }
      c.optional && f("}");
    }
    return f("return null");
  }
  return pt;
}
var mt = {}, br;
function dn() {
  return br || (br = 1, function(t) {
    var e = t, r = ye(), o = te();
    function s(i, f, n, u) {
      var l = !1;
      if (f.resolvedType)
        if (f.resolvedType instanceof r) {
          i("switch(d%s){", u);
          for (var c = f.resolvedType.values, h = Object.keys(c), p = 0; p < h.length; ++p)
            c[h[p]] === f.typeDefault && !l && (i("default:")('if(typeof(d%s)==="number"){m%s=d%s;break}', u, u, u), f.repeated || i("break"), l = !0), i("case%j:", h[p])("case %i:", c[h[p]])("m%s=%j", u, c[h[p]])("break");
          i("}");
        } else i("if(!util.isObject(d%s))", u)("throw TypeError(%j)", f.fullName + ": object expected")("m%s=types[%i].fromObject(d%s,n+1)", u, n, u);
      else {
        var g = !1;
        switch (f.type) {
          case "double":
          case "float":
            i("m%s=Number(d%s)", u, u);
            break;
          case "uint32":
          case "fixed32":
            i("m%s=d%s>>>0", u, u);
            break;
          case "int32":
          case "sint32":
          case "sfixed32":
            i("m%s=d%s|0", u, u);
            break;
          case "uint64":
          case "fixed64":
            g = !0;
          case "int64":
          case "sint64":
          case "sfixed64":
            i("if(util.Long)")("m%s=util.Long.fromValue(d%s,%j)", u, u, g)('else if(typeof d%s==="string")', u)("m%s=parseInt(d%s,10)", u, u)('else if(typeof d%s==="number")', u)("m%s=d%s", u, u)('else if(typeof d%s==="object")', u)("m%s=new util.LongBits(d%s.low>>>0,d%s.high>>>0).toNumber(%s)", u, u, u, g ? "true" : "");
            break;
          case "bytes":
            i('if(typeof d%s==="string")', u)("util.base64.decode(d%s,m%s=util.newBuffer(util.base64.length(d%s)),0)", u, u, u)("else if(d%s.length >= 0)", u)("m%s=d%s", u, u);
            break;
          case "string":
            i("m%s=String(d%s)", u, u);
            break;
          case "bool":
            i("m%s=Boolean(d%s)", u, u);
            break;
        }
      }
      return i;
    }
    e.fromObject = function(f) {
      var n = f.fieldsArray, u = o.codegen(["d", "n"], f.name + "$fromObject")("if(d instanceof this.ctor)")("return d");
      if (!n.length) return u("return new this.ctor");
      u("if(!util.isObject(d))")("throw TypeError(%j)", f.fullName + ": object expected")("if(n===undefined)n=0")("if(n>util.recursionLimit)")('throw Error("maximum nesting depth exceeded")'), u("var m=new this.ctor");
      for (var l = 0; l < n.length; ++l) {
        var c = n[l].resolve(), h = o.safeProp(c.name);
        c.map ? (u("if(d%s){", h)("if(!util.isObject(d%s))", h)("throw TypeError(%j)", c.fullName + ": object expected")("m%s={}", h)("for(var ks=Object.keys(d%s),i=0;i<ks.length;++i){", h), u('if(ks[i]==="__proto__")')("util.makeProp(m%s,ks[i])", h), s(
          u,
          c,
          /* not sorted */
          l,
          h + "[ks[i]]"
        )("}")("}")) : c.repeated ? (u("if(d%s){", h)("if(!Array.isArray(d%s))", h)("throw TypeError(%j)", c.fullName + ": array expected")("m%s=[]", h)("for(var i=0;i<d%s.length;++i){", h), s(
          u,
          c,
          /* not sorted */
          l,
          h + "[i]"
        )("}")("}")) : (c.resolvedType instanceof r || u("if(d%s!=null){", h), s(
          u,
          c,
          /* not sorted */
          l,
          h
        ), c.resolvedType instanceof r || u("}"));
      }
      return u("return m");
    };
    function a(i, f, n, u) {
      if (f.resolvedType)
        f.resolvedType instanceof r ? i("d%s=o.enums===String?(types[%i].values[m%s]===undefined?m%s:types[%i].values[m%s]):m%s", u, n, u, u, n, u, u) : i("d%s=types[%i].toObject(m%s,o,q+1)", u, n, u);
      else {
        var l = !1;
        switch (f.type) {
          case "double":
          case "float":
            i("d%s=o.json&&!isFinite(m%s)?String(m%s):m%s", u, u, u, u);
            break;
          case "uint64":
          case "fixed64":
            l = !0;
          case "int64":
          case "sint64":
          case "sfixed64":
            i('if(typeof BigInt!=="undefined"&&o.longs===BigInt)')('d%s=typeof m%s==="number"?BigInt(m%s):util.Long.fromBits(m%s.low>>>0,m%s.high>>>0,%j).toBigInt()', u, u, u, u, u, l)('else if(typeof m%s==="number")', u)("d%s=o.longs===String?String(m%s):m%s", u, u, u)("else")("d%s=o.longs===String?util.Long.prototype.toString.call(m%s):o.longs===Number?new util.LongBits(m%s.low>>>0,m%s.high>>>0).toNumber(%s):m%s", u, u, u, u, l ? "true" : "", u);
            break;
          case "bytes":
            i("d%s=o.bytes===String?util.base64.encode(m%s,0,m%s.length):o.bytes===Array?Array.prototype.slice.call(m%s):m%s", u, u, u, u, u);
            break;
          default:
            i("d%s=m%s", u, u);
            break;
        }
      }
      return i;
    }
    e.toObject = function(f) {
      var n = f.fieldsArray.slice().sort(o.compareFieldsById);
      if (!n.length)
        return o.codegen()("return {}");
      for (var u = o.codegen(["m", "o", "q"], f.name + "$toObject")("if(!o)")("o={}")("if(q===undefined)q=0")("if(q>util.recursionLimit)")('throw Error("max depth exceeded")')("var d={}"), l = [], c = [], h = [], p = 0; p < n.length; ++p)
        n[p].partOf || (n[p].resolve().repeated ? l : n[p].map ? c : h).push(n[p]);
      if (l.length) {
        for (u("if(o.arrays||o.defaults){"), p = 0; p < l.length; ++p) u("d%s=[]", o.safeProp(l[p].name));
        u("}");
      }
      if (c.length) {
        for (u("if(o.objects||o.defaults){"), p = 0; p < c.length; ++p) u("d%s={}", o.safeProp(c[p].name));
        u("}");
      }
      if (h.length) {
        for (u("if(o.defaults){"), p = 0; p < h.length; ++p) {
          var g = h[p], v = o.safeProp(g.name);
          if (g.resolvedType instanceof r) u("d%s=o.enums===String?%j:%j", v, g.resolvedType.valuesById[g.typeDefault], g.typeDefault);
          else if (g.long) u("if(util.Long){")("var n=new util.Long(%i,%i,%j)", g.typeDefault.low, g.typeDefault.high, g.typeDefault.unsigned)('d%s=o.longs===String?n.toString():o.longs===Number?n.toNumber():typeof BigInt!=="undefined"&&o.longs===BigInt?n.toBigInt():n', v)("}else")('d%s=o.longs===String?%j:typeof BigInt!=="undefined"&&o.longs===BigInt?BigInt(%j):%i', v, g.typeDefault.toString(), g.typeDefault.toString(), g.typeDefault.toNumber());
          else if (g.bytes) {
            var T = Array.prototype.slice.call(g.typeDefault);
            u("if(o.bytes===String)d%s=%j", v, String.fromCharCode.apply(String, g.typeDefault))("else{")("d%s=%j", v, T)("if(o.bytes!==Array)d%s=util.newBuffer(d%s)", v, v)("}");
          } else u("d%s=%j", v, g.typeDefault);
        }
        u("}");
      }
      var N = !1;
      for (p = 0; p < n.length; ++p) {
        var g = n[p], m = f._fieldsArray.indexOf(g), v = o.safeProp(g.name);
        g.map ? (N || (N = !0, u("var ks2")), u("if(m%s&&(ks2=Object.keys(m%s)).length){", v, v)("d%s={}", v)("for(var j=0;j<ks2.length;++j){"), u('if(ks2[j]==="__proto__")')("util.makeProp(d%s,ks2[j])", v), a(
          u,
          g,
          /* sorted */
          m,
          v + "[ks2[j]]"
        )("}")) : g.repeated ? (u("if(m%s&&m%s.length){", v, v)("d%s=[]", v)("for(var j=0;j<m%s.length;++j){", v), a(
          u,
          g,
          /* sorted */
          m,
          v + "[j]"
        )("}")) : (u("if(m%s!=null&&Object.hasOwnProperty.call(m,%j)){", v, g.name), a(
          u,
          g,
          /* sorted */
          m,
          v
        ), g.partOf && u("if(o.oneofs)")("d%s=%j", o.safeProp(g.partOf.name), g.name)), u("}");
      }
      return u("return d");
    };
  }(mt)), mt;
}
var Ht = {};
(function(t) {
  var e = t, r = Wt, o = me();
  e[".google.protobuf.Any"] = {
    fromObject: function(s, a) {
      if (s && s["@type"]) {
        var i = s["@type"].substring(s["@type"].lastIndexOf("/") + 1), f = this.lookup(i);
        if (f) {
          var n = s["@type"].charAt(0) === "." ? s["@type"].slice(1) : s["@type"];
          return n.indexOf("/") === -1 && (n = "/" + n), this.create({
            type_url: n,
            value: f.encode(f.fromObject(s, a === void 0 ? 1 : a + 1)).finish()
          });
        }
      }
      return this.fromObject(s, a);
    },
    toObject: function(s, a, i) {
      if (i === void 0 && (i = 0), i > o.recursionLimit)
        throw Error("max depth exceeded");
      var f = "type.googleapis.com/", n = "", u = "";
      if (a && a.json && s.type_url && s.value) {
        u = s.type_url.substring(s.type_url.lastIndexOf("/") + 1), n = s.type_url.substring(0, s.type_url.lastIndexOf("/") + 1);
        var l = this.lookup(u);
        l && (s = l.decode(s.value, void 0, void 0, i + 1));
      }
      if (!(s instanceof this.ctor) && s instanceof r) {
        var c = s.$type.toObject(s, a, i + 1), h = s.$type.fullName[0] === "." ? s.$type.fullName.slice(1) : s.$type.fullName;
        return n === "" && (n = f), u = n + h, c["@type"] = u, c;
      }
      return this.toObject(s, a, i);
    }
  };
})(Ht);
var gt, Or;
function Zt() {
  if (Or) return gt;
  Or = 1, gt = v;
  var t = Ge();
  ((v.prototype = Object.create(t.prototype)).constructor = v).className = "Type";
  var e = ye(), r = Le(), o = Ne(), s = Vt(), a = Gt(), i = Wt, f = Ut, n = Bt, u = te(), l = pn(), c = cn(), h = hn(), p = dn(), g = Ht;
  function v(N, m) {
    N = N.replace(/\W/g, ""), t.call(this, N, m), this.fields = {}, this.oneofs = void 0, this.extensions = void 0, this.reserved = void 0, this.group = void 0, this._fieldsById = null, this._fieldsArray = null, this._oneofsArray = null, this._ctor = null;
  }
  Object.defineProperties(v.prototype, {
    /**
     * Message fields by id.
     * @name Type#fieldsById
     * @type {Object.<number,Field>}
     * @readonly
     */
    fieldsById: {
      get: function() {
        if (this._fieldsById)
          return this._fieldsById;
        this._fieldsById = {};
        for (var N = Object.keys(this.fields), m = 0; m < N.length; ++m) {
          var w = this.fields[N[m]], y = w.id;
          if (this._fieldsById[y])
            throw Error("duplicate id " + y + " in " + this);
          this._fieldsById[y] = w;
        }
        return this._fieldsById;
      }
    },
    /**
     * Fields of this message as an array for iteration.
     * @name Type#fieldsArray
     * @type {Field[]}
     * @readonly
     */
    fieldsArray: {
      get: function() {
        return this._fieldsArray || (this._fieldsArray = u.toArray(this.fields));
      }
    },
    /**
     * Oneofs of this message as an array for iteration.
     * @name Type#oneofsArray
     * @type {OneOf[]}
     * @readonly
     */
    oneofsArray: {
      get: function() {
        return this._oneofsArray || (this._oneofsArray = u.toArray(this.oneofs));
      }
    },
    /**
     * The registered constructor, if any registered, otherwise a generic constructor.
     * Assigning a function replaces the internal constructor. If the function does not extend {@link Message} yet, its prototype will be setup accordingly and static methods will be populated. If it already extends {@link Message}, it will just replace the internal constructor.
     * @name Type#ctor
     * @type {Constructor<{}>}
     */
    ctor: {
      get: function() {
        return this._ctor || (this.ctor = v.generateConstructor(this)());
      },
      set: function(N) {
        var m = N.prototype;
        m instanceof i || ((N.prototype = new i()).constructor = N, u.merge(N.prototype, m)), N.$type = N.prototype.$type = this, u.merge(N, i, !0), this._ctor = N;
        for (var w = 0; w < /* initializes */
        this.fieldsArray.length; ++w)
          this._fieldsArray[w].resolve();
        var y = {};
        for (w = 0; w < /* initializes */
        this.oneofsArray.length; ++w)
          y[this._oneofsArray[w].resolve().name] = {
            get: u.oneOfGetter(this._oneofsArray[w].oneof),
            set: u.oneOfSetter(this._oneofsArray[w].oneof)
          };
        w && Object.defineProperties(N.prototype, y);
      }
    }
  }), v.generateConstructor = function(m) {
    for (var w = u.codegen(["p"], m.name), y = 0, b; y < m.fieldsArray.length; ++y)
      (b = m._fieldsArray[y]).map ? w("this%s={}", u.safeProp(b.name)) : b.repeated && w("this%s=[]", u.safeProp(b.name));
    return w('if(p)for(var ks=Object.keys(p),i=0;i<ks.length;++i)if(p[ks[i]]!=null&&ks[i]!=="__proto__")')("this[ks[i]]=p[ks[i]]");
  };
  function T(N) {
    return N._fieldsById = N._fieldsArray = N._oneofsArray = null, delete N.encode, delete N.decode, delete N.verify, N;
  }
  return v.fromJSON = function(m, w, y) {
    if (y === void 0 && (y = 0), y > u.nestingLimit)
      throw Error("max depth exceeded");
    var b = new v(m, w.options);
    b.extensions = w.extensions, b.reserved = w.reserved;
    for (var E = Object.keys(w.fields), R = 0; R < E.length; ++R)
      b.add(
        (typeof w.fields[E[R]].keyType < "u" ? s.fromJSON : o.fromJSON)(E[R], w.fields[E[R]])
      );
    if (w.oneofs)
      for (E = Object.keys(w.oneofs), R = 0; R < E.length; ++R)
        b.add(r.fromJSON(E[R], w.oneofs[E[R]]));
    if (w.nested)
      for (E = Object.keys(w.nested), R = 0; R < E.length; ++R) {
        var U = w.nested[E[R]];
        b.add(
          // most to least likely
          (U.id !== void 0 ? o.fromJSON : U.fields !== void 0 ? v.fromJSON : U.values !== void 0 ? e.fromJSON : U.methods !== void 0 ? a.fromJSON : t.fromJSON)(E[R], U, y + 1)
        );
      }
    return w.extensions && w.extensions.length && (b.extensions = w.extensions), w.reserved && w.reserved.length && (b.reserved = w.reserved), w.group && (b.group = !0), w.comment && (b.comment = w.comment), w.edition && (b._edition = w.edition), b._defaultEdition = "proto3", b;
  }, v.prototype.toJSON = function(m) {
    var w = t.prototype.toJSON.call(this, m), y = m ? !!m.keepComments : !1;
    return u.toObject([
      "edition",
      this._editionToJSON(),
      "options",
      w && w.options || void 0,
      "oneofs",
      t.arrayToJSON(this.oneofsArray, m),
      "fields",
      t.arrayToJSON(this.fieldsArray.filter(function(b) {
        return !b.declaringField;
      }), m) || {},
      "extensions",
      this.extensions && this.extensions.length ? this.extensions : void 0,
      "reserved",
      this.reserved && this.reserved.length ? this.reserved : void 0,
      "group",
      this.group || void 0,
      "nested",
      w && w.nested || void 0,
      "comment",
      y ? this.comment : void 0
    ]);
  }, v.prototype.resolveAll = function() {
    if (!this._needsRecursiveResolve) return this;
    t.prototype.resolveAll.call(this);
    var m = this.oneofsArray;
    for (y = 0; y < m.length; )
      m[y++].resolve();
    for (var w = this.fieldsArray, y = 0; y < w.length; )
      w[y++].resolve();
    return this;
  }, v.prototype._resolveFeaturesRecursive = function(m) {
    return this._needsRecursiveFeatureResolution ? (m = this._edition || m, t.prototype._resolveFeaturesRecursive.call(this, m), this.oneofsArray.forEach((w) => {
      w._resolveFeatures(m);
    }), this.fieldsArray.forEach((w) => {
      w._resolveFeatures(m);
    }), this) : this;
  }, v.prototype.get = function(m) {
    return Object.prototype.hasOwnProperty.call(this.fields, m) ? this.fields[m] : this.oneofs && Object.prototype.hasOwnProperty.call(this.oneofs, m) ? this.oneofs[m] : this.nested && Object.prototype.hasOwnProperty.call(this.nested, m) ? this.nested[m] : null;
  }, v.prototype.add = function(m) {
    if (this.get(m.name))
      throw Error("duplicate name '" + m.name + "' in " + this);
    if (m instanceof o && m.extend === void 0) {
      if (this._fieldsById ? (
        /* istanbul ignore next */
        this._fieldsById[m.id]
      ) : this.fieldsById[m.id])
        throw Error("duplicate id " + m.id + " in " + this);
      if (this.isReservedId(m.id))
        throw Error("id " + m.id + " is reserved in " + this);
      if (this.isReservedName(m.name) || m.name.charAt(0) === "$")
        throw Error("name '" + m.name + "' is reserved in " + this);
      return m.name === "__proto__" ? this : (m.parent && m.parent.remove(m), this.fields[m.name] = m, m.message = this, m.onAdd(this), T(this));
    }
    if (m instanceof r) {
      if (m.name.charAt(0) === "$")
        throw Error("name '" + m.name + "' is reserved in " + this);
      return m.name === "__proto__" ? this : (this.oneofs || (this.oneofs = {}), this.oneofs[m.name] = m, m.onAdd(this), T(this));
    }
    return t.prototype.add.call(this, m);
  }, v.prototype.remove = function(m) {
    if (m instanceof o && m.extend === void 0) {
      if (!this.fields || this.fields[m.name] !== m)
        throw Error(m + " is not a member of " + this);
      return delete this.fields[m.name], m.parent = null, m.onRemove(this), T(this);
    }
    if (m instanceof r) {
      if (!this.oneofs || this.oneofs[m.name] !== m)
        throw Error(m + " is not a member of " + this);
      return delete this.oneofs[m.name], m.parent = null, m.onRemove(this), T(this);
    }
    return t.prototype.remove.call(this, m);
  }, v.prototype.isReservedId = function(m) {
    return t.isReservedId(this.reserved, m);
  }, v.prototype.isReservedName = function(m) {
    return t.isReservedName(this.reserved, m);
  }, v.prototype.create = function(m) {
    return new this.ctor(m);
  }, v.prototype.setup = function() {
    for (var m = this.fullName, w = [], y = 0; y < /* initializes */
    this.fieldsArray.length; ++y)
      w.push(this._fieldsArray[y].resolve().resolvedType);
    this.encode = l(this)({
      Writer: n,
      types: w,
      util: u
    }), this.decode = c(this)({
      Reader: f,
      types: w,
      util: u
    }), this.verify = h(this)({
      types: w,
      util: u
    }), this.fromObject = p.fromObject(this)({
      types: w,
      util: u
    }), this.toObject = p.toObject(this)({
      types: w,
      util: u
    });
    var b = g[m];
    if (b) {
      var E = Object.create(this);
      E.fromObject = this.fromObject, this.fromObject = b.fromObject.bind(E), E.toObject = this.toObject, this.toObject = b.toObject.bind(E);
    }
    return this;
  }, v.prototype.encode = function(m, w) {
    return this.setup().encode.apply(this, arguments);
  }, v.prototype.encodeDelimited = function(m, w) {
    return this.encode(m, w && w.len ? w.fork() : w).ldelim();
  }, v.prototype.decode = function(m, w, y, b) {
    return this.setup().decode(m, w, y, b);
  }, v.prototype.decodeDelimited = function(m) {
    return m instanceof f || (m = f.create(m)), this.decode(m, m.uint32());
  }, v.prototype.verify = function(m, w) {
    return this.setup().verify(m, w);
  }, v.prototype.fromObject = function(m, w) {
    return this.setup().fromObject(m, w);
  }, v.prototype.toObject = function(m, w) {
    return this.setup().toObject.apply(this, arguments);
  }, v.d = function(m) {
    return function(y) {
      u.decorateType(y, m);
    };
  }, gt;
}
var yt, Er;
function Xt() {
  if (Er) return yt;
  Er = 1, yt = n;
  var t = Ge();
  ((n.prototype = Object.create(t.prototype)).constructor = n).className = "Root";
  var e = Ne(), r = ye(), o = Le(), s = te(), a, i, f;
  function n(h) {
    t.call(this, "", h), this.deferred = [], this.files = [], this._edition = "proto2", this._fullyQualifiedObjects = {};
  }
  n.fromJSON = function(p, g, v) {
    return v = s.checkDepth(v), g || (g = new n()), p.options && g.setOptions(p.options), g.addJSON(p.nested, v).resolveAll();
  }, n.prototype.resolvePath = s.path.resolve, n.prototype.fetch = s.fetch;
  function u() {
  }
  n.prototype.load = function h(p, g, v) {
    typeof g == "function" && (v = g, g = void 0);
    var T = this;
    if (!v)
      return s.asPromise(h, T, p, g);
    var N = v === u;
    function m(A, L) {
      if (v) {
        if (N)
          throw A;
        L && L.resolveAll();
        var B = v;
        v = null, B(A, L);
      }
    }
    function w(A) {
      var L = A.lastIndexOf("google/protobuf/");
      if (L > -1) {
        var B = A.substring(L);
        if (B in f) return B;
      }
      return null;
    }
    function y(A, L, B) {
      B === void 0 && (B = 0);
      try {
        if (B > s.recursionLimit)
          throw Error("max depth exceeded");
        if (s.isString(L) && L.charAt(0) === "{" && (L = JSON.parse(L)), !s.isString(L))
          T.setOptions(L.options).addJSON(L.nested);
        else {
          i.filename = A;
          var $ = i(L, T, g), O, _ = 0;
          if ($.imports)
            for (; _ < $.imports.length; ++_)
              (O = w($.imports[_]) || T.resolvePath(A, $.imports[_])) && b(O, !1, B + 1);
          if ($.weakImports)
            for (_ = 0; _ < $.weakImports.length; ++_)
              (O = w($.weakImports[_]) || T.resolvePath(A, $.weakImports[_])) && b(O, !0, B + 1);
        }
      } catch (d) {
        m(d);
      }
      !N && !E && m(null, T);
    }
    function b(A, L, B) {
      if (B === void 0 && (B = 0), A = w(A) || A, !(T.files.indexOf(A) > -1)) {
        if (T.files.push(A), A in f) {
          N ? y(A, f[A], B) : (++E, setTimeout(function() {
            --E, y(A, f[A], B);
          }));
          return;
        }
        if (N) {
          var $;
          try {
            $ = s.fs.readFileSync(A).toString("utf8");
          } catch (O) {
            L || m(O);
            return;
          }
          y(A, $, B);
        } else
          ++E, T.fetch(A, function(O, _) {
            if (--E, !!v) {
              if (O) {
                L ? E || m(null, T) : m(O);
                return;
              }
              y(A, _, B);
            }
          });
      }
    }
    var E = 0;
    s.isString(p) && (p = [p]);
    for (var R = 0, U; R < p.length; ++R)
      (U = T.resolvePath("", p[R])) && b(U);
    return N ? (T.resolveAll(), T) : (E || m(null, T), T);
  }, n.prototype.loadSync = function(p, g) {
    if (!s.isNode)
      throw Error("not supported");
    return this.load(p, g, u);
  }, n.prototype.resolveAll = function() {
    if (!this._needsRecursiveResolve) return this;
    if (this.deferred.length)
      throw Error("unresolvable extensions: " + this.deferred.map(function(p) {
        return "'extend " + p.extend + "' in " + p.parent.fullName;
      }).join(", "));
    return t.prototype.resolveAll.call(this);
  };
  var l = /^[A-Z]/;
  function c(h, p) {
    var g = p.parent.lookup(p.extend);
    if (g) {
      var v = new e(p.fullName, p.id, p.type, p.rule, void 0, p.options);
      return g.get(v.name) || (v.declaringField = p, p.extensionField = v, g.add(v)), !0;
    }
    return !1;
  }
  return n.prototype._handleAdd = function(p) {
    if (p instanceof e)
      /* an extension field (implies not part of a oneof) */
      p.extend !== void 0 && /* not already handled */
      !p.extensionField && (c(this, p) || this.deferred.push(p));
    else if (p instanceof r)
      l.test(p.name) && (p.parent[p.name] = p.values);
    else if (!(p instanceof o)) {
      if (p instanceof a)
        for (var g = 0; g < this.deferred.length; )
          c(this, this.deferred[g]) ? this.deferred.splice(g, 1) : ++g;
      for (var v = 0; v < /* initializes */
      p.nestedArray.length; ++v)
        this._handleAdd(p._nestedArray[v]);
      l.test(p.name) && (p.parent[p.name] = p);
    }
    (p instanceof a || p instanceof r || p instanceof e) && (this._fullyQualifiedObjects[p.fullName] = p);
  }, n.prototype._handleRemove = function(p) {
    if (p instanceof e) {
      if (
        /* an extension field */
        p.extend !== void 0
      )
        if (
          /* already handled */
          p.extensionField
        )
          p.extensionField.parent.remove(p.extensionField), p.extensionField = null;
        else {
          var g = this.deferred.indexOf(p);
          g > -1 && this.deferred.splice(g, 1);
        }
    } else if (p instanceof r)
      l.test(p.name) && delete p.parent[p.name];
    else if (p instanceof t) {
      for (var v = 0; v < /* initializes */
      p.nestedArray.length; ++v)
        this._handleRemove(p._nestedArray[v]);
      l.test(p.name) && delete p.parent[p.name];
    }
    delete this._fullyQualifiedObjects[p.fullName];
  }, n._configure = function(h, p, g) {
    a = h, i = p, f = g;
  }, yt;
}
var kr;
function te() {
  if (kr) return at.exports;
  kr = 1;
  var t = at.exports = me(), e = an, r, o;
  t.codegen = ti, t.fetch = ai, t.path = fn, t.patterns = ln;
  var s = t.patterns.reservedRe;
  t.fs = fi, t.checkDepth = function(n) {
    if (n === void 0 && (n = 0), n > t.recursionLimit)
      throw Error("max depth exceeded");
    return n;
  }, t.toArray = function(n) {
    if (n) {
      for (var u = Object.keys(n), l = new Array(u.length), c = 0; c < u.length; )
        l[c] = n[u[c++]];
      return l;
    }
    return [];
  }, t.toObject = function(n) {
    for (var u = {}, l = 0; l < n.length; ) {
      var c = n[l++], h = n[l++];
      h !== void 0 && (u[c] = h);
    }
    return u;
  }, t.isReserved = function(n) {
    return s.test(n);
  }, t.safeProp = function(n) {
    return !/^[$\w_]+$/.test(n) || s.test(n) ? "[" + JSON.stringify(n) + "]" : "." + n;
  }, t.ucFirst = function(n) {
    return n.charAt(0).toUpperCase() + n.substring(1);
  };
  var a = /_([a-z])/g;
  t.camelCase = function(n) {
    return n.substring(0, 1) + n.substring(1).replace(a, function(u, l) {
      return l.toUpperCase();
    });
  }, t.compareFieldsById = function(n, u) {
    return n.id - u.id;
  }, t.decorateType = function(n, u) {
    if (n.$type)
      return u && n.$type.name !== u && (t.decorateRoot.remove(n.$type), n.$type.name = u, t.decorateRoot.add(n.$type)), n.$type;
    r || (r = Zt());
    var l = new r(u || n.name);
    return t.decorateRoot.add(l), l.ctor = n, Object.defineProperty(n, "$type", { value: l, enumerable: !1 }), Object.defineProperty(n.prototype, "$type", { value: l, enumerable: !1 }), l;
  };
  var i = 0;
  return t.decorateEnum = function(n) {
    if (n.$type)
      return n.$type;
    o || (o = ye());
    var u = new o("Enum" + i++, n);
    return t.decorateRoot.add(u), Object.defineProperty(n, "$type", { value: u, enumerable: !1 }), u;
  }, t.setProperty = function(n, u, l, c) {
    function h(p, g, v) {
      var T = g.shift();
      if (t.isUnsafeProperty(T))
        return p;
      if (g.length > 0)
        p[T] = h(p[T] || {}, g, v);
      else {
        var N = p[T];
        if (N && c)
          return p;
        N && (v = [].concat(N).concat(v)), p[T] = v;
      }
      return p;
    }
    if (typeof n != "object")
      throw TypeError("dst must be an object");
    if (!u)
      throw TypeError("path must be specified");
    if (u = u.split("."), u.length > t.recursionLimit)
      throw Error("max depth exceeded");
    return h(n, u, l);
  }, Object.defineProperty(t, "decorateRoot", {
    get: function() {
      return e.decorated || (e.decorated = new (Xt())());
    }
  }), at.exports;
}
var Nr;
function Re() {
  return Nr || (Nr = 1, function(t) {
    var e = t, r = te(), o = [
      "double",
      // 0
      "float",
      // 1
      "int32",
      // 2
      "uint32",
      // 3
      "sint32",
      // 4
      "fixed32",
      // 5
      "sfixed32",
      // 6
      "int64",
      // 7
      "uint64",
      // 8
      "sint64",
      // 9
      "fixed64",
      // 10
      "sfixed64",
      // 11
      "bool",
      // 12
      "string",
      // 13
      "bytes"
      // 14
    ];
    function s(a, i) {
      var f = 0, n = /* @__PURE__ */ Object.create(null);
      for (i |= 0; f < a.length; ) n[o[f + i]] = a[f++];
      return n;
    }
    e.basic = s([
      /* double   */
      1,
      /* float    */
      5,
      /* int32    */
      0,
      /* uint32   */
      0,
      /* sint32   */
      0,
      /* fixed32  */
      5,
      /* sfixed32 */
      5,
      /* int64    */
      0,
      /* uint64   */
      0,
      /* sint64   */
      0,
      /* fixed64  */
      1,
      /* sfixed64 */
      1,
      /* bool     */
      0,
      /* string   */
      2,
      /* bytes    */
      2
    ]), e.defaults = s([
      /* double   */
      0,
      /* float    */
      0,
      /* int32    */
      0,
      /* uint32   */
      0,
      /* sint32   */
      0,
      /* fixed32  */
      0,
      /* sfixed32 */
      0,
      /* int64    */
      0,
      /* uint64   */
      0,
      /* sint64   */
      0,
      /* fixed64  */
      0,
      /* sfixed64 */
      0,
      /* bool     */
      !1,
      /* string   */
      "",
      /* bytes    */
      r.emptyArray,
      /* message  */
      null
    ]), e.long = s([
      /* int64    */
      0,
      /* uint64   */
      0,
      /* sint64   */
      0,
      /* fixed64  */
      1,
      /* sfixed64 */
      1
    ], 7), e.mapKey = s([
      /* int32    */
      0,
      /* uint32   */
      0,
      /* sint32   */
      0,
      /* fixed32  */
      5,
      /* sfixed32 */
      5,
      /* int64    */
      0,
      /* uint64   */
      0,
      /* sint64   */
      0,
      /* fixed64  */
      1,
      /* sfixed64 */
      1,
      /* bool     */
      0,
      /* string   */
      2
    ], 2), e.packed = s([
      /* double   */
      1,
      /* float    */
      5,
      /* int32    */
      0,
      /* uint32   */
      0,
      /* sint32   */
      0,
      /* fixed32  */
      5,
      /* sfixed32 */
      5,
      /* int64    */
      0,
      /* uint64   */
      0,
      /* sint64   */
      0,
      /* fixed64  */
      1,
      /* sfixed64 */
      1,
      /* bool     */
      0
    ]);
  }(ot)), ot;
}
var vt, Sr;
function Ne() {
  if (Sr) return vt;
  Sr = 1, vt = i;
  var t = Se();
  ((i.prototype = Object.create(t.prototype)).constructor = i).className = "Field";
  var e = ye(), r = Re(), o = te(), s, a = /^required|optional|repeated$/;
  i.fromJSON = function(n, u) {
    var l = new i(n, u.id, u.type, u.rule, u.extend, u.options, u.comment);
    return u.edition && (l._edition = u.edition), l._defaultEdition = "proto3", l;
  };
  function i(f, n, u, l, c, h, p) {
    if (o.isObject(l) ? (p = c, h = l, l = c = void 0) : o.isObject(c) && (p = h, h = c, c = void 0), t.call(this, f, h), !o.isInteger(n) || n < 0)
      throw TypeError("id must be a non-negative integer");
    if (!o.isString(u))
      throw TypeError("type must be a string");
    if (l !== void 0 && !a.test(l = l.toString().toLowerCase()))
      throw TypeError("rule must be a string rule");
    if (c !== void 0 && !o.isString(c))
      throw TypeError("extend must be a string");
    l === "proto3_optional" && (l = "optional"), this.rule = l && l !== "optional" ? l : void 0, this.type = u, this.id = n, this.extend = c || void 0, this.repeated = l === "repeated", this.map = !1, this.message = null, this.partOf = null, this.typeDefault = null, this.defaultValue = null, this.long = o.Long ? r.long[u] !== void 0 : (
      /* istanbul ignore next */
      !1
    ), this.bytes = u === "bytes", this.resolvedType = null, this.extensionField = null, this.declaringField = null, this.comment = p;
  }
  return Object.defineProperty(i.prototype, "required", {
    get: function() {
      return this._features.field_presence === "LEGACY_REQUIRED";
    }
  }), Object.defineProperty(i.prototype, "optional", {
    get: function() {
      return !this.required;
    }
  }), Object.defineProperty(i.prototype, "delimited", {
    get: function() {
      return this.resolvedType instanceof s && this._features.message_encoding === "DELIMITED";
    }
  }), Object.defineProperty(i.prototype, "packed", {
    get: function() {
      return this._features.repeated_field_encoding === "PACKED";
    }
  }), Object.defineProperty(i.prototype, "hasPresence", {
    get: function() {
      return this.repeated || this.map ? !1 : this.partOf || // oneofs
      this.declaringField || this.extensionField || // extensions
      this._features.field_presence !== "IMPLICIT";
    }
  }), i.prototype.setOption = function(n, u, l) {
    return t.prototype.setOption.call(this, n, u, l);
  }, i.prototype.toJSON = function(n) {
    var u = n ? !!n.keepComments : !1;
    return o.toObject([
      "edition",
      this._editionToJSON(),
      "rule",
      this.rule !== "optional" && this.rule || void 0,
      "type",
      this.type,
      "id",
      this.id,
      "extend",
      this.extend,
      "options",
      this.options,
      "comment",
      u ? this.comment : void 0
    ]);
  }, i.prototype.resolve = function() {
    if (this.resolved)
      return this;
    if ((this.typeDefault = r.defaults[this.type]) === void 0 ? (this.resolvedType = (this.declaringField ? this.declaringField.parent : this.parent).lookupTypeOrEnum(this.type), this.resolvedType instanceof s ? this.typeDefault = null : this.typeDefault = this.resolvedType.values[Object.keys(this.resolvedType.values)[0]]) : this.options && this.options.proto3_optional && (this.typeDefault = null), this.options && this.options.default != null && (this.typeDefault = this.options.default, this.resolvedType instanceof e && typeof this.typeDefault == "string" && (this.typeDefault = this.resolvedType.values[this.typeDefault])), this.options && (this.options.packed !== void 0 && this.resolvedType && !(this.resolvedType instanceof e) && delete this.options.packed, Object.keys(this.options).length || (this.options = void 0)), this.long)
      this.typeDefault = o.Long.fromNumber(this.typeDefault, this.type === "uint64" || this.type === "fixed64"), Object.freeze && Object.freeze(this.typeDefault);
    else if (this.bytes && typeof this.typeDefault == "string") {
      var n;
      o.base64.test(this.typeDefault) ? o.base64.decode(this.typeDefault, n = o.newBuffer(o.base64.length(this.typeDefault)), 0) : o.utf8.write(this.typeDefault, n = o.newBuffer(o.utf8.length(this.typeDefault)), 0), this.typeDefault = n;
    }
    return this.map ? this.defaultValue = o.emptyObject : this.repeated ? this.defaultValue = o.emptyArray : this.defaultValue = this.typeDefault, this.parent instanceof s && (this.parent.ctor.prototype[this.name] = this.defaultValue), t.prototype.resolve.call(this);
  }, i.prototype._inferLegacyProtoFeatures = function(n) {
    if (n !== "proto2" && n !== "proto3")
      return {};
    var u = {};
    if (this.rule === "required" && (u.field_presence = "LEGACY_REQUIRED"), this.parent && r.defaults[this.type] === void 0) {
      var l = this.parent.get(this.type.split(".").pop());
      l && l instanceof s && l.group && (u.message_encoding = "DELIMITED");
    }
    return this.getOption("packed") === !0 ? u.repeated_field_encoding = "PACKED" : this.getOption("packed") === !1 && (u.repeated_field_encoding = "EXPANDED"), u;
  }, i.prototype._resolveFeatures = function(n) {
    return t.prototype._resolveFeatures.call(this, this._edition || n);
  }, i.d = function(n, u, l, c) {
    return typeof u == "function" ? u = o.decorateType(u).name : u && typeof u == "object" && (u = o.decorateEnum(u).name), function(p, g) {
      o.decorateType(p.constructor).add(new i(g, n, u, l, { default: c }));
    };
  }, i._configure = function(n) {
    s = n;
  }, vt;
}
var wt, Ar;
function Le() {
  if (Ar) return wt;
  Ar = 1, wt = o;
  var t = Se();
  ((o.prototype = Object.create(t.prototype)).constructor = o).className = "OneOf";
  var e = Ne(), r = te();
  function o(a, i, f, n) {
    if (Array.isArray(i) || (f = i, i = void 0), t.call(this, a, f), !(i === void 0 || Array.isArray(i)))
      throw TypeError("fieldNames must be an Array");
    this.oneof = i || [], this.fieldsArray = [], this.comment = n;
  }
  o.fromJSON = function(i, f) {
    return new o(i, f.oneof, f.options, f.comment);
  }, o.prototype.toJSON = function(i) {
    var f = i ? !!i.keepComments : !1;
    return r.toObject([
      "options",
      this.options,
      "oneof",
      this.oneof,
      "comment",
      f ? this.comment : void 0
    ]);
  };
  function s(a) {
    if (a.parent)
      for (var i = 0; i < a.fieldsArray.length; ++i)
        a.fieldsArray[i].parent || a.parent.add(a.fieldsArray[i]);
  }
  return o.prototype.add = function(i) {
    if (!(i instanceof e))
      throw TypeError("field must be a Field");
    return i.parent && i.parent !== this.parent && i.parent.remove(i), this.oneof.push(i.name), this.fieldsArray.push(i), i.partOf = this, s(this), this;
  }, o.prototype.remove = function(i) {
    if (!(i instanceof e))
      throw TypeError("field must be a Field");
    var f = this.fieldsArray.indexOf(i);
    if (f < 0)
      throw Error(i + " is not a member of " + this);
    return this.fieldsArray.splice(f, 1), f = this.oneof.indexOf(i.name), f > -1 && this.oneof.splice(f, 1), i.partOf = null, this;
  }, o.prototype.onAdd = function(i) {
    t.prototype.onAdd.call(this, i);
    for (var f = this, n = 0; n < this.oneof.length; ++n) {
      var u = i.get(this.oneof[n]);
      u && !u.partOf && (u.partOf = f, f.fieldsArray.push(u));
    }
    s(this);
  }, o.prototype.onRemove = function(i) {
    for (var f = 0, n; f < this.fieldsArray.length; ++f)
      (n = this.fieldsArray[f]).parent && n.parent.remove(n);
    t.prototype.onRemove.call(this, i);
  }, Object.defineProperty(o.prototype, "isProto3Optional", {
    get: function() {
      if (this.fieldsArray == null || this.fieldsArray.length !== 1)
        return !1;
      var a = this.fieldsArray[0];
      return a.options != null && a.options.proto3_optional === !0;
    }
  }), o.d = function() {
    for (var i = new Array(arguments.length), f = 0; f < arguments.length; )
      i[f] = arguments[f++];
    return function(u, l) {
      r.decorateType(u.constructor).add(new o(l, i)), Object.defineProperty(u, l, {
        get: r.oneOfGetter(i),
        set: r.oneOfSetter(i)
      });
    };
  }, wt;
}
var _t, xr;
function Se() {
  if (xr) return _t;
  xr = 1, _t = i, i.className = "ReflectionObject";
  const t = Le();
  var e = te(), r, o = { enum_type: "OPEN", field_presence: "EXPLICIT", json_format: "ALLOW", message_encoding: "LENGTH_PREFIXED", repeated_field_encoding: "PACKED", utf8_validation: "VERIFY" }, s = { enum_type: "CLOSED", field_presence: "EXPLICIT", json_format: "LEGACY_BEST_EFFORT", message_encoding: "LENGTH_PREFIXED", repeated_field_encoding: "EXPANDED", utf8_validation: "NONE" }, a = { enum_type: "OPEN", field_presence: "IMPLICIT", json_format: "ALLOW", message_encoding: "LENGTH_PREFIXED", repeated_field_encoding: "PACKED", utf8_validation: "VERIFY" };
  function i(f, n) {
    if (!e.isString(f))
      throw TypeError("name must be a string");
    if (n && !e.isObject(n))
      throw TypeError("options must be an object");
    this.options = n, this.parsedOptions = null, this.name = f, this._edition = null, this._defaultEdition = "proto2", this._features = {}, this._featuresResolved = !1, this.parent = null, this.resolved = !1, this.comment = null, this.filename = null;
  }
  return Object.defineProperties(i.prototype, {
    /**
     * Reference to the root namespace.
     * @name ReflectionObject#root
     * @type {Root}
     * @readonly
     */
    root: {
      get: function() {
        for (var f = this; f.parent !== null; )
          f = f.parent;
        return f;
      }
    },
    /**
     * Full name including leading dot.
     * @name ReflectionObject#fullName
     * @type {string}
     * @readonly
     */
    fullName: {
      get: function() {
        for (var f = [this.name], n = this.parent; n; )
          f.unshift(n.name), n = n.parent;
        return f.join(".");
      }
    }
  }), i.prototype.toJSON = /* istanbul ignore next */
  function() {
    throw Error();
  }, i.prototype.onAdd = function(n) {
    this.parent && this.parent !== n && this.parent.remove(this), this.parent = n, this.resolved = !1;
    var u = n.root;
    u instanceof r && u._handleAdd(this);
  }, i.prototype.onRemove = function(n) {
    var u = n.root;
    u instanceof r && u._handleRemove(this), this.parent = null, this.resolved = !1;
  }, i.prototype.resolve = function() {
    return this.resolved ? this : (this.root instanceof r && (this.resolved = !0), this);
  }, i.prototype._resolveFeaturesRecursive = function(n) {
    return this._resolveFeatures(this._edition || n);
  }, i.prototype._resolveFeatures = function(n) {
    if (!this._featuresResolved) {
      var u = {};
      if (!n)
        throw new Error("Unknown edition for " + this.fullName);
      var l = e.merge(
        {},
        this.options && this.options.features,
        this._inferLegacyProtoFeatures(n)
      );
      if (this._edition) {
        if (n === "proto2")
          u = Object.assign({}, s);
        else if (n === "proto3")
          u = Object.assign({}, a);
        else if (n === "2023")
          u = Object.assign({}, o);
        else
          throw new Error("Unknown edition: " + n);
        this._features = e.merge(u, l), this._featuresResolved = !0;
        return;
      }
      if (this.partOf instanceof t) {
        var c = e.merge({}, this.partOf._features);
        this._features = e.merge(c, l);
      } else if (!this.declaringField) if (this.parent) {
        var h = e.merge({}, this.parent._features);
        this._features = e.merge(h, l);
      } else
        throw new Error("Unable to find a parent for " + this.fullName);
      this.extensionField && (this.extensionField._features = this._features), this._featuresResolved = !0;
    }
  }, i.prototype._inferLegacyProtoFeatures = function() {
    return {};
  }, i.prototype.getOption = function(n) {
    if (this.options)
      return this.options[n];
  }, i.prototype.setOption = function(n, u, l) {
    return n === "__proto__" ? this : (this.options || (this.options = {}), /^features\./.test(n) ? e.setProperty(this.options, n, u, l) : (!l || this.options[n] === void 0) && (this.getOption(n) !== u && (this.resolved = !1), this.options[n] = u), this);
  }, i.prototype.setParsedOption = function(n, u, l) {
    if (n === "__proto__")
      return this;
    this.parsedOptions || (this.parsedOptions = []);
    var c = this.parsedOptions;
    if (l) {
      var h = c.find(function(v) {
        return Object.prototype.hasOwnProperty.call(v, n);
      });
      if (h) {
        var p = h[n];
        e.setProperty(p, l, u);
      } else
        h = {}, h[n] = e.setProperty({}, l, u), c.push(h);
    } else {
      var g = {};
      g[n] = u, c.push(g);
    }
    return this;
  }, i.prototype.setOptions = function(n, u) {
    if (n)
      for (var l = Object.keys(n), c = 0; c < l.length; ++c)
        this.setOption(l[c], n[l[c]], u);
    return this;
  }, i.prototype.toString = function() {
    var n = this.constructor.className, u = this.fullName;
    return u.length ? n + " " + u : n;
  }, i.prototype._editionToJSON = function() {
    if (!(!this._edition || this._edition === "proto3"))
      return this._edition;
  }, i._configure = function(f) {
    r = f;
  }, _t;
}
var bt, Tr;
function ye() {
  if (Tr) return bt;
  Tr = 1, bt = o;
  var t = Se();
  ((o.prototype = Object.create(t.prototype)).constructor = o).className = "Enum";
  var e = Ge(), r = te();
  function o(s, a, i, f, n, u) {
    if (t.call(this, s, i), a && typeof a != "object")
      throw TypeError("values must be an object");
    if (this.valuesById = {}, this.values = Object.create(this.valuesById), this.comment = f, this.comments = n || {}, this.valuesOptions = u, this._valuesFeatures = {}, this.reserved = void 0, a)
      for (var l = Object.keys(a), c = 0; c < l.length; ++c)
        l[c] !== "__proto__" && typeof a[l[c]] == "number" && (this.valuesById[this.values[l[c]] = a[l[c]]] = l[c]);
  }
  return o.prototype._resolveFeatures = function(a) {
    return a = this._edition || a, t.prototype._resolveFeatures.call(this, a), Object.keys(this.values).forEach((i) => {
      var f = r.merge({}, this._features);
      this._valuesFeatures[i] = r.merge(f, this.valuesOptions && this.valuesOptions[i] && this.valuesOptions[i].features || {});
    }), this;
  }, o.fromJSON = function(a, i) {
    var f = new o(a, i.values, i.options, i.comment, i.comments);
    return f.reserved = i.reserved, i.edition && (f._edition = i.edition), f._defaultEdition = "proto3", f;
  }, o.prototype.toJSON = function(a) {
    var i = a ? !!a.keepComments : !1;
    return r.toObject([
      "edition",
      this._editionToJSON(),
      "options",
      this.options,
      "valuesOptions",
      this.valuesOptions,
      "values",
      this.values,
      "reserved",
      this.reserved && this.reserved.length ? this.reserved : void 0,
      "comment",
      i ? this.comment : void 0,
      "comments",
      i ? this.comments : void 0
    ]);
  }, o.prototype.add = function(a, i, f, n) {
    if (!r.isString(a))
      throw TypeError("name must be a string");
    if (!r.isInteger(i))
      throw TypeError("id must be an integer");
    if (a === "__proto__")
      return this;
    if (this.values[a] !== void 0)
      throw Error("duplicate name '" + a + "' in " + this);
    if (this.isReservedId(i))
      throw Error("id " + i + " is reserved in " + this);
    if (this.isReservedName(a))
      throw Error("name '" + a + "' is reserved in " + this);
    if (this.valuesById[i] !== void 0) {
      if (!(this.options && this.options.allow_alias))
        throw Error("duplicate id " + i + " in " + this);
      this.values[a] = i;
    } else
      this.valuesById[this.values[a] = i] = a;
    return n && (this.valuesOptions === void 0 && (this.valuesOptions = {}), this.valuesOptions[a] = n || null), this.comments[a] = f || null, this;
  }, o.prototype.remove = function(a) {
    if (!r.isString(a))
      throw TypeError("name must be a string");
    var i = this.values[a];
    if (i == null)
      throw Error("name '" + a + "' does not exist in " + this);
    return delete this.valuesById[i], delete this.values[a], delete this.comments[a], this.valuesOptions && delete this.valuesOptions[a], this;
  }, o.prototype.isReservedId = function(a) {
    return e.isReservedId(this.reserved, a);
  }, o.prototype.isReservedName = function(a) {
    return e.isReservedName(this.reserved, a);
  }, bt;
}
var Ot, Fr;
function pn() {
  if (Fr) return Ot;
  Fr = 1, Ot = s;
  var t = ye(), e = Re(), r = te();
  function o(a, i, f, n) {
    return i.delimited ? a("types[%i].encode(%s,w.uint32(%i),q+1).uint32(%i)", f, n, (i.id << 3 | 3) >>> 0, (i.id << 3 | 4) >>> 0) : a("types[%i].encode(%s,w.uint32(%i).fork(),q+1).ldelim()", f, n, (i.id << 3 | 2) >>> 0);
  }
  function s(a) {
    for (var i = r.codegen(["m", "w", "q"], a.name + "$encode")("if(!w)")("w=Writer.create()")("if(q===undefined)q=0")("if(q>util.recursionLimit)")('throw Error("max depth exceeded")'), f, n, u = (
      /* initializes */
      a.fieldsArray.slice().sort(r.compareFieldsById)
    ), f = 0; f < u.length; ++f) {
      var l = u[f].resolve(), c = a._fieldsArray.indexOf(l), h = l.resolvedType instanceof t ? "int32" : l.type, p = e.basic[h];
      n = "m" + r.safeProp(l.name), l.map ? (i("if(%s!=null&&Object.hasOwnProperty.call(m,%j)){", n, l.name)("for(var ks=Object.keys(%s),i=0;i<ks.length;++i){", n)("w.uint32(%i).fork().uint32(%i).%s(ks[i])", (l.id << 3 | 2) >>> 0, 8 | e.mapKey[l.keyType], l.keyType), p === void 0 ? i("types[%i].encode(%s[ks[i]],w.uint32(18).fork(),q+1).ldelim().ldelim()", c, n) : i(".uint32(%i).%s(%s[ks[i]]).ldelim()", 16 | p, h, n), i("}")("}")) : l.repeated ? (i("if(%s!=null&&%s.length){", n, n), l.packed && e.packed[h] !== void 0 ? i("w.uint32(%i).fork()", (l.id << 3 | 2) >>> 0)("for(var i=0;i<%s.length;++i)", n)("w.%s(%s[i])", h, n)("w.ldelim()") : (i("for(var i=0;i<%s.length;++i)", n), p === void 0 ? o(i, l, c, n + "[i]") : i("w.uint32(%i).%s(%s[i])", (l.id << 3 | p) >>> 0, h, n)), i("}")) : (l.optional && i("if(%s!=null&&Object.hasOwnProperty.call(m,%j))", n, l.name), p === void 0 ? o(i, l, c, n) : i("w.uint32(%i).%s(%s)", (l.id << 3 | p) >>> 0, h, n));
    }
    return i("return w");
  }
  return Ot;
}
var z = Xr.exports = Qr;
z.build = "light";
function ci(t, e, r) {
  return typeof e == "function" ? (r = e, e = new z.Root()) : e || (e = new z.Root()), e.load(t, r);
}
z.load = ci;
function hi(t, e) {
  return e || (e = new z.Root()), e.loadSync(t);
}
z.loadSync = hi;
z.encoder = pn();
z.decoder = cn();
z.verifier = hn();
z.converter = dn();
z.ReflectionObject = Se();
z.Namespace = Ge();
z.Root = Xt();
z.Enum = ye();
z.Type = Zt();
z.Field = Ne();
z.OneOf = Le();
z.MapField = Vt();
z.Service = Gt();
z.Method = Jt();
z.Message = Wt;
z.wrappers = Ht;
z.types = Re();
z.util = te();
z.ReflectionObject._configure(z.Root);
z.Namespace._configure(z.Type, z.Service, z.Enum);
z.Root._configure(z.Type);
z.Field._configure(z.Type);
var di = Xr.exports, mn = yn, Et = /[\s{}=;:[\],'"()<>]/g, pi = /(?:"([^"\\]*(?:\\.[^"\\]*)*)")/g, mi = /(?:'([^'\\]*(?:\\.[^'\\]*)*)')/g, gi = /^ *[*/]+ */, yi = /^\s*\*?\/*/, vi = /\n/g, wi = /\s/, _i = /\\(.?)/g, bi = {
  0: "\0",
  r: "\r",
  n: `
`,
  t: "	"
};
function gn(t) {
  return t.replace(_i, function(e, r) {
    switch (r) {
      case "\\":
      case "":
        return r;
      default:
        return bi[r] || "";
    }
  });
}
yn.unescape = gn;
function yn(t, e) {
  t = t.toString();
  var r = 0, o = t.length, s = 1, a = 0, i = {}, f = [], n = null;
  function u(y) {
    return Error("illegal " + y + " (line " + s + ")");
  }
  function l() {
    var y = n === "'" ? mi : pi;
    y.lastIndex = r - 1;
    var b = y.exec(t);
    if (!b)
      throw u("string");
    return r = y.lastIndex, T(n), n = null, gn(b[1]);
  }
  function c(y) {
    return t.charAt(y);
  }
  function h(y, b, E) {
    var R = {
      type: t.charAt(y++),
      lineEmpty: !1,
      leading: E
    }, U;
    e ? U = 2 : U = 3;
    var A = y - U, L;
    do
      if (--A < 0 || (L = t.charAt(A)) === `
`) {
        R.lineEmpty = !0;
        break;
      }
    while (L === " " || L === "	");
    for (var B = t.substring(y, b).split(vi), $ = 0; $ < B.length; ++$)
      B[$] = B[$].replace(e ? yi : gi, "").trim();
    R.text = B.join(`
`).trim(), i[s] = R, a = s;
  }
  function p(y) {
    var b = g(y), E = t.substring(y, b), R = /^\s*\/\//.test(E);
    return R;
  }
  function g(y) {
    for (var b = y; b < o && c(b) !== `
`; )
      b++;
    return b;
  }
  function v() {
    if (f.length > 0)
      return f.shift();
    if (n)
      return l();
    var y, b, E, R, U, A = r === 0;
    do {
      if (r === o)
        return null;
      for (y = !1; wi.test(E = c(r)); )
        if (E === `
` && (A = !0, ++s), ++r === o)
          return null;
      if (c(r) === "/") {
        if (++r === o)
          throw u("comment");
        if (c(r) === "/")
          if (e) {
            if (R = r, U = !1, p(r - 1)) {
              U = !0;
              do
                if (r = g(r), r === o || (r++, !A))
                  break;
              while (p(r));
            } else
              r = Math.min(o, g(r) + 1);
            U && (h(R, r, A), A = !0), s++, y = !0;
          } else {
            for (U = c(R = r + 1) === "/"; c(++r) !== `
`; )
              if (r === o)
                return null;
            ++r, U && (h(R, r - 1, A), A = !0), ++s, y = !0;
          }
        else if ((E = c(r)) === "*") {
          R = r + 1, U = e || c(R) === "*";
          do {
            if (E === `
` && ++s, ++r === o)
              throw u("comment");
            b = E, E = c(r);
          } while (b !== "*" || E !== "/");
          ++r, U && (h(R, r - 2, A), A = !0), y = !0;
        } else
          return "/";
      }
    } while (y);
    var L = r;
    Et.lastIndex = 0;
    var B = Et.test(c(L++));
    if (!B)
      for (; L < o && !Et.test(c(L)); )
        ++L;
    var $ = t.substring(r, r = L);
    return ($ === '"' || $ === "'") && (n = $), $;
  }
  function T(y) {
    f.push(y);
  }
  function N() {
    if (!f.length) {
      var y = v();
      if (y === null)
        return null;
      T(y);
    }
    return f[0];
  }
  function m(y, b) {
    var E = N(), R = E === y;
    if (R)
      return v(), !0;
    if (!b)
      throw u("token '" + E + "', '" + y + "' expected");
    return !1;
  }
  function w(y) {
    var b = null, E;
    return y === void 0 ? (E = i[s - 1], delete i[s - 1], E && (e || E.type === "*" || E.lineEmpty) && (b = E.leading ? E.text : null)) : (a < y && N(), E = i[y], delete i[y], E && !E.lineEmpty && (e || E.type === "/") && (b = E.leading ? null : E.text)), b;
  }
  return Object.defineProperty({
    next: v,
    peek: N,
    push: T,
    skip: m,
    cmnt: w
  }, "line", {
    get: function() {
      return s;
    }
  });
}
var Oi = _e;
_e.filename = null;
_e.defaults = { keepCase: !1 };
var Ei = mn, Ir = Xt(), Rr = Zt(), Lr = Ne(), ki = Vt(), Pr = Le(), Ni = ye(), Si = Gt(), Ai = Jt(), xi = Se(), Ti = Re(), we = te(), Fi = /^[1-9][0-9]*$/, Ii = /^-?[1-9][0-9]*$/, Ri = /^0[x][0-9a-fA-F]+$/, Li = /^-?0[x][0-9a-fA-F]+$/, Pi = /^0[0-7]+$/, Bi = /^-?0[0-7]+$/, Di = we.patterns.numberRe, ce = /^[a-zA-Z_][a-zA-Z_0-9]*$/, he = we.patterns.typeRefRe;
function _e(t, e, r) {
  e instanceof Ir || (r = e, e = new Ir()), r || (r = _e.defaults);
  var o = r.preferTrailingComment || !1, s = Ei(t, r.alternateCommentMode || !1), a = s.next, i = s.push, f = s.peek, n = s.skip, u = s.cmnt, l = !0, c, h, p, g = "proto2", v = e, T = [], N = {}, m = r.keepCase ? function(S) {
    return S;
  } : we.camelCase;
  function w() {
    T.forEach((S) => {
      S._edition = g, Object.keys(N).forEach((k) => {
        S.getOption(k) === void 0 && S.setOption(k, N[k], !0);
      });
    });
  }
  function y(S, k, x) {
    var F = _e.filename;
    return x || (_e.filename = null), Error("illegal " + (k || "token") + " '" + S + "' (" + (F ? F + ", " : "") + "line " + s.line + ")");
  }
  function b() {
    var S = [], k;
    do {
      if ((k = a()) !== '"' && k !== "'")
        throw y(k);
      S.push(a()), n(k), k = f();
    } while (k === '"' || k === "'");
    return S.join("");
  }
  function E(S) {
    var k = a();
    switch (k) {
      case "'":
      case '"':
        return i(k), b();
      case "true":
      case "TRUE":
        return !0;
      case "false":
      case "FALSE":
        return !1;
    }
    try {
      return U(
        k,
        /* insideTryCatch */
        !0
      );
    } catch {
      if (he.test(k))
        return k;
      throw y(k, "value");
    }
  }
  function R(S, k) {
    var x, F;
    do
      if (k && ((x = f()) === '"' || x === "'")) {
        var M = b();
        if (S.push(M), g >= 2023)
          throw y(M, "id");
      } else
        try {
          S.push([F = A(a()), n("to", !0) ? A(a()) : F]);
        } catch (q) {
          if (k && he.test(x) && g >= 2023)
            S.push(x);
          else
            throw q;
        }
    while (n(",", !0));
    var P = { options: void 0 };
    P.setOption = function(q, K) {
      this.options === void 0 && (this.options = {}), this.options[q] = K;
    }, d(
      P,
      function(K) {
        if (K === "option")
          J(P, K), n(";");
        else
          throw y(K);
      },
      function() {
        Y(P);
      }
    );
  }
  function U(S, k) {
    var x = 1;
    switch (S.charAt(0) === "-" && (x = -1, S = S.substring(1)), S) {
      case "inf":
      case "INF":
      case "Inf":
        return x * (1 / 0);
      case "nan":
      case "NAN":
      case "Nan":
      case "NaN":
        return NaN;
      case "0":
        return 0;
    }
    if (Fi.test(S))
      return x * parseInt(S, 10);
    if (Ri.test(S))
      return x * parseInt(S, 16);
    if (Pi.test(S))
      return x * parseInt(S, 8);
    if (Di.test(S))
      return x * parseFloat(S);
    throw y(S, "number", k);
  }
  function A(S, k) {
    switch (S) {
      case "max":
      case "MAX":
      case "Max":
        return 536870911;
      case "0":
        return 0;
    }
    if (!k && S.charAt(0) === "-")
      throw y(S, "id");
    if (Ii.test(S))
      return parseInt(S, 10);
    if (Li.test(S))
      return parseInt(S, 16);
    if (Bi.test(S))
      return parseInt(S, 8);
    throw y(S, "id");
  }
  function L() {
    if (c !== void 0)
      throw y("package");
    if (c = a(), !he.test(c))
      throw y(c, "name");
    v = v.define(c), n(";");
  }
  function B() {
    var S = f(), k;
    switch (S) {
      case "weak":
        k = p || (p = []), a();
        break;
      case "public":
        a();
      default:
        k = h || (h = []);
        break;
    }
    S = b(), n(";"), k.push(S);
  }
  function $() {
    if (n("="), g = b(), g < 2023)
      throw y(g, "syntax");
    n(";");
  }
  function O() {
    if (n("="), g = b(), !["2023"].includes(g))
      throw y(g, "edition");
    n(";");
  }
  function _(S, k, x) {
    switch (x === void 0 && (x = 0), k) {
      case "option":
        return J(S, k), n(";"), !0;
      case "message":
        return I(S, k, x + 1), !0;
      case "enum":
        return Z(S, k), !0;
      case "service":
        return Ae(S, k, x + 1), !0;
      case "extend":
        return Sn(S, k, x), !0;
    }
    return !1;
  }
  function d(S, k, x) {
    var F = s.line;
    if (S && (typeof S.comment != "string" && (S.comment = u()), S.filename = _e.filename), n("{", !0)) {
      for (var M; (M = a()) !== "}"; )
        k(M);
      n(";", !0);
    } else
      x && x(), n(";"), S && (typeof S.comment != "string" || o) && (S.comment = u(F) || S.comment);
  }
  function I(S, k, x) {
    if (x === void 0 && (x = 0), x > we.nestingLimit)
      throw Error("max depth exceeded");
    if (!ce.test(k = a()))
      throw y(k, "type name");
    var F = new Rr(k);
    d(F, function(P) {
      if (!_(F, P, x))
        switch (P) {
          case "map":
            C(F);
            break;
          case "required":
            if (g !== "proto2")
              throw y(P);
          case "repeated":
            D(F, P, void 0, x + 1);
            break;
          case "optional":
            if (g === "proto3")
              D(F, "proto3_optional", void 0, x + 1);
            else {
              if (g !== "proto2")
                throw y(P);
              D(F, "optional", void 0, x + 1);
            }
            break;
          case "oneof":
            G(F, P, x + 1);
            break;
          case "extensions":
            R(F.extensions || (F.extensions = []));
            break;
          case "reserved":
            R(F.reserved || (F.reserved = []), !0);
            break;
          default:
            if (g === "proto2" || !he.test(P))
              throw y(P);
            i(P), D(F, "optional", void 0, x + 1);
            break;
        }
    }), S.add(F), S === v && T.push(F);
  }
  function D(S, k, x, F) {
    var M = a();
    if (M === "group") {
      V(S, k, F);
      return;
    }
    for (; M.endsWith(".") || f().startsWith("."); )
      M += a();
    if (!he.test(M))
      throw y(M, "type");
    var P = a();
    if (!ce.test(P))
      throw y(P, "name");
    P = m(P), n("=");
    var q = new Lr(P, A(a()), M, k, x);
    if (d(q, function(ne) {
      if (ne === "option")
        J(q, ne), n(";");
      else
        throw y(ne);
    }, function() {
      Y(q);
    }), k === "proto3_optional") {
      var K = new Pr("_" + P);
      q.setOption("proto3_optional", !0), K.add(q), S.add(K);
    } else
      S.add(q);
    S === v && T.push(q);
  }
  function V(S, k, x) {
    if (x === void 0 && (x = 0), x > we.nestingLimit)
      throw Error("max depth exceeded");
    if (g >= 2023)
      throw y("group");
    var F = a();
    if (!ce.test(F))
      throw y(F, "name");
    var M = we.lcFirst(F);
    F === M && (F = we.ucFirst(F)), n("=");
    var P = A(a()), q = new Rr(F);
    q.group = !0;
    var K = new Lr(M, P, F, k);
    K.filename = _e.filename, d(q, function(ne) {
      switch (ne) {
        case "option":
          J(q, ne), n(";");
          break;
        case "required":
        case "repeated":
          D(q, ne, void 0, x + 1);
          break;
        case "optional":
          g === "proto3" ? D(q, "proto3_optional", void 0, x + 1) : D(q, "optional", void 0, x + 1);
          break;
        case "message":
          I(q, ne, x + 1);
          break;
        case "enum":
          Z(q, ne);
          break;
        case "reserved":
          R(q.reserved || (q.reserved = []), !0);
          break;
        default:
          throw y(ne);
      }
    }), S.add(q).add(K);
  }
  function C(S) {
    n("<");
    var k = a();
    if (Ti.mapKey[k] === void 0)
      throw y(k, "type");
    n(",");
    var x = a();
    if (!he.test(x))
      throw y(x, "type");
    n(">");
    var F = a();
    if (!ce.test(F))
      throw y(F, "name");
    n("=");
    var M = new ki(m(F), A(a()), k, x);
    d(M, function(q) {
      if (q === "option")
        J(M, q), n(";");
      else
        throw y(q);
    }, function() {
      Y(M);
    }), S.add(M);
  }
  function G(S, k, x) {
    if (!ce.test(k = a()))
      throw y(k, "name");
    var F = new Pr(m(k));
    d(F, function(P) {
      P === "option" ? (J(F, P), n(";")) : (i(P), D(F, "optional", void 0, x));
    }), S.add(F);
  }
  function Z(S, k) {
    if (!ce.test(k = a()))
      throw y(k, "name");
    var x = new Ni(k);
    d(x, function(M) {
      switch (M) {
        case "option":
          J(x, M), n(";");
          break;
        case "reserved":
          R(x.reserved || (x.reserved = []), !0), x.reserved === void 0 && (x.reserved = []);
          break;
        default:
          ie(x, M);
      }
    }), S.add(x), S === v && T.push(x);
  }
  function ie(S, k) {
    if (!ce.test(k))
      throw y(k, "name");
    n("=");
    var x = A(a(), !0), F = {
      options: void 0
    };
    F.getOption = function(M) {
      return this.options[M];
    }, F.setOption = function(M, P) {
      xi.prototype.setOption.call(F, M, P);
    }, F.setParsedOption = function() {
    }, d(F, function(P) {
      if (P === "option")
        J(F, P), n(";");
      else
        throw y(P);
    }, function() {
      Y(F);
    }), S.add(k, x, F.comment, F.parsedOptions || F.options);
  }
  function J(S, k) {
    var x, F, M = !0;
    for (k === "option" && (k = a()); k !== "="; ) {
      if (k === "(") {
        var P = a();
        n(")"), k = "(" + P + ")";
      }
      if (M) {
        if (M = !1, k.includes(".") && !k.includes("(")) {
          var q = k.split(".");
          x = q[0] + ".", k = q[1];
          continue;
        }
        x = k;
      } else
        F = F ? F += k : k;
      k = a();
    }
    var K = F ? x.concat(F) : x, le = W(S, K);
    F = F && F[0] === "." ? F.slice(1) : F, x = x && x[x.length - 1] === "." ? x.slice(0, -1) : x, X(S, x, le, F);
  }
  function W(S, k, x) {
    if (x === void 0 && (x = 0), x > we.recursionLimit)
      throw Error("max depth exceeded");
    if (n("{", !0)) {
      for (var F = {}; !n("}", !0); ) {
        if (!ce.test(ee = a()))
          throw y(ee, "name");
        if (ee === null)
          throw y(ee, "end of input");
        var M, P = ee;
        if (n(":", !0), f() === "{")
          M = W(S, k + "." + ee, x + 1);
        else if (f() === "[") {
          M = [];
          var q;
          if (n("[", !0)) {
            do
              q = E(), M.push(q);
            while (n(",", !0));
            n("]"), typeof q < "u" && re(S, k + "." + ee, q);
          }
        } else
          M = E(), re(S, k + "." + ee, M);
        var K = F[P];
        K && (M = [].concat(K).concat(M)), P !== "__proto__" && (F[P] = M), n(",", !0), n(";", !0);
      }
      return F;
    }
    var le = E();
    return re(S, k, le), le;
  }
  function re(S, k, x) {
    if (v === S && /^features\./.test(k)) {
      N[k] = x;
      return;
    }
    S.setOption && S.setOption(k, x);
  }
  function X(S, k, x, F) {
    S.setParsedOption && S.setParsedOption(k, x, F);
  }
  function Y(S) {
    if (n("[", !0)) {
      do
        J(S, "option");
      while (n(",", !0));
      n("]");
    }
    return S;
  }
  function Ae(S, k, x) {
    if (x === void 0 && (x = 0), x > we.recursionLimit)
      throw Error("max depth exceeded");
    if (!ce.test(k = a()))
      throw y(k, "service name");
    var F = new Si(k);
    d(F, function(P) {
      if (!_(F, P, x))
        if (P === "rpc")
          Nn(F, P);
        else
          throw y(P);
    }), S.add(F), S === v && T.push(F);
  }
  function Nn(S, k) {
    var x = u(), F = k;
    if (!ce.test(k = a()))
      throw y(k, "name");
    var M = k, P, q, K, le;
    if (n("("), n("stream", !0) && (q = !0), !he.test(k = a()) || (P = k, n(")"), n("returns"), n("("), n("stream", !0) && (le = !0), !he.test(k = a())))
      throw y(k);
    K = k, n(")");
    var ne = new Ai(M, F, P, K, q, le);
    ne.comment = x, d(ne, function(rt) {
      if (rt === "option")
        J(ne, rt), n(";");
      else
        throw y(rt);
    }), S.add(ne);
  }
  function Sn(S, k, x) {
    if (!he.test(k = a()))
      throw y(k, "reference");
    var F = k;
    d(null, function(P) {
      switch (P) {
        case "required":
        case "repeated":
          D(S, P, F, x + 1);
          break;
        case "optional":
          g === "proto3" ? D(S, "proto3_optional", F, x + 1) : D(S, "optional", F, x + 1);
          break;
        default:
          if (g === "proto2" || !he.test(P))
            throw y(P);
          i(P), D(S, "optional", F, x + 1);
          break;
      }
    });
  }
  for (var ee; (ee = a()) !== null; )
    switch (ee) {
      case "package":
        if (!l)
          throw y(ee);
        L();
        break;
      case "import":
        if (!l)
          throw y(ee);
        B();
        break;
      case "syntax":
        if (!l)
          throw y(ee);
        $();
        break;
      case "edition":
        if (!l)
          throw y(ee);
        O();
        break;
      case "option":
        J(v, ee), n(";", !0);
        break;
      default:
        if (_(v, ee, 0)) {
          l = !1;
          continue;
        }
        throw y(ee);
    }
  return w(), _e.filename = null, {
    package: c,
    imports: h,
    weakImports: p,
    root: e
  };
}
var $i = fe, Mi = /\/|\./;
function fe(t, e) {
  Mi.test(t) || (t = "google/protobuf/" + t + ".proto", e = { nested: { google: { nested: { protobuf: { nested: e } } } } }), fe[t] = e;
}
fe("any", {
  /**
   * Properties of a google.protobuf.Any message.
   * @interface IAny
   * @type {Object}
   * @property {string} [typeUrl]
   * @property {Uint8Array} [bytes]
   * @memberof common
   */
  Any: {
    fields: {
      type_url: {
        type: "string",
        id: 1
      },
      value: {
        type: "bytes",
        id: 2
      }
    }
  }
});
var vn;
fe("duration", {
  /**
   * Properties of a google.protobuf.Duration message.
   * @interface IDuration
   * @type {Object}
   * @property {number|Long} [seconds]
   * @property {number} [nanos]
   * @memberof common
   */
  Duration: vn = {
    fields: {
      seconds: {
        type: "int64",
        id: 1
      },
      nanos: {
        type: "int32",
        id: 2
      }
    }
  }
});
fe("timestamp", {
  /**
   * Properties of a google.protobuf.Timestamp message.
   * @interface ITimestamp
   * @type {Object}
   * @property {number|Long} [seconds]
   * @property {number} [nanos]
   * @memberof common
   */
  Timestamp: vn
});
fe("empty", {
  /**
   * Properties of a google.protobuf.Empty message.
   * @interface IEmpty
   * @memberof common
   */
  Empty: {
    fields: {}
  }
});
fe("struct", {
  /**
   * Properties of a google.protobuf.Struct message.
   * @interface IStruct
   * @type {Object}
   * @property {Object.<string,IValue>} [fields]
   * @memberof common
   */
  Struct: {
    fields: {
      fields: {
        keyType: "string",
        type: "Value",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.Value message.
   * @interface IValue
   * @type {Object}
   * @property {string} [kind]
   * @property {0} [nullValue]
   * @property {number} [numberValue]
   * @property {string} [stringValue]
   * @property {boolean} [boolValue]
   * @property {IStruct} [structValue]
   * @property {IListValue} [listValue]
   * @memberof common
   */
  Value: {
    oneofs: {
      kind: {
        oneof: [
          "nullValue",
          "numberValue",
          "stringValue",
          "boolValue",
          "structValue",
          "listValue"
        ]
      }
    },
    fields: {
      nullValue: {
        type: "NullValue",
        id: 1
      },
      numberValue: {
        type: "double",
        id: 2
      },
      stringValue: {
        type: "string",
        id: 3
      },
      boolValue: {
        type: "bool",
        id: 4
      },
      structValue: {
        type: "Struct",
        id: 5
      },
      listValue: {
        type: "ListValue",
        id: 6
      }
    }
  },
  NullValue: {
    values: {
      NULL_VALUE: 0
    }
  },
  /**
   * Properties of a google.protobuf.ListValue message.
   * @interface IListValue
   * @type {Object}
   * @property {Array.<IValue>} [values]
   * @memberof common
   */
  ListValue: {
    fields: {
      values: {
        rule: "repeated",
        type: "Value",
        id: 1
      }
    }
  }
});
fe("wrappers", {
  /**
   * Properties of a google.protobuf.DoubleValue message.
   * @interface IDoubleValue
   * @type {Object}
   * @property {number} [value]
   * @memberof common
   */
  DoubleValue: {
    fields: {
      value: {
        type: "double",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.FloatValue message.
   * @interface IFloatValue
   * @type {Object}
   * @property {number} [value]
   * @memberof common
   */
  FloatValue: {
    fields: {
      value: {
        type: "float",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.Int64Value message.
   * @interface IInt64Value
   * @type {Object}
   * @property {number|Long} [value]
   * @memberof common
   */
  Int64Value: {
    fields: {
      value: {
        type: "int64",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.UInt64Value message.
   * @interface IUInt64Value
   * @type {Object}
   * @property {number|Long} [value]
   * @memberof common
   */
  UInt64Value: {
    fields: {
      value: {
        type: "uint64",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.Int32Value message.
   * @interface IInt32Value
   * @type {Object}
   * @property {number} [value]
   * @memberof common
   */
  Int32Value: {
    fields: {
      value: {
        type: "int32",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.UInt32Value message.
   * @interface IUInt32Value
   * @type {Object}
   * @property {number} [value]
   * @memberof common
   */
  UInt32Value: {
    fields: {
      value: {
        type: "uint32",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.BoolValue message.
   * @interface IBoolValue
   * @type {Object}
   * @property {boolean} [value]
   * @memberof common
   */
  BoolValue: {
    fields: {
      value: {
        type: "bool",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.StringValue message.
   * @interface IStringValue
   * @type {Object}
   * @property {string} [value]
   * @memberof common
   */
  StringValue: {
    fields: {
      value: {
        type: "string",
        id: 1
      }
    }
  },
  /**
   * Properties of a google.protobuf.BytesValue message.
   * @interface IBytesValue
   * @type {Object}
   * @property {Uint8Array} [value]
   * @memberof common
   */
  BytesValue: {
    fields: {
      value: {
        type: "bytes",
        id: 1
      }
    }
  }
});
fe("field_mask", {
  /**
   * Properties of a google.protobuf.FieldMask message.
   * @interface IDoubleValue
   * @type {Object}
   * @property {number} [value]
   * @memberof common
   */
  FieldMask: {
    fields: {
      paths: {
        rule: "repeated",
        type: "string",
        id: 1
      }
    }
  }
});
fe.get = function(e) {
  return fe[e] || null;
};
var Ee = Zr.exports = di;
Ee.build = "full";
Ee.tokenize = mn;
Ee.parse = Oi;
Ee.common = $i;
Ee.Root._configure(Ee.Type, Ee.parse, Ee.common);
var Ci = Zr.exports, Ui = Ci, qi = /* @__PURE__ */ Mn(Ui);
const zi = `
syntax = "proto3";
package onnx;

message StringStringEntryProto {
  string key = 1;
  string value = 2;
}

message TensorShapeProto {
  message Dimension {
    int64 dim_value = 1;
    string dim_param = 2;
    string denotation = 3;
  }
  repeated Dimension dim = 1;
}

message TypeProto {
  message Tensor {
    int32 elem_type = 1;
    TensorShapeProto shape = 2;
  }
  Tensor tensor_type = 1;
}

message TensorProto {
  enum DataType {
    UNDEFINED = 0;
    FLOAT = 1;
    UINT8 = 2;
    INT8 = 3;
    UINT16 = 4;
    INT16 = 5;
    INT32 = 6;
    INT64 = 7;
    STRING = 8;
    BOOL = 9;
    FLOAT16 = 10;
    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;
    COMPLEX128 = 15;
    BFLOAT16 = 16;
  }
  enum DataLocation {
    DEFAULT = 0;
    EXTERNAL = 1;
  }
  message Segment {
    int64 begin = 1;
    int64 end = 2;
  }
  repeated int64 dims = 1;
  int32 data_type = 2;
  Segment segment = 3;
  repeated float float_data = 4 [packed = true];
  repeated int32 int32_data = 5 [packed = true];
  repeated bytes string_data = 6;
  repeated int64 int64_data = 7 [packed = true];
  string name = 8;
  bytes raw_data = 9;
  repeated double double_data = 10 [packed = true];
  repeated uint64 uint64_data = 11 [packed = true];
  string doc_string = 12;
  repeated StringStringEntryProto external_data = 13;
  DataLocation data_location = 14;
}

message SparseTensorProto {
  TensorProto values = 1;
  TensorProto indices = 2;
  repeated int64 dims = 3;
}

message AttributeProto {
  enum AttributeType {
    UNDEFINED = 0;
    FLOAT = 1;
    INT = 2;
    STRING = 3;
    TENSOR = 4;
    GRAPH = 5;
    FLOATS = 6;
    INTS = 7;
    STRINGS = 8;
    TENSORS = 9;
    GRAPHS = 10;
    SPARSE_TENSOR = 11;
    SPARSE_TENSORS = 12;
    TYPE_PROTO = 13;
    TYPE_PROTOS = 14;
  }
  string name = 1;
  float f = 2;
  int64 i = 3;
  bytes s = 4;
  TensorProto t = 5;
  GraphProto g = 6;
  repeated float floats = 7;
  repeated int64 ints = 8;
  repeated bytes strings = 9;
  repeated TensorProto tensors = 10;
  repeated GraphProto graphs = 11;
  string doc_string = 13;
  TypeProto tp = 14;
  repeated TypeProto type_protos = 15;
  AttributeType type = 20;
  string ref_attr_name = 21;
  SparseTensorProto sparse_tensor = 22;
  repeated SparseTensorProto sparse_tensors = 23;
}

message ValueInfoProto {
  string name = 1;
  TypeProto type = 2;
  string doc_string = 3;
}

message NodeProto {
  repeated string input = 1;
  repeated string output = 2;
  string name = 3;
  string op_type = 4;
  repeated AttributeProto attribute = 5;
  string doc_string = 6;
  string domain = 7;
}

message GraphProto {
  repeated NodeProto node = 1;
  string name = 2;
  repeated TensorProto initializer = 5;
  string doc_string = 10;
  repeated ValueInfoProto input = 11;
  repeated ValueInfoProto output = 12;
  repeated ValueInfoProto value_info = 13;
  repeated SparseTensorProto sparse_initializer = 15;
}

message OperatorSetIdProto {
  string domain = 1;
  int64 version = 2;
}

message ModelProto {
  int64 ir_version = 1;
  string producer_name = 2;
  string producer_version = 3;
  string domain = 4;
  int64 model_version = 5;
  string doc_string = 6;
  GraphProto graph = 7;
  repeated OperatorSetIdProto opset_import = 8;
  repeated StringStringEntryProto metadata_props = 14;
}
`;
let kt = null;
function wn() {
  return kt || (kt = qi.parse(zi, { keepCase: !0 }).root.lookupType("onnx.ModelProto")), kt;
}
const Rt = 1e5, ji = 1e-9;
function Vi(t) {
  const e = (t & 32768) >> 15, r = (t & 31744) >> 10, o = t & 1023;
  let s;
  return r === 0 ? s = o * Math.pow(2, -24) : r === 31 ? s = o ? NaN : 1 / 0 : s = (1 + o / 1024) * Math.pow(2, r - 15), e ? -s : s;
}
function Ji(t) {
  const e = t.data_type ?? 0, r = t.float_data && t.float_data.length && t.float_data || t.double_data && t.double_data.length && t.double_data || t.int32_data && t.int32_data.length && t.int32_data || null;
  if (r) return Br(r.map(Number));
  if (t.int64_data && t.int64_data.length) return Br(Ie(t.int64_data));
  const o = t.raw_data;
  if (!o || o.length === 0) return null;
  const s = Hr[e];
  if (!s) return null;
  const a = Math.floor(o.length / s), i = Math.max(1, Math.floor(a / Rt)), f = new DataView(o.buffer, o.byteOffset, o.byteLength), n = [];
  for (let u = 0; u < a; u += i) {
    const l = u * s;
    switch (e) {
      case 1:
        n.push(f.getFloat32(l, !0));
        break;
      case 11:
        n.push(f.getFloat64(l, !0));
        break;
      case 10:
        n.push(Vi(f.getUint16(l, !0)));
        break;
      case 6:
        n.push(f.getInt32(l, !0));
        break;
      case 3:
        n.push(f.getInt8(l));
        break;
      case 2:
        n.push(f.getUint8(l));
        break;
      case 5:
        n.push(f.getInt16(l, !0));
        break;
      case 7:
        n.push(Number(f.getBigInt64(l, !0)));
        break;
      default:
        return null;
    }
  }
  return n;
}
function Br(t) {
  if (t.length <= Rt) return t;
  const e = Math.ceil(t.length / Rt), r = [];
  for (let o = 0; o < t.length; o += e) r.push(t[o]);
  return r;
}
const De = (t) => Math.round(t * 1e4) / 1e4;
function Gi(t) {
  const e = Ie(t.dims), r = e.reduce((l, c) => l * c, e.length ? 1 : 0), o = Ji(t);
  if (!o || o.length === 0) return { shape: e, size: r };
  let s = 1 / 0, a = -1 / 0, i = 0, f = 0;
  for (const l of o)
    l < s && (s = l), l > a && (a = l), i += l, Math.abs(l) < ji && f++;
  const n = i / o.length;
  let u = 0;
  for (const l of o) u += (l - n) * (l - n);
  return u /= o.length, {
    shape: e,
    size: r,
    min: De(s),
    max: De(a),
    mean: De(n),
    std: De(Math.sqrt(u)),
    sparse: De(f / o.length)
  };
}
function Wi(t) {
  const e = Ie(t.dims);
  return e.reduce((r, o) => r * o, e.length ? 1 : 0);
}
function Hi(t, e = {}) {
  const r = t instanceof Uint8Array ? t : new Uint8Array(t), o = wn().decode(r), s = o.graph;
  if (!s) throw new Error("ONNX model has no graph.");
  const a = /* @__PURE__ */ new Map();
  for (const m of s.initializer ?? [])
    m.name && a.set(m.name, m);
  const i = In(s), f = /* @__PURE__ */ new Map();
  for (const m of s.initializer ?? [])
    m.name && f.set(m.name, xt[m.data_type ?? 1] ?? "float32");
  const n = (m = s.input ?? []) => {
    var w, y;
    for (const b of m) {
      const E = (y = (w = b.type) == null ? void 0 : w.tensor_type) == null ? void 0 : y.elem_type;
      b.name && E != null && f.set(b.name, xt[E] ?? "float32");
    }
  };
  n(s.input), n(s.value_info), n(s.output);
  const u = new Set((s.output ?? []).map((m) => m.name).filter(Boolean)), l = [], c = /* @__PURE__ */ new Map();
  let h = 0;
  for (const m of s.input ?? []) {
    if (!m.name || a.has(m.name)) continue;
    const w = i.get(m.name), y = f.get(m.name) ?? "float32", b = h++;
    c.set(m.name, b), l.push({
      id: b,
      name: m.name,
      type: "Tensor",
      op: "Input",
      shape: tr(w),
      dt: y,
      params: 0,
      flops: 0,
      macs: 0,
      mem: Dr(w, y),
      group: "input",
      ins: [],
      outs: [{ n: m.name, s: w }],
      attr: {},
      w: null,
      math: "model input",
      insight: "Graph input tensor. Apply preprocessing (resize/normalize) upstream.",
      qSens: 0.1,
      compIssues: []
    });
  }
  for (const m of s.node ?? []) {
    const w = m.op_type ?? "Unknown", y = Tn(m), b = m.input ?? [], E = m.output ?? [], R = b.map((C) => i.get(C)), U = E[0];
    let A = U ? i.get(U) : void 0;
    if (!A) {
      const C = Ln(w, y, R);
      C && (A = C, U && i.set(U, C));
    }
    let L = 0, B = null, $ = 0;
    for (const C of b) {
      const G = a.get(C);
      if (!G) continue;
      const Z = Wi(G);
      L += Z, Z > $ && ($ = Z, B = G);
    }
    const { flops: O, macs: _ } = Pn(w, y, R, A), d = $n(w), I = U && f.get(U) || b[0] && f.get(b[0]) || "float32", D = h++;
    for (const C of E) C && c.set(C, D);
    const V = B ? Gi(B) : null;
    l.push({
      id: D,
      name: m.name || U || `${w}_${D}`,
      type: w,
      op: w,
      shape: tr(A),
      dt: I,
      params: L,
      flops: O,
      macs: _,
      mem: Dr(A, I),
      group: U && u.has(U) ? "output" : d.group,
      ins: b.filter((C) => !a.has(C)).map((C) => ({ n: C, s: i.get(C) })),
      outs: E.map((C) => ({ n: C, s: i.get(C) })),
      attr: y,
      w: V,
      math: d.math,
      insight: d.insight,
      qSens: d.qSens,
      compIssues: []
    });
  }
  const p = /* @__PURE__ */ new Set(), g = [];
  for (const m of s.node ?? []) {
    const w = c.get((m.output ?? [])[0] ?? "");
    if (w != null)
      for (const y of m.input ?? []) {
        if (a.has(y)) continue;
        const b = c.get(y);
        if (b == null || b === w) continue;
        const E = `${b}->${w}`;
        p.has(E) || (p.add(E), g.push([b, w]));
      }
  }
  const v = (o.opset_import ?? []).find((m) => !m.domain || m.domain === "ai.onnx"), T = (s.input ?? []).find((m) => m.name && !a.has(m.name)), N = (s.output ?? [])[0];
  return {
    name: e.name || s.name || o.producer_name || "model",
    format: "ONNX",
    framework: o.producer_name || "unknown",
    opset: v ? qe(v.version) : 0,
    sizeBytes: r.byteLength,
    inputShape: T ? $r(i.get(T.name)) : [],
    outputShape: N ? $r(i.get(N.name)) : [],
    layers: l,
    edges: g,
    producer: o.producer_name || void 0,
    irVersion: o.ir_version != null ? qe(o.ir_version) : void 0
  };
}
function Dr(t, e) {
  var s;
  if (!t) return 0;
  const r = (s = Object.entries(xt).find(([, a]) => a === e)) == null ? void 0 : s[0], o = r ? Hr[Number(r)] ?? 4 : 4;
  return ze(t) * o;
}
function $r(t) {
  return (t ?? []).map((e) => e < 0 ? 0 : e);
}
class Qt {
  constructor(e, r = !0) {
    Pe(this, "view");
    Pe(this, "off", 0);
    this.buf = e, this.le = r, this.view = new DataView(e.buffer, e.byteOffset, e.byteLength);
  }
  get length() {
    return this.buf.byteLength;
  }
  get eof() {
    return this.off >= this.buf.byteLength;
  }
  seek(e) {
    this.off = e;
  }
  u8() {
    const e = this.view.getUint8(this.off);
    return this.off += 1, e;
  }
  u16() {
    const e = this.view.getUint16(this.off, this.le);
    return this.off += 2, e;
  }
  i16() {
    const e = this.view.getInt16(this.off, this.le);
    return this.off += 2, e;
  }
  u32() {
    const e = this.view.getUint32(this.off, this.le);
    return this.off += 4, e;
  }
  i32() {
    const e = this.view.getInt32(this.off, this.le);
    return this.off += 4, e;
  }
  u64() {
    const e = this.view.getBigUint64(this.off, this.le);
    return this.off += 8, Number(e);
  }
  i64() {
    const e = this.view.getBigInt64(this.off, this.le);
    return this.off += 8, Number(e);
  }
  f32() {
    const e = this.view.getFloat32(this.off, this.le);
    return this.off += 4, e;
  }
  f64() {
    const e = this.view.getFloat64(this.off, this.le);
    return this.off += 8, e;
  }
  bytes(e) {
    const r = this.buf.subarray(this.off, this.off + e);
    return this.off += e, r;
  }
  str(e) {
    return new TextDecoder().decode(this.bytes(e));
  }
}
function Zi(t) {
  const e = t.toLowerCase();
  return /embed|embd|patch_embed|tok_embeddings|wte|wpe/.test(e) ? "input" : /lm_head|classifier|logits|\.head\.|output|fc_out|score/.test(e) ? "output" : /norm|ln_f|layernorm|bn|batch_norm/.test(e) ? "neck" : "backbone";
}
function Ve(t, e) {
  var i, f;
  const r = e.map((n, u) => {
    const l = n.shape.reduce((c, h) => c * h, (n.shape.length, 1));
    return {
      id: u,
      name: n.name,
      type: "Tensor",
      op: "Weight",
      shape: n.shape.length ? n.shape.join("×") : "scalar",
      dt: n.dtype,
      params: l,
      flops: 0,
      macs: 0,
      mem: n.bytes,
      group: Zi(n.name),
      ins: [],
      outs: [{ n: n.name, s: n.shape }],
      attr: {},
      w: { shape: n.shape, size: l },
      math: `${n.dtype}${n.shape.length ? " " + n.shape.join("×") : ""}`,
      insight: t.note || "Stored weight tensor (file order; not a compute edge).",
      qSens: 0.3,
      compIssues: []
    };
  }), o = [];
  for (let n = 1; n < r.length; n++) o.push([n - 1, n]);
  const s = ((i = e[0]) == null ? void 0 : i.shape) ?? [], a = ((f = e[e.length - 1]) == null ? void 0 : f.shape) ?? [];
  return {
    name: t.name,
    format: t.format,
    framework: t.framework,
    opset: 0,
    sizeBytes: t.sizeBytes,
    inputShape: s,
    outputShape: a,
    layers: r,
    edges: o
  };
}
const Xi = {
  F64: "float64",
  F32: "float32",
  F16: "float16",
  BF16: "bfloat16",
  I64: "int64",
  I32: "int32",
  I16: "int16",
  I8: "int8",
  U64: "uint64",
  U32: "uint32",
  U16: "uint16",
  U8: "uint8",
  BOOL: "bool",
  F8_E4M3: "float8_e4m3",
  F8_E5M2: "float8_e5m2"
};
function Qi(t, e = "model.safetensors") {
  const r = t instanceof Uint8Array ? t : new Uint8Array(t), o = new Qt(r), s = o.u64();
  if (s <= 0 || s + 8 > r.byteLength) throw new Error("Invalid safetensors header length.");
  const a = JSON.parse(o.str(s)), i = [];
  for (const [f, n] of Object.entries(a)) {
    if (f === "__metadata__") continue;
    const u = n;
    !Array.isArray(u.shape) || !Array.isArray(u.data_offsets) || i.push({
      name: f,
      shape: u.shape,
      dtype: Xi[u.dtype] ?? u.dtype,
      bytes: u.data_offsets[1] - u.data_offsets[0]
    });
  }
  return i.sort((f, n) => f.name.localeCompare(n.name)), Ve(
    { name: e, format: "Safetensors", framework: "safetensors", sizeBytes: r.byteLength },
    i
  );
}
const Ki = 101010256, Yi = 33639248;
function Ye(t) {
  return t.length > 4 && t[0] === 80 && t[1] === 75 && t[2] === 3 && t[3] === 4;
}
function et(t) {
  const e = new DataView(t.buffer, t.byteOffset, t.byteLength);
  let r = -1;
  const o = Math.max(0, t.length - 22 - 65535);
  for (let f = t.length - 22; f >= o; f--)
    if (e.getUint32(f, !0) === Ki) {
      r = f;
      break;
    }
  if (r < 0) throw new Error("Not a valid ZIP (no EOCD record).");
  const s = e.getUint16(r + 10, !0);
  let a = e.getUint32(r + 16, !0);
  const i = [];
  for (let f = 0; f < s && e.getUint32(a, !0) === Yi; f++) {
    const n = e.getUint16(a + 10, !0), u = e.getUint32(a + 20, !0), l = e.getUint32(a + 24, !0), c = e.getUint16(a + 28, !0), h = e.getUint16(a + 30, !0), p = e.getUint16(a + 32, !0), g = e.getUint32(a + 42, !0), v = new TextDecoder().decode(t.subarray(a + 46, a + 46 + c)), T = e.getUint16(g + 26, !0), N = e.getUint16(g + 28, !0), m = g + 30 + T + N;
    i.push({ name: v, method: n, compressedSize: u, size: l, dataOffset: m }), a += 46 + c + h + p;
  }
  return i;
}
function Kt(t, e) {
  return e.method !== 0 ? null : t.subarray(e.dataOffset, e.dataOffset + e.size);
}
const es = [147, 78, 85, 77, 80, 89], ts = {
  f2: 2,
  f4: 4,
  f8: 8,
  i1: 1,
  i2: 2,
  i4: 4,
  i8: 8,
  u1: 1,
  u2: 2,
  u4: 4,
  u8: 8,
  b1: 1,
  c8: 8,
  c16: 16
}, rs = {
  f2: "float16",
  f4: "float32",
  f8: "float64",
  i1: "int8",
  i2: "int16",
  i4: "int32",
  i8: "int64",
  u1: "uint8",
  u2: "uint16",
  u4: "uint32",
  u8: "uint64",
  b1: "bool",
  c8: "complex64",
  c16: "complex128"
};
function Mr(t) {
  return es.every((e, r) => t[r] === e);
}
function Cr(t) {
  const e = new DataView(t.buffer, t.byteOffset, t.byteLength), r = t[6];
  let o, s;
  r <= 1 ? (o = e.getUint16(8, !0), s = 10 + o) : (o = e.getUint32(8, !0), s = 12 + o);
  const a = new TextDecoder().decode(t.subarray(r <= 1 ? 10 : 12, s)), i = a.match(/'descr'\s*:\s*'([^']+)'/), f = a.match(/'shape'\s*:\s*\(([^)]*)\)/), n = i ? i[1] : "|u1", u = n.replace(/^[<>|=]/, ""), l = f ? f[1].split(",").map((h) => h.trim()).filter(Boolean).map(Number) : [], c = l.reduce((h, p) => h * p, (l.length, 1));
  return { dtype: rs[u] ?? n, shape: l, bytes: c * (ts[u] ?? 1) };
}
function ns(t, e = "array.npy") {
  const r = t instanceof Uint8Array ? t : new Uint8Array(t), o = [];
  if (Ye(r)) {
    for (const a of et(r)) {
      if (!a.name.endsWith(".npy")) continue;
      const i = Kt(r, a), f = a.name.replace(/\.npy$/, "");
      if (i && Mr(i)) {
        const n = Cr(i);
        o.push({ name: f, shape: n.shape, dtype: n.dtype, bytes: n.bytes });
      } else
        o.push({ name: f, shape: [], dtype: "compressed", bytes: a.size });
    }
    return Ve(
      { name: e, format: "NumPy (.npz)", framework: "numpy", sizeBytes: r.byteLength },
      o
    );
  }
  if (!Mr(r)) throw new Error("Not a NumPy .npy/.npz file.");
  const s = Cr(r);
  return o.push({ name: e.replace(/\.npy$/, ""), shape: s.shape, dtype: s.dtype, bytes: s.bytes }), Ve(
    { name: e, format: "NumPy (.npy)", framework: "numpy", sizeBytes: r.byteLength },
    o
  );
}
const is = {
  0: { name: "F32", bpe: 4 },
  1: { name: "F16", bpe: 2 },
  2: { name: "Q4_0", bpe: 0.5 },
  3: { name: "Q4_1", bpe: 0.5 },
  6: { name: "Q5_0", bpe: 0.625 },
  7: { name: "Q5_1", bpe: 0.625 },
  8: { name: "Q8_0", bpe: 1 },
  9: { name: "Q8_1", bpe: 1 },
  10: { name: "Q2_K", bpe: 0.34 },
  11: { name: "Q3_K", bpe: 0.43 },
  12: { name: "Q4_K", bpe: 0.56 },
  13: { name: "Q5_K", bpe: 0.69 },
  14: { name: "Q6_K", bpe: 0.82 },
  15: { name: "Q8_K", bpe: 1.06 },
  24: { name: "I8", bpe: 1 },
  25: { name: "I16", bpe: 2 },
  26: { name: "I32", bpe: 4 },
  28: { name: "I64", bpe: 8 },
  29: { name: "F64", bpe: 8 },
  30: { name: "BF16", bpe: 2 }
};
function Lt(t) {
  const e = t.u64();
  return t.str(e);
}
function _n(t, e) {
  switch (e) {
    case 0:
      return t.u8();
    case 1:
      return t.view.getInt8(t.off++);
    case 2:
      return t.u16();
    case 3:
      return t.i16();
    case 4:
      return t.u32();
    case 5:
      return t.i32();
    case 6:
      return t.f32();
    case 7:
      return t.u8() !== 0;
    case 8:
      return Lt(t);
    case 9: {
      const r = t.u32(), o = t.u64(), s = [];
      for (let a = 0; a < o; a++) s.push(_n(t, r));
      return s;
    }
    case 10:
      return t.u64();
    case 11:
      return t.i64();
    case 12:
      return t.f64();
    default:
      throw new Error(`Unknown GGUF metadata value type ${e}`);
  }
}
function ss(t) {
  const e = new Qt(t);
  if (e.str(4) !== "GGUF") throw new Error("Not a GGUF file (bad magic).");
  const r = e.u32(), o = e.u64(), s = e.u64(), a = /* @__PURE__ */ new Map();
  for (let f = 0; f < s; f++) {
    const n = Lt(e), u = e.u32();
    a.set(n, _n(e, u));
  }
  const i = [];
  for (let f = 0; f < o; f++) {
    const n = Lt(e), u = e.u32(), l = [];
    for (let g = 0; g < u; g++) l.push(e.u64());
    const c = e.u32();
    e.u64();
    const h = is[c] ?? { name: `ggml_${c}`, bpe: 4 }, p = l.reduce((g, v) => g * v, (l.length, 1));
    i.push({ name: n, shape: l, dtype: h.name, bytes: Math.round(p * h.bpe) });
  }
  return { version: r, meta: a, tensors: i };
}
function os(t, e = "model.gguf") {
  const r = t instanceof Uint8Array ? t : new Uint8Array(t), { version: o, meta: s, tensors: a } = ss(r), i = String(s.get("general.architecture") ?? "unknown"), f = String(s.get("general.name") ?? e), n = s.get("general.file_type");
  return Ve(
    {
      name: f,
      format: "GGUF",
      framework: `GGUF v${o} · ${i}`,
      sizeBytes: r.byteLength,
      note: `GGUF ${i} model${n != null ? `, file_type ${n}` : ""}. Stored weight tensor.`
    },
    a
  );
}
function as(t) {
  const e = [];
  let r = null;
  for (const o of t.split(/\r?\n/)) {
    const s = o.replace(/[#;].*$/, "").trim();
    if (!s) continue;
    const a = s.match(/^\[(.+)\]$/);
    if (a)
      r = { type: a[1].trim(), props: {} }, e.push(r);
    else if (r) {
      const i = s.indexOf("=");
      i > 0 && (r.props[s.slice(0, i).trim()] = s.slice(i + 1).trim());
    }
  }
  return e;
}
const se = (t, e, r) => t[e] != null && !isNaN(Number(t[e])) ? Number(t[e]) : r;
function us(t) {
  return t === "net" || t === "network" ? "input" : t === "yolo" || t === "region" || t === "detection" || t === "softmax" ? "output" : t === "route" || t === "upsample" || t === "reorg" ? "neck" : t === "connected" ? "head" : "backbone";
}
function fs(t, e = "model.cfg") {
  const r = as(t);
  if (!r.length) throw new Error("Empty or invalid Darknet .cfg.");
  const o = r[0];
  let s = se(o.props, "width", 0), a = se(o.props, "height", 0), i = se(o.props, "channels", 3);
  const f = [], n = [], u = [], l = [];
  f.push({
    id: 0,
    name: "input",
    type: "Input",
    op: "Placeholder",
    shape: `1×${i}×${a}×${s}`,
    dt: "float32",
    params: 0,
    flops: 0,
    macs: 0,
    mem: i * s * a * 4,
    group: "input",
    ins: [],
    outs: [{ n: "input", s: [1, i, a, s] }],
    attr: { width: s, height: a, channels: i },
    w: null,
    math: `Input 1×${i}×${a}×${s}`,
    insight: "Darknet network input.",
    qSens: 0,
    compIssues: []
  });
  const c = r.slice(1);
  let h = 1;
  c.forEach((g, v) => {
    const T = g.props, N = h++;
    n[v] = N;
    const m = v > 0 ? l[v - 1] : { c: i, w: s, h: a };
    let w = { ...m }, y = 0;
    const b = [];
    if (g.type === "convolutional") {
      const E = se(T, "filters", m.c), R = se(T, "size", 1), U = se(T, "stride", 1), A = se(T, "pad", 0) ? Math.floor(R / 2) : se(T, "padding", 0), L = se(T, "batch_normalize", 0);
      w = {
        c: E,
        w: Math.floor((m.w + 2 * A - R) / U) + 1,
        h: Math.floor((m.h + 2 * A - R) / U) + 1
      }, y = E * m.c * R * R + E + (L ? 2 * E : 0), b.push(v > 0 ? n[v - 1] : 0);
    } else if (g.type === "maxpool" || g.type === "avgpool") {
      const E = se(T, "size", g.type === "avgpool" ? m.w : 2), R = se(T, "stride", g.type === "avgpool" ? 1 : E);
      w = g.type === "avgpool" ? { c: m.c, w: 1, h: 1 } : { c: m.c, w: Math.floor(m.w / R), h: Math.floor(m.h / R) }, b.push(v > 0 ? n[v - 1] : 0);
    } else if (g.type === "upsample") {
      const E = se(T, "stride", 2);
      w = { c: m.c, w: m.w * E, h: m.h * E }, b.push(v > 0 ? n[v - 1] : 0);
    } else if (g.type === "route") {
      const E = (T.layers || "").split(",").map((A) => Number(A.trim())).filter((A) => !isNaN(A));
      let R = 0, U = m;
      E.forEach((A, L) => {
        const B = A < 0 ? v + A : A;
        B >= 0 && B < l.length && (R += l[B].c, L === 0 && (U = l[B]), b.push(n[B] ?? 0));
      }), w = { c: R || m.c, w: U.w, h: U.h };
    } else if (g.type === "shortcut") {
      const E = se(T, "from", -3), R = E < 0 ? v + E : E;
      b.push(v > 0 ? n[v - 1] : 0), R >= 0 && R < l.length && b.push(n[R]), w = { ...m };
    } else if (g.type === "connected") {
      const E = se(T, "output", m.c);
      y = m.c * m.w * m.h * E + E, w = { c: E, w: 1, h: 1 }, b.push(v > 0 ? n[v - 1] : 0);
    } else
      b.push(v > 0 ? n[v - 1] : 0);
    l[v] = w;
    for (const E of b) u.push([E, N]);
    f.push({
      id: N,
      name: `${g.type}_${v}`,
      type: bn(g.type),
      op: g.type,
      shape: `1×${w.c}×${w.h}×${w.w}`,
      dt: "float32",
      params: y,
      flops: 0,
      macs: 0,
      mem: w.c * w.w * w.h * 4,
      group: us(g.type),
      ins: b.map((E) => {
        var R;
        return { n: ((R = f[E]) == null ? void 0 : R.name) ?? `#${E}` };
      }),
      outs: [{ n: `${g.type}_${v}`, s: [1, w.c, w.h, w.w] }],
      attr: { ...T },
      w: y > 0 ? { shape: "[multi]", size: y } : null,
      math: ls(g),
      insight: cs(g.type),
      qSens: g.type === "convolutional" ? 0.5 : 0.1,
      compIssues: []
    });
  });
  const p = l[l.length - 1] ?? { c: i, w: s, h: a };
  return {
    name: e,
    format: "Darknet",
    framework: "Darknet (YOLO)",
    opset: 0,
    sizeBytes: t.length,
    inputShape: [1, i, a, s],
    outputShape: [1, p.c, p.h, p.w],
    layers: f,
    edges: u
  };
}
const bn = (t) => t.charAt(0).toUpperCase() + t.slice(1);
function ls(t) {
  return t.type === "convolutional" ? `Conv ${t.props.filters ?? "?"}×${t.props.size ?? "?"}, stride ${t.props.stride ?? 1}, act ${t.props.activation ?? "linear"}` : t.type === "route" ? `Route layers=${t.props.layers ?? ""}` : t.type === "shortcut" ? `Shortcut from=${t.props.from ?? ""}` : bn(t.type);
}
function cs(t) {
  switch (t) {
    case "convolutional":
      return "Darknet convolution (+ optional BN + activation).";
    case "route":
      return "Concatenates/selects earlier feature maps (FPN-style).";
    case "shortcut":
      return "Residual add from an earlier layer.";
    case "yolo":
      return "YOLO detection head — decodes boxes at this scale.";
    case "upsample":
      return "Nearest upsample for multi-scale fusion.";
    default:
      return `Darknet ${t} layer.`;
  }
}
class hs {
  constructor(e) {
    Pe(this, "view");
    this.buf = e, this.view = new DataView(e.buffer, e.byteOffset, e.byteLength);
  }
  u8(e) {
    return this.view.getUint8(e);
  }
  i8(e) {
    return this.view.getInt8(e);
  }
  u32(e) {
    return this.view.getUint32(e, !0);
  }
  i32(e) {
    return this.view.getInt32(e, !0);
  }
  root() {
    return this.u32(0);
  }
  /** Absolute position of a table field, or null if absent. */
  field(e, r) {
    const o = e - this.i32(e), s = this.view.getUint16(o, !0), a = 4 + r * 2;
    if (a >= s) return null;
    const i = this.view.getUint16(o + a, !0);
    return i === 0 ? null : e + i;
  }
  indirect(e) {
    return e + this.u32(e);
  }
  // sub-table/string/vector
  string(e, r) {
    const o = this.field(e, r);
    if (o == null) return "";
    const s = this.indirect(o), a = this.u32(s);
    return new TextDecoder().decode(this.buf.subarray(s + 4, s + 4 + a));
  }
  /** Vector position + length for an offset-typed field. */
  vec(e, r) {
    const o = this.field(e, r);
    if (o == null) return null;
    const s = this.indirect(o);
    return { base: s + 4, len: this.u32(s) };
  }
  /** Element position of an offset-vector item (resolves the uoffset). */
  vecTable(e, r) {
    const o = e + r * 4;
    return o + this.u32(o);
  }
  intVec(e, r) {
    const o = this.vec(e, r);
    if (!o) return [];
    const s = [];
    for (let a = 0; a < o.len; a++) s.push(this.i32(o.base + a * 4));
    return s;
  }
  scalarByte(e, r, o = 0) {
    const s = this.field(e, r);
    return s == null ? o : this.u8(s);
  }
  scalarI32(e, r, o = 0) {
    const s = this.field(e, r);
    return s == null ? o : this.i32(s);
  }
}
const ds = {
  0: "float32",
  1: "float16",
  2: "int32",
  3: "uint8",
  4: "int64",
  5: "string",
  6: "bool",
  7: "int16",
  8: "complex64",
  9: "int8",
  10: "float64",
  11: "complex128",
  12: "uint64",
  15: "uint32",
  16: "uint16",
  17: "int4"
}, ps = {
  0: "Add",
  1: "AveragePool2D",
  2: "Concatenation",
  3: "Conv2D",
  4: "DepthwiseConv2D",
  6: "Dequantize",
  9: "FullyConnected",
  14: "Logistic",
  17: "MaxPool2D",
  18: "Mul",
  19: "Relu",
  21: "Relu6",
  22: "Reshape",
  23: "ResizeBilinear",
  25: "Softmax",
  28: "Tanh",
  34: "Pad",
  40: "Mean",
  41: "Sub",
  49: "Split",
  56: "ArgMax",
  83: "Pack",
  88: "Slice",
  97: "ResizeNearestNeighbor",
  99: "LeakyRelu",
  102: "Unpack",
  114: "Quantize",
  117: "HardSwish",
  124: "Transpose"
};
function ms(t, e) {
  return e ? "output" : /Conv|Depthwise/.test(t) ? "backbone" : /FullyConnected|Softmax|ArgMax/.test(t) ? "head" : /Pool|Concat|Reshape|Resize|Pad|Transpose/.test(t) ? "neck" : "backbone";
}
function gs(t, e = "model.tflite") {
  const r = t instanceof Uint8Array ? t : new Uint8Array(t), o = new hs(r), s = o.root(), a = [], i = o.vec(s, 1);
  if (i)
    for (let A = 0; A < i.len; A++) {
      const L = o.vecTable(i.base, A), B = o.scalarByte(L, 0, 0), $ = o.scalarI32(L, 3, 0), O = Math.max(B, $), _ = o.string(L, 1);
      a.push(O === 0 && _ ? _ : ps[O] ?? `OP_${O}`);
    }
  const f = o.vec(s, 2);
  if (!f || f.len === 0) throw new Error("TFLite model has no subgraphs.");
  const n = o.vecTable(f.base, 0), u = o.vec(n, 0), l = [], c = [], h = [];
  if (u)
    for (let A = 0; A < u.len; A++) {
      const L = o.vecTable(u.base, A);
      c.push(o.intVec(L, 0)), h.push(ds[o.scalarByte(L, 1, 0)] ?? "float32"), l.push(o.string(L, 3) || `tensor_${A}`);
    }
  const p = o.intVec(n, 1), g = o.intVec(n, 2), v = new Set(g), T = [], N = /* @__PURE__ */ new Map();
  let m = 0;
  for (const A of p) {
    const L = m++;
    N.set(A, L), T.push(Ur(L, l[A], "Input", "Input", c[A], h[A], "input", [], [A], l, c));
  }
  const w = o.vec(n, 3), y = [], b = [];
  if (w)
    for (let A = 0; A < w.len; A++) {
      const L = o.vecTable(w.base, A), B = o.scalarI32(L, 0, 0);
      b.push({ opName: a[B] ?? `OP_${B}`, ins: o.intVec(L, 1), outs: o.intVec(L, 2) });
    }
  for (const A of b) {
    const L = m++, B = A.outs[0], $ = A.outs.some((O) => v.has(O));
    for (const O of A.outs) N.set(O, L);
    T.push(Ur(L, l[B] ?? A.opName, A.opName, A.opName, c[B] ?? [], h[B] ?? "float32", ms(A.opName, $), A.ins, A.outs, l, c));
  }
  const E = /* @__PURE__ */ new Set();
  for (let A = 0; A < T.length; A++) {
    const L = T[A];
    for (const B of L.ins) {
      const $ = B.ti;
      if ($ == null) continue;
      const O = N.get($);
      if (O == null || O === L.id) continue;
      const _ = `${O}->${L.id}`;
      E.has(_) || (E.add(_), y.push([O, L.id]));
    }
  }
  const R = p[0], U = g[0];
  return {
    name: e,
    format: "TFLite",
    framework: "TensorFlow Lite",
    opset: 3,
    sizeBytes: r.byteLength,
    inputShape: R != null ? c[R] ?? [] : [],
    outputShape: U != null ? c[U] ?? [] : [],
    layers: T,
    edges: y
  };
}
function Ur(t, e, r, o, s, a, i, f, n, u, l) {
  const c = s.reduce((h, p) => h * p, s.length ? 1 : 0);
  return {
    id: t,
    name: e,
    type: r,
    op: o,
    shape: s.length ? s.join("×") : "?",
    dt: a,
    params: 0,
    flops: 0,
    macs: 0,
    mem: c * 4,
    group: i,
    // Carry the source tensor index on each input so edges can be resolved.
    ins: f.map((h) => ({ n: u[h] ?? `#${h}`, s: l[h], ti: h })),
    outs: n.map((h) => ({ n: u[h] ?? `#${h}`, s: l[h] })),
    attr: {},
    w: null,
    math: o,
    insight: `TFLite ${o} operator.`,
    qSens: /Conv|FullyConnected/.test(o) ? 0.5 : 0.15,
    compIssues: []
  };
}
const qr = Symbol("mark");
class pe {
  constructor(e) {
    Pe(this, "attrs", /* @__PURE__ */ new Map());
    this.cls = e;
  }
}
const Ce = (t) => typeof t == "object" && t !== null && t.__global__ === !0;
function ys(t, e) {
  const r = new DataView(t.buffer, t.byteOffset, t.byteLength);
  let o = 0;
  const s = [], a = /* @__PURE__ */ new Map(), i = () => t[o++], f = () => {
    const m = r.getUint16(o, !0);
    return o += 2, m;
  }, n = () => {
    const m = r.getInt32(o, !0);
    return o += 4, m;
  }, u = () => {
    const m = r.getUint32(o, !0);
    return o += 4, m;
  }, l = () => {
    const m = Number(r.getBigUint64(o, !0));
    return o += 8, m;
  }, c = () => {
    const m = r.getFloat64(o, !1);
    return o += 8, m;
  }, h = (m) => {
    const w = t.subarray(o, o + m);
    return o += m, w;
  }, p = (m) => new TextDecoder().decode(h(m)), g = () => {
    let m = "";
    for (; t[o] !== 10; ) m += String.fromCharCode(t[o++]);
    return o++, m;
  }, v = (m) => {
    if (m === 0) return 0;
    let w = 0;
    for (let y = 0; y < m; y++) w += t[o + y] * 2 ** (8 * y);
    return t[o + m - 1] & 128 && (w -= 2 ** (8 * m)), o += m, w;
  }, T = () => {
    const m = [];
    for (; s.length; ) {
      const w = s.pop();
      if (w === qr) break;
      m.push(w);
    }
    return m.reverse();
  }, N = () => s[s.length - 1];
  for (let m = 0; o < t.length; m++) {
    const w = t[o++];
    switch (w) {
      case 128:
        i();
        break;
      case 149:
        l();
        break;
      case 40:
        s.push(qr);
        break;
      case 46:
        return s.pop();
      case 78:
        s.push(null);
        break;
      case 136:
        s.push(!0);
        break;
      case 137:
        s.push(!1);
        break;
      case 74:
        s.push(n());
        break;
      case 75:
        s.push(i());
        break;
      case 77:
        s.push(f());
        break;
      case 138:
        s.push(v(i()));
        break;
      case 139:
        s.push(v(u()));
        break;
      case 71:
        s.push(c());
        break;
      case 88:
        s.push(p(u()));
        break;
      case 140:
        s.push(p(i()));
        break;
      case 141:
        s.push(p(l()));
        break;
      case 84:
        s.push(p(u()));
        break;
      case 85:
        s.push(p(i()));
        break;
      case 66:
        s.push(h(u()));
        break;
      case 67:
        s.push(h(i()));
        break;
      case 142:
        s.push(h(l()));
        break;
      case 41:
        s.push([]);
        break;
      case 93:
        s.push([]);
        break;
      case 125:
        s.push(/* @__PURE__ */ new Map());
        break;
      case 143:
        s.push(/* @__PURE__ */ new Set());
        break;
      case 116:
        s.push(T());
        break;
      case 133:
        s.push([s.pop()]);
        break;
      case 134: {
        const y = s.pop(), b = s.pop();
        s.push([b, y]);
        break;
      }
      case 135: {
        const y = s.pop(), b = s.pop(), E = s.pop();
        s.push([E, b, y]);
        break;
      }
      case 97: {
        const y = s.pop();
        N().push(y);
        break;
      }
      case 101: {
        const y = T();
        N().push(...y);
        break;
      }
      case 115: {
        const y = s.pop(), b = s.pop();
        N().set(b, y);
        break;
      }
      case 117: {
        const y = T(), b = N();
        for (let E = 0; E < y.length; E += 2) b.set(y[E], y[E + 1]);
        break;
      }
      case 100: {
        const y = T(), b = /* @__PURE__ */ new Map();
        for (let E = 0; E < y.length; E += 2) b.set(y[E], y[E + 1]);
        s.push(b);
        break;
      }
      case 99: {
        const y = g(), b = g();
        s.push(e.findClass(y, b));
        break;
      }
      case 147: {
        const y = s.pop(), b = s.pop();
        s.push(e.findClass(b, y));
        break;
      }
      case 82: {
        const y = s.pop(), b = s.pop();
        s.push(e.reduce(b, y));
        break;
      }
      case 129: {
        const y = s.pop(), b = s.pop();
        s.push(zr(b, y, e));
        break;
      }
      case 146: {
        s.pop();
        const y = s.pop(), b = s.pop();
        s.push(zr(b, y, e));
        break;
      }
      case 98: {
        const y = s.pop(), b = s.pop();
        s.push(vs(b, y, e));
        break;
      }
      case 81: {
        const y = s.pop();
        s.push(e.persistentLoad(y));
        break;
      }
      case 80: {
        const y = g();
        s.push(e.persistentLoad(y));
        break;
      }
      case 104:
        s.push(a.get(i()));
        break;
      case 106:
        s.push(a.get(u()));
        break;
      case 103:
        s.push(a.get(parseInt(g(), 10)));
        break;
      case 113:
        a.set(i(), N());
        break;
      case 114:
        a.set(u(), N());
        break;
      case 112:
        a.set(parseInt(g(), 10), N());
        break;
      case 148:
        a.set(a.size, N());
        break;
      case 48:
        s.pop();
        break;
      case 49:
        T();
        break;
      case 50:
        s.push(N());
        break;
      case 73: {
        const y = g();
        s.push(y === "01" ? !0 : y === "00" ? !1 : parseInt(y, 10));
        break;
      }
      case 76:
        s.push(parseInt(g().replace(/L$/, ""), 10));
        break;
      case 70:
        s.push(parseFloat(g()));
        break;
      case 83:
        s.push(g().replace(/^'|'$/g, ""));
        break;
      case 86:
        s.push(g());
        break;
      default:
        throw new Error(`Unsupported pickle opcode 0x${w.toString(16)} at ${o - 1}`);
    }
    if (m > 5e7) throw new Error("Pickle stream too long.");
  }
  return s.pop();
}
function zr(t, e, r) {
  if (Ce(t)) {
    const o = r.reduce(t, e);
    if (o !== void 0) return o;
  }
  return new pe(Ce(t) ? t : null);
}
function vs(t, e, r) {
  if (r.build) {
    const o = r.build(t, e);
    if (o !== void 0) return o;
  }
  if (t instanceof pe) {
    let o = e;
    Array.isArray(o) && o.length === 2 && (o[0] instanceof Map || o[0] === null) && (o = o[0]), o instanceof Map && (t.attrs = o);
  }
  return t;
}
const ws = {
  pt: { format: "PyTorch", framework: "PyTorch" },
  pth: { format: "PyTorch", framework: "PyTorch" },
  ckpt: { format: "PyTorch checkpoint", framework: "PyTorch" },
  bin: { format: "PyTorch bin", framework: "PyTorch/HF" },
  mlmodel: { format: "Core ML", framework: "Core ML" },
  mlpackage: { format: "Core ML package", framework: "Core ML" },
  pb: { format: "TensorFlow", framework: "TensorFlow" },
  caffemodel: { format: "Caffe", framework: "Caffe" },
  prototxt: { format: "Caffe prototxt", framework: "Caffe" },
  xml: { format: "OpenVINO IR", framework: "OpenVINO" },
  pdmodel: { format: "PaddlePaddle", framework: "Paddle" },
  rknn: { format: "RKNN", framework: "Rockchip" },
  param: { format: "ncnn", framework: "ncnn" },
  mnn: { format: "MNN", framework: "Alibaba MNN" },
  mlir: { format: "MLIR", framework: "MLIR" },
  pkl: { format: "scikit-learn (pickle)", framework: "scikit-learn" },
  joblib: { format: "joblib", framework: "scikit-learn" }
};
function _s(t) {
  var e;
  return ((e = t.split(".").pop()) == null ? void 0 : e.toLowerCase()) ?? "";
}
function $e(t, e) {
  const r = t instanceof Uint8Array ? t : new Uint8Array(t), o = _s(e), s = ws[o] ?? { format: o.toUpperCase() || "Unknown", framework: "unknown" };
  if (Ye(r)) {
    const f = et(r), n = f.some((l) => /(^|\/)(data|constants)\.pkl$/.test(l.name) || /\/data\//.test(l.name)), u = f.filter((l) => !l.name.endsWith("/")).map((l) => ({ name: l.name, shape: [], dtype: l.method === 0 ? "stored" : "deflate", bytes: l.size }));
    return Ve(
      {
        name: e,
        format: n ? "PyTorch (zip)" : `${s.format} (zip)`,
        framework: s.framework,
        sizeBytes: r.byteLength,
        note: n ? "PyTorch zip archive — tensor storages listed. Full graph needs unpickling (not yet implemented)." : "Archive members listed. Full graph parsing not yet implemented."
      },
      u
    );
  }
  const a = [...r.subarray(0, 4)].map((f) => f.toString(16).padStart(2, "0")).join(" "), i = {
    id: 0,
    name: e,
    type: s.format,
    op: "Detected",
    shape: "?",
    dt: "—",
    params: 0,
    flops: 0,
    macs: 0,
    mem: r.byteLength,
    group: "input",
    ins: [],
    outs: [],
    attr: { format: s.format, sizeBytes: r.byteLength, magic: a },
    w: null,
    math: `Magic bytes: ${a}`,
    insight: `Detected ${s.format}. Full graph parsing for this format is not yet implemented — ONNX, TFLite, Safetensors, GGUF, NumPy and Darknet are fully parsed today.`,
    qSens: 0,
    compIssues: []
  };
  return {
    name: e,
    format: s.format,
    framework: s.framework,
    opset: 0,
    sizeBytes: r.byteLength,
    inputShape: [],
    outputShape: [],
    layers: [i],
    edges: []
  };
}
const bs = {
  FloatStorage: "float32",
  HalfStorage: "float16",
  DoubleStorage: "float64",
  LongStorage: "int64",
  IntStorage: "int32",
  ShortStorage: "int16",
  CharStorage: "int8",
  ByteStorage: "uint8",
  BoolStorage: "bool",
  BFloat16Storage: "bfloat16"
}, tt = (t) => typeof t == "object" && t !== null && t.__tensor__ === !0, Ue = (t) => t.reduce((e, r) => e * r, (t.length, 1)), Nt = (t) => t instanceof Map ? t : null;
function Os() {
  return {
    persistentLoad(t) {
      const e = t;
      if (Array.isArray(e) && e[0] === "storage") {
        const r = e[1];
        return { __storage__: !0, dtype: Ce(r) ? bs[r.name] ?? "float32" : "float32" };
      }
      return { __storage__: !0, dtype: "float32" };
    },
    findClass: (t, e) => ({ __global__: !0, module: t, name: e }),
    reduce(t, e) {
      if (Ce(t)) {
        const r = t.name;
        if (/^_rebuild_tensor/.test(r)) {
          const o = e[0], s = e[2];
          return { __tensor__: !0, shape: Array.isArray(s) ? s.map(Number) : [], dtype: (o == null ? void 0 : o.dtype) ?? "float32" };
        }
        if (r === "_rebuild_parameter") {
          const o = e[0];
          return tt(o) ? { ...o, __param__: !0 } : o;
        }
        return r === "OrderedDict" ? /* @__PURE__ */ new Map() : r === "_reconstructor" || r === "__newobj__" ? new pe(Ce(e[0]) ? e[0] : null) : new pe(t);
      }
      return new pe(null);
    }
  };
}
function Es(t) {
  return /Detect|Classify|Segment|Pose|Linear|head/i.test(t) ? "head" : /Norm/.test(t) || /Pool|Upsample|Concat|Sequential|ModuleList|Identity/.test(t) ? "neck" : "backbone";
}
function jr(t) {
  const e = [];
  if (t)
    for (const [r, o] of t) tt(o) && e.push({ name: String(r), t: o });
  return e;
}
function On(t, e, r, o, s, a) {
  var g, v, T;
  const i = ((g = t.cls) == null ? void 0 : g.name) || "Module", f = a.id++, n = jr(Nt(t.attrs.get("_parameters"))), u = jr(Nt(t.attrs.get("_buffers"))), l = ((v = n.find((N) => N.name === "weight")) == null ? void 0 : v.t) ?? ((T = n[0]) == null ? void 0 : T.t) ?? null, c = n.reduce((N, m) => N + Ue(m.t.shape), 0), h = {};
  for (const N of [...n, ...u]) h[N.name] = N.t.shape.length ? N.t.shape.join("×") : "scalar";
  o.push({
    id: f,
    name: e,
    type: i,
    op: i,
    shape: l ? l.shape.length ? l.shape.join("×") : "scalar" : "",
    dt: (l == null ? void 0 : l.dtype) ?? "float32",
    params: c,
    flops: 0,
    macs: 0,
    mem: c * 4,
    group: Es(i),
    ins: [],
    outs: [],
    attr: h,
    w: l ? { shape: l.shape, size: Ue(l.shape) } : null,
    math: i,
    insight: `PyTorch ${i} module.`,
    qSens: /Conv|Linear/.test(i) ? 0.5 : 0.1,
    compIssues: []
  }), r >= 0 && s.push([r, f]);
  const p = Nt(t.attrs.get("_modules"));
  if (p)
    for (const [N, m] of p) m instanceof pe && On(m, e ? `${e}.${N}` : String(N), f, o, s, a);
}
function ks(t) {
  if (t instanceof pe && t.attrs.get("_modules")) return t;
  if (t instanceof Map) {
    for (const r of ["model", "ema", "module", "net"]) {
      const o = t.get(r);
      if (o instanceof pe && o.attrs.get("_modules")) return o;
    }
    let e = 0;
    for (const r of t.values()) tt(r) && e++;
    if (e > 0) return t;
    for (const r of t.values()) if (r instanceof pe && r.attrs.get("_modules")) return r;
  }
  return null;
}
function Ns(t, e = "model.pt") {
  var o;
  const r = t instanceof Uint8Array ? t : new Uint8Array(t);
  try {
    if (!Ye(r)) return $e(r, e);
    const s = et(r), a = s.find((c) => /(^|\/)data\.pkl$/.test(c.name)) ?? s.find((c) => c.name.endsWith(".pkl")), i = a ? Kt(r, a) : null;
    if (!i) return $e(r, e);
    const f = ys(i, Os()), n = ks(f), u = [], l = [];
    if (n instanceof pe)
      On(n, ((o = n.cls) == null ? void 0 : o.name) ?? "model", -1, u, l, { id: 0 });
    else if (n instanceof Map) {
      let c = 0;
      for (const [h, p] of n) {
        if (!tt(p)) continue;
        const g = p;
        u.push({
          id: c++,
          name: String(h),
          type: "Tensor",
          op: "Weight",
          shape: g.shape.length ? g.shape.join("×") : "scalar",
          dt: g.dtype,
          params: Ue(g.shape),
          flops: 0,
          macs: 0,
          mem: Ue(g.shape) * 4,
          group: "backbone",
          ins: [],
          outs: [{ n: String(h), s: g.shape }],
          attr: {},
          w: { shape: g.shape, size: Ue(g.shape) },
          math: "weight",
          insight: "State-dict tensor.",
          qSens: 0.3,
          compIssues: []
        });
      }
      for (let h = 1; h < u.length; h++) l.push([h - 1, h]);
    }
    return u.length === 0 ? $e(r, e) : {
      name: e,
      format: "PyTorch",
      framework: "PyTorch",
      opset: 0,
      sizeBytes: r.byteLength,
      inputShape: [],
      outputShape: [],
      layers: u,
      edges: l,
      producer: `${u.length} modules`
    };
  } catch {
    return $e(r, e);
  }
}
const Ss = [
  { id: "onnx", label: "ONNX", status: "full", exts: ["onnx"] },
  { id: "tflite", label: "TFLite", status: "full", exts: ["tflite"] },
  { id: "safetensors", label: "Safetensors", status: "full", exts: ["safetensors"] },
  { id: "gguf", label: "GGUF", status: "full", exts: ["gguf"] },
  { id: "numpy", label: "NumPy", status: "full", exts: ["npy", "npz"] },
  { id: "darknet", label: "Darknet", status: "full", exts: ["cfg"] },
  { id: "pytorch", label: "PyTorch", status: "full", exts: ["pt", "pth", "ckpt", "bin"] },
  { id: "coreml", label: "Core ML", status: "metadata", exts: ["mlmodel", "mlpackage"] },
  { id: "openvino", label: "OpenVINO", status: "metadata", exts: ["xml"] },
  { id: "tensorflow", label: "TensorFlow", status: "metadata", exts: ["pb"] },
  { id: "caffe", label: "Caffe", status: "metadata", exts: ["caffemodel", "prototxt"] },
  { id: "paddle", label: "PaddlePaddle", status: "metadata", exts: ["pdmodel"] },
  { id: "ncnn", label: "ncnn", status: "metadata", exts: ["param"] },
  { id: "rknn", label: "RKNN", status: "metadata", exts: ["rknn"] },
  { id: "mnn", label: "MNN", status: "metadata", exts: ["mnn"] },
  { id: "mlir", label: "MLIR", status: "metadata", exts: ["mlir"] },
  { id: "sklearn", label: "scikit-learn", status: "metadata", exts: ["pkl", "joblib"] }
];
function As(t) {
  var e;
  return ((e = t.split(".").pop()) == null ? void 0 : e.toLowerCase()) ?? "";
}
function Vr(t, e) {
  return e.every((r, o) => t[o] === r);
}
function xs(t, e) {
  const r = As(t);
  if (e.length >= 4) {
    if (Vr(e, [71, 71, 85, 70])) return "gguf";
    if (Vr(e, [147, 78, 85, 77])) return "numpy";
  }
  const o = Ss.find((s) => s.exts.includes(r));
  return (o == null ? void 0 : o.id) ?? "unknown";
}
function Ts(t, e) {
  const r = t instanceof Uint8Array ? t : new Uint8Array(t);
  switch (xs(e, r)) {
    case "onnx":
      return Hi(r, { name: e });
    case "tflite":
      return gs(r, e);
    case "safetensors":
      return Qi(r, e);
    case "gguf":
      return os(r, e);
    case "numpy":
      return ns(r, e);
    case "darknet":
      return fs(new TextDecoder().decode(r), e);
    case "pytorch":
      return Ns(r, e);
    default:
      return $e(r, e);
  }
}
const En = new Float32Array(1), Fs = new Uint32Array(En.buffer);
function Is(t) {
  En[0] = t;
  const e = Fs[0], r = e >>> 16 & 32768, o = e >>> 23 & 255, s = e & 8388607;
  if (o === 255) return r | (s ? 32256 : 31744);
  const a = o - 127;
  if (a > 15) return r | 31744;
  if (a >= -14) {
    const i = s >>> 13, f = s & 8191;
    let n = a + 15 << 10 | i;
    return (f > 4096 || f === 4096 && i & 1) && n++, r | n;
  }
  if (a >= -25) {
    const i = s | 8388608, f = -1 - a, n = i >>> f, u = 1 << f - 1, l = i & (1 << f) - 1;
    let c = n;
    return (l > u || l === u && n & 1) && c++, r | c;
  }
  return r;
}
function Rs(t) {
  const e = t & 32768 ? -1 : 1, r = t >> 10 & 31, o = t & 1023;
  return r === 0 ? e * o * Math.pow(2, -24) : r === 31 ? o ? NaN : e * (1 / 0) : e * (1 + o / 1024) * Math.pow(2, r - 15);
}
function Ls(t) {
  if (t.byteLength % 2 !== 0) throw new Error(`f16BytesToF32Bytes: ${t.byteLength} bytes is not a multiple of 2.`);
  const e = t.byteLength / 2, r = new DataView(t.buffer, t.byteOffset, t.byteLength), o = new Uint8Array(e * 4), s = new DataView(o.buffer);
  for (let a = 0; a < e; a++) s.setFloat32(a * 4, Rs(r.getUint16(a * 2, !0)), !0);
  return o;
}
function St(t) {
  if (t.byteLength % 4 !== 0)
    throw new Error(`f32BytesToF16Bytes: ${t.byteLength} bytes is not a multiple of 4 (corrupt float32 data?).`);
  const e = t.byteLength / 4, r = new DataView(t.buffer, t.byteOffset, t.byteLength), o = new Uint8Array(e * 2), s = new DataView(o.buffer);
  for (let a = 0; a < e; a++)
    s.setUint16(a * 2, Is(r.getFloat32(a * 4, !0)), !0);
  return o;
}
function Ps(t, e) {
  const r = {};
  e && (r.__metadata__ = e);
  let o = 0;
  for (const c of t)
    r[c.name] = { dtype: c.dtype, shape: c.shape, data_offsets: [o, o + c.data.byteLength] }, o += c.data.byteLength;
  const s = o;
  let a = JSON.stringify(r), i = new TextEncoder().encode(a);
  const f = (8 - (8 + i.length) % 8) % 8;
  f && (a += " ".repeat(f), i = new TextEncoder().encode(a));
  const n = new Uint8Array(8 + i.length + s);
  new DataView(n.buffer).setBigUint64(0, BigInt(i.length), !0), n.set(i, 8);
  let l = 8 + i.length;
  for (const c of t)
    n.set(c.data, l), l += c.data.byteLength;
  return n;
}
let At = null;
function Bs() {
  if (At) return At;
  const t = new Uint32Array(256);
  for (let e = 0; e < 256; e++) {
    let r = e;
    for (let o = 0; o < 8; o++) r = r & 1 ? 3988292384 ^ r >>> 1 : r >>> 1;
    t[e] = r >>> 0;
  }
  return At = t, t;
}
function Ds(t) {
  const e = Bs();
  let r = 4294967295;
  for (let o = 0; o < t.length; o++) r = e[(r ^ t[o]) & 255] ^ r >>> 8;
  return (r ^ 4294967295) >>> 0;
}
const $s = new TextEncoder();
function Ms(t) {
  const e = [], r = [], o = (l, c) => l.push(c & 255, c >>> 8 & 255), s = (l, c) => l.push(c & 255, c >>> 8 & 255, c >>> 16 & 255, c >>> 24 & 255), a = (l, c) => {
    for (let h = 0; h < c.length; h++) l.push(c[h]);
  }, i = [];
  for (const l of t) {
    const c = $s.encode(l.name), h = Ds(l.data), p = e.length;
    s(e, 67324752), o(e, 20), o(e, 0), o(e, 0), o(e, 0), o(e, 33), s(e, h), s(e, l.data.length), s(e, l.data.length), o(e, c.length), o(e, 0), a(e, c), a(e, l.data), i.push({ name: c, crc: h, size: l.data.length, offset: p });
  }
  const f = e.length;
  for (const l of i)
    s(r, 33639248), o(r, 20), o(r, 20), o(r, 0), o(r, 0), o(r, 0), o(r, 33), s(r, l.crc), s(r, l.size), s(r, l.size), o(r, l.name.length), o(r, 0), o(r, 0), o(r, 0), o(r, 0), s(r, 0), s(r, l.offset), a(r, l.name);
  const n = [];
  s(n, 101010256), o(n, 0), o(n, 0), o(n, i.length), o(n, i.length), s(n, r.length), s(n, f), o(n, 0);
  const u = new Uint8Array(e.length + r.length + n.length);
  return u.set(e, 0), u.set(r, e.length), u.set(n, e.length + r.length), u;
}
const kn = [147, 78, 85, 77, 80, 89], Cs = {
  F64: "<f8",
  F32: "<f4",
  F16: "<f2",
  I64: "<i8",
  I32: "<i4",
  I16: "<i2",
  I8: "|i1",
  U64: "<u8",
  U32: "<u4",
  U16: "<u2",
  U8: "|u1",
  BOOL: "|b1"
}, Us = {
  f8: "F64",
  f4: "F32",
  f2: "F16",
  i1: "I8",
  i2: "I16",
  i4: "I32",
  i8: "I64",
  u1: "U8",
  u2: "U16",
  u4: "U32",
  u8: "U64",
  b1: "BOOL"
}, qs = (t) => new TextEncoder().encode(t);
function zs(t) {
  return kn.every((e, r) => t[r] === e);
}
function js(t) {
  const e = Cs[t.dtype];
  if (!e) throw new Error(`NumPy export: dtype ${t.dtype} has no numpy equivalent.`);
  const r = t.shape.length === 0 ? "()" : t.shape.length === 1 ? `(${t.shape[0]},)` : `(${t.shape.join(", ")})`;
  let o = `{'descr': '${e}', 'fortran_order': False, 'shape': ${r}, }`;
  const s = (64 - (10 + o.length + 1) % 64) % 64;
  o = o + " ".repeat(s) + `
`;
  const a = qs(o), i = new Uint8Array(10 + a.length + t.data.byteLength);
  return i.set(kn, 0), i[6] = 1, i[7] = 0, new DataView(i.buffer).setUint16(8, a.length, !0), i.set(a, 10), i.set(t.data, 10 + a.length), i;
}
function Vs(t) {
  const e = t.map((r) => ({ name: `${r.name}.npy`, data: js(r) }));
  return Ms(e);
}
function Jr(t, e) {
  if (!zs(t)) throw new Error("Not a NumPy .npy file.");
  const r = new DataView(t.buffer, t.byteOffset, t.byteLength), o = t[6], s = o <= 1 ? r.getUint16(8, !0) : r.getUint32(8, !0), a = (o <= 1 ? 10 : 12) + s, i = new TextDecoder().decode(t.subarray(o <= 1 ? 10 : 12, a)), f = i.match(/'descr'\s*:\s*'([^']+)'/), n = i.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (/'fortran_order'\s*:\s*True/.test(i)) throw new Error(`NumPy: ${e} is Fortran-ordered (column-major) — not supported.`);
  const l = f ? f[1] : "|u1";
  if (/^>/.test(l) && !/^>[ui]1$/.test(l)) throw new Error(`NumPy: ${e} is big-endian (${l}) — not supported.`);
  const c = l.replace(/^[<>|=]/, ""), h = Us[c];
  if (!h) throw new Error(`NumPy: unsupported dtype ${l} in ${e}.`);
  const p = n ? n[1].split(",").map((g) => g.trim()).filter(Boolean).map(Number) : [];
  return { name: e, dtype: h, shape: p, data: t.subarray(a) };
}
function Js(t, e) {
  if (Ye(t)) {
    const r = [];
    let o = 0;
    for (const s of et(t)) {
      if (!s.name.endsWith(".npy")) continue;
      const a = Kt(t, s);
      if (!a) {
        o++;
        continue;
      }
      r.push(Jr(a, s.name.replace(/\.npy$/, "")));
    }
    if (r.length === 0)
      throw new Error(o > 0 ? "This .npz is compressed (np.savez_compressed) — only uncompressed np.savez is supported." : "No arrays found in this .npz.");
    return r;
  }
  return [Jr(t, e.replace(/\.npy$/, ""))];
}
const Gs = 1;
function Ws(t) {
  return BigInt(typeof t == "number" ? Math.trunc(t) : t.toString());
}
function Hs(t) {
  const e = t.data_type ?? 0, r = (o, s, a) => {
    const i = new Uint8Array(o * s), f = new DataView(i.buffer);
    for (let n = 0; n < o; n++) a(f, n * s, n);
    return i;
  };
  if (t.float_data && t.float_data.length) return r(t.float_data.length, 4, (o, s, a) => o.setFloat32(s, t.float_data[a], !0));
  if (t.double_data && t.double_data.length) return r(t.double_data.length, 8, (o, s, a) => o.setFloat64(s, t.double_data[a], !0));
  if (t.int64_data && t.int64_data.length) {
    const o = t.int64_data;
    return r(o.length, 8, (s, a, i) => s.setBigInt64(a, Ws(o[i]), !0));
  }
  if (t.int32_data && t.int32_data.length) {
    const o = t.int32_data;
    switch (e) {
      case 10:
      case 16:
        return r(o.length, 2, (s, a, i) => s.setUint16(a, o[i] & 65535, !0));
      case 4:
        return r(o.length, 2, (s, a, i) => s.setUint16(a, o[i] & 65535, !0));
      case 5:
        return r(o.length, 2, (s, a, i) => s.setInt16(a, o[i], !0));
      case 2:
      case 9:
        return r(o.length, 1, (s, a, i) => s.setUint8(a, o[i] & 255));
      case 3:
        return r(o.length, 1, (s, a, i) => s.setInt8(a, o[i]));
      default:
        return r(o.length, 4, (s, a, i) => s.setInt32(a, o[i], !0));
    }
  }
  return null;
}
function Zs(t) {
  var i;
  const e = t instanceof Uint8Array ? t : new Uint8Array(t), o = ((i = wn().decode(e).graph) == null ? void 0 : i.initializer) ?? [], s = [];
  let a = 0;
  for (const f of o) {
    if (!f.name) continue;
    if (f.data_location === Gs || f.external_data && f.external_data.length) {
      a++;
      continue;
    }
    const n = f.raw_data && f.raw_data.length ? f.raw_data.slice() : Hs(f);
    n && s.push({ name: f.name, dataType: f.data_type ?? 1, dims: Ie(f.dims), bytes: n });
  }
  return { tensors: s, skippedExternal: a };
}
const Xs = {
  1: "F32",
  2: "U8",
  3: "I8",
  4: "U16",
  5: "I16",
  6: "I32",
  7: "I64",
  9: "BOOL",
  10: "F16",
  11: "F64",
  12: "U32",
  13: "U64",
  16: "BF16"
}, Qs = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  F8_E4M3: 1,
  F8_E5M2: 1,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U64: 8,
  U32: 4,
  U16: 2,
  U8: 1,
  BOOL: 1
}, Gr = (t) => t.replace(/\.[^.]+$/, "");
function Ks(t) {
  const e = new Qt(t), r = e.u64(), o = JSON.parse(e.str(r)), s = 8 + r, a = t.byteLength - s, i = [], f = [];
  for (const [u, l] of Object.entries(o)) {
    if (u === "__metadata__") continue;
    if (!l || !Array.isArray(l.data_offsets)) throw new Error(`Safetensors: tensor ${u} has no data_offsets.`);
    const [c, h] = l.data_offsets;
    if (!Number.isInteger(c) || !Number.isInteger(h) || c < 0 || h < c || h > a)
      throw new Error(`Safetensors: tensor ${u} has invalid data_offsets [${c}, ${h}] (data block is ${a} bytes).`);
    f.push([c, h]), i.push({ name: u, dtype: l.dtype, shape: l.shape, data: t.subarray(s + c, s + h) });
  }
  f.sort((u, l) => u[0] - l[0]);
  let n = 0;
  for (const [u, l] of f) {
    if (u !== n) throw new Error(`Safetensors: tensor data is not contiguous (gap/overlap at byte ${n}).`);
    n = l;
  }
  return i;
}
function Pt(t) {
  const e = t.byteLength / 2, r = new DataView(t.buffer, t.byteOffset, t.byteLength), o = new Uint8Array(e * 4), s = new DataView(o.buffer);
  for (let a = 0; a < e; a++) s.setUint32(a * 4, r.getUint16(a * 2, !0) << 16 >>> 0, !0);
  return o;
}
function Wr(t) {
  const e = t.byteLength / 8, r = new DataView(t.buffer, t.byteOffset, t.byteLength), o = new Uint8Array(e * 4), s = new DataView(o.buffer);
  for (let a = 0; a < e; a++) s.setFloat32(a * 4, r.getFloat64(a * 8, !0), !0);
  return o;
}
function Ys(t, e) {
  return e === "keep" ? t : t.map((r) => e === "f16" ? r.dtype === "F32" ? { ...r, dtype: "F16", data: St(r.data) } : r.dtype === "F64" ? { ...r, dtype: "F16", data: St(Wr(r.data)) } : r.dtype === "BF16" ? { ...r, dtype: "F16", data: St(Pt(r.data)) } : r : r.dtype === "F16" ? { ...r, dtype: "F32", data: Ls(r.data) } : r.dtype === "BF16" ? { ...r, dtype: "F32", data: Pt(r.data) } : r.dtype === "F64" ? { ...r, dtype: "F32", data: Wr(r.data) } : r);
}
function eo(t, e, r) {
  if (/onnx/i.test(e)) {
    const { tensors: o, skippedExternal: s } = Zs(t);
    if (o.length === 0)
      throw new Error(
        s > 0 ? `All ${s} weights use external data (stored outside the file) — can't repack in-browser.` : "No weight initializers found in this ONNX model."
      );
    return o.map((a) => {
      const i = Xs[a.dataType];
      if (!i) throw new Error(`Unsupported ONNX dtype ${a.dataType} for tensor ${a.name}.`);
      return { name: a.name, dtype: i, shape: a.dims, data: a.bytes };
    });
  }
  if (/safetensors/i.test(e)) return Ks(t);
  if (/numpy/i.test(e)) return Js(t, r);
  throw new Error(`Weights export isn't supported for ${e} sources (ONNX, Safetensors, NumPy are).`);
}
function to(t) {
  for (const e of t) {
    const r = Qs[e.dtype];
    if (r === void 0) throw new Error(`Tensor ${e.name}: unsupported dtype "${e.dtype}".`);
    const o = e.shape.reduce((s, a) => s * a, (e.shape.length, 1));
    if (e.data.byteLength !== o * r)
      throw new Error(`Tensor ${e.name}: ${e.data.byteLength} bytes but shape ${e.shape.join("×")} × ${e.dtype} expects ${o * r}.`);
  }
}
function ro(t, e, r) {
  const o = t instanceof Uint8Array ? t : new Uint8Array(t), s = r.target ?? "safetensors", a = r.precision ?? "keep";
  let i = Ys(eo(o, r.format, e), a);
  to(i);
  const f = a === "f16" ? ".fp16" : a === "f32" ? ".fp32" : "";
  if (s === "npz")
    return i = i.map(
      (u) => u.dtype === "BF16" ? { ...u, dtype: "F32", data: Pt(u.data) } : u
    ), { filename: `${Gr(e)}${f}.npz`, mime: "application/octet-stream", data: Vs(i) };
  const n = Ps(i, { format: "modelvisio", source: r.format, precision: a });
  return { filename: `${Gr(e)}${f}.safetensors`, mime: "application/octet-stream", data: n };
}
const Ze = self;
Ze.onmessage = (t) => {
  const e = t.data;
  try {
    if (e.kind === "convert") {
      const r = ro(e.buffer, e.name, { format: e.format, target: e.target, precision: e.precision });
      Ze.postMessage({ id: e.id, result: r });
    } else {
      const r = Ts(e.buffer, e.name);
      Ze.postMessage({ id: e.id, model: r });
    }
  } catch (r) {
    Ze.postMessage({ id: e.id, error: r instanceof Error ? r.message : String(r) });
  }
};
