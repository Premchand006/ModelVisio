import { useT } from "../theme/ThemeContext";
import type { ThemePalette, ThemeName } from "../theme/theme";
import logoUrl from "../assets/logo.png";

/**
 * Full-screen "under construction" view shown when the user opens a surface that
 * isn't shipped yet (the desktop app / VS Code extension). Reads the active
 * palette via useT(), so it always matches whatever theme the site is in.
 */
export function BuildingPage({
  label, theme, onToggleTheme, onBack,
}: {
  label: string;
  theme: ThemeName;
  onToggleTheme: () => void;
  onBack: () => void;
}) {
  const t = useT();
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, background: t.bg, color: t.t0, display: "flex", flexDirection: "column", fontFamily: "'Inter',-apple-system,sans-serif", overflow: "auto" }}>
      <style>{`
        @keyframes mvHoist{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
        @keyframes mvFade{0%,100%{opacity:.3}50%{opacity:1}}
        @keyframes mvSpin{to{transform:rotate(360deg)}}
        @keyframes mvSpinR{to{transform:rotate(-360deg)}}
        @keyframes mvRise{0%{opacity:0;transform:translateY(10px)}100%{opacity:1;transform:translateY(0)}}
      `}</style>

      {/* Top bar — mirrors the app ribbon so the page feels native */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "13px 24px", borderBottom: `1px solid ${t.bdr}`, background: t.bg1, flexShrink: 0 }}>
        <button type="button" onClick={onBack} style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 7, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, fontSize: 12, fontWeight: 600, cursor: "pointer" }}>← Back</button>
        <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 0 }}>
          <img src={logoUrl} alt="ModelVisio" width={28} height={28} style={{ width: 28, height: 28, borderRadius: 7, objectFit: "cover", flexShrink: 0 }} />
          <span style={{ fontSize: 15, fontWeight: 800, color: t.t0, fontFamily: "'Space Grotesk'", letterSpacing: -.3, whiteSpace: "nowrap" }}>ModelVisio</span>
        </div>
        <button type="button" onClick={onToggleTheme} title="Toggle theme" style={{ width: 32, height: 32, borderRadius: 7, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t1, fontSize: 13, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>{theme === "dark" ? "☀" : "☾"}</button>
      </div>

      {/* Centered content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: "40px 20px", gap: 24 }}>
        <div style={{ animation: "mvRise .5s ease-out both" }}>
          <ArchitectArt t={t} />
        </div>

        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "5px 13px", borderRadius: 999, border: `1px solid ${t.acc}55`, background: t.acc + "14", color: t.acc, fontSize: 11, fontWeight: 700, letterSpacing: .8, textTransform: "uppercase" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: t.acc, animation: "mvFade 1.6s infinite" }} />
          Under construction
        </div>

        <div>
          <div style={{ fontSize: "clamp(34px, 8vw, 56px)", fontWeight: 800, fontFamily: "'Space Grotesk'", letterSpacing: "-0.03em", lineHeight: 1.05, color: t.t0 }}>Building</div>
          <div style={{ fontSize: "clamp(13px, 3vw, 16px)", color: t.t2, marginTop: 12, maxWidth: 460, lineHeight: 1.6 }}>
            The <b style={{ color: t.t1 }}>{label}</b> is being built. We're still drafting the blueprints — it'll land here soon.
          </div>
        </div>

        <button type="button" onClick={onBack} style={{ padding: "10px 26px", borderRadius: 8, border: "none", background: t.acc, color: "#fff", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
          Back to ModelVisio
        </button>
      </div>
    </div>
  );
}

/**
 * Blueprint scene: a lattice tower crane lowering a block onto a tiered glass
 * tower, with an architect's dimension annotation and turning machinery gears.
 * All strokes derive from the theme so it reads cleanly on dark and light.
 */
function ArchitectArt({ t }: { t: ThemePalette }) {
  const acc = t.acc;          // primary blueprint ink (building, hook, gears)
  const ink = t.t1;           // structural ink (crane)
  const grid = t.bdr;         // faint grid
  const panel = t.bg1;        // blueprint paper
  const edge = t.bdr2;        // panel border + ground
  const dim = t.t3;           // dimension annotations
  const fill = acc + "1A";    // translucent glass fill
  const GROUND = 232;

  return (
    <svg viewBox="0 0 360 268" fill="none" role="img" aria-label="Under construction blueprint" style={{ width: "min(430px, 90vw)", height: "auto", display: "block" }}>
      {/* Blueprint paper + grid */}
      <rect x="1" y="1" width="358" height="266" rx="18" fill={panel} stroke={edge} strokeWidth="1.5" />
      <g stroke={grid} strokeWidth="1" opacity="0.6">
        {Array.from({ length: 14 }, (_, i) => <line key={"v" + i} x1={24 + i * 24} y1={14} x2={24 + i * 24} y2={254} />)}
        {Array.from({ length: 10 }, (_, i) => <line key={"h" + i} x1={14} y1={24 + i * 24} x2={346} y2={24 + i * 24} />)}
      </g>

      {/* Ground */}
      <line x1={28} y1={GROUND} x2={332} y2={GROUND} stroke={edge} strokeWidth="3" strokeLinecap="round" />

      {/* Architect dimension line (left) */}
      <g stroke={dim} strokeWidth="1.4" strokeLinecap="round">
        <line x1={50} y1={GROUND} x2={50} y2={66} />
        <path d={`M46,72 L50,66 L54,72`} fill="none" />
        <path d={`M46,${GROUND - 6} L50,${GROUND} L54,${GROUND - 6}`} fill="none" />
        <line x1={50} y1={66} x2={64} y2={66} opacity="0.5" />
        <line x1={50} y1={GROUND} x2={64} y2={GROUND} opacity="0.5" />
      </g>

      {/* Tiered glass tower (built section) */}
      <g strokeLinejoin="round">
        <rect x={64} y={150} width={116} height={82} rx={3} fill={fill} stroke={acc} strokeWidth="2.5" />
        <rect x={80} y={106} width={84} height={44} rx={3} fill={fill} stroke={acc} strokeWidth="2.5" />
      </g>
      {/* Facade mullions */}
      <g stroke={acc} strokeWidth="1" opacity="0.45">
        {[80, 98, 116, 134, 152, 168].map((x) => <line key={"m1" + x} x1={x} y1={158} x2={x} y2={226} />)}
        {[172, 196, 214].map((y) => <line key={"f1" + y} x1={70} y1={y} x2={174} y2={y} />)}
        {[96, 112, 128, 148].map((x) => <line key={"m2" + x} x1={x} y1={112} x2={x} y2={146} />)}
        <line x1={86} y1={128} x2={158} y2={128} />
      </g>

      {/* Top tier — under construction (dashed wireframe, pulsing) */}
      <rect x={92} y={66} width={60} height={40} rx={3} fill="none" stroke={acc} strokeWidth="2.5" strokeDasharray="7 5" style={{ animation: "mvFade 2.2s ease-in-out infinite" }} />

      {/* Tower crane */}
      <g stroke={ink} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none">
        {/* Lattice mast */}
        <line x1={248} y1={GROUND} x2={248} y2={44} />
        <line x1={260} y1={GROUND} x2={260} y2={44} />
        {[232, 206, 180, 154, 128, 102, 76].map((y) => <line key={"rung" + y} x1={248} y1={y} x2={260} y2={y} />)}
        {[[232, 206], [180, 154], [128, 102]].map(([a, b], i) => <line key={"dL" + i} x1={248} y1={a} x2={260} y2={b} />)}
        {[[206, 180], [154, 128], [102, 76]].map(([a, b], i) => <line key={"dR" + i} x1={260} y1={a} x2={248} y2={b} />)}
        {/* Mast feet */}
        <line x1={248} y1={GROUND} x2={238} y2={GROUND} />
        <line x1={260} y1={GROUND} x2={270} y2={GROUND} />

        {/* Apex + tie cables */}
        <line x1={254} y1={44} x2={254} y2={18} />
        <line x1={254} y1={18} x2={130} y2={42} />
        <line x1={254} y1={18} x2={310} y2={42} />

        {/* Working jib (truss) */}
        <line x1={254} y1={42} x2={130} y2={42} />
        <line x1={254} y1={56} x2={150} y2={42} />
        <line x1={210} y1={42} x2={210} y2={48} />
        <line x1={186} y1={42} x2={186} y2={51} />
        <line x1={162} y1={42} x2={162} y2={54} />

        {/* Counter-jib */}
        <line x1={254} y1={42} x2={308} y2={42} />
        <line x1={254} y1={54} x2={300} y2={42} />
      </g>
      {/* Counterweight */}
      <rect x={300} y={34} width={20} height={22} rx={2} fill={ink} />
      {/* Trolley — sits over the open bay, clear of the building */}
      <rect x={193} y={40} width={13} height={9} rx={1.5} fill={ink} />

      {/* Hook + lifted block (animated descent/lift) */}
      <g style={{ animation: "mvHoist 2.8s ease-in-out infinite" }}>
        <line x1={199} y1={49} x2={199} y2={60} stroke={ink} strokeWidth="1.6" />
        <rect x={187} y={60} width={24} height={17} rx={2.5} fill={acc} stroke={ink} strokeWidth="1.4" />
        <line x1={193} y1={64} x2={193} y2={73} stroke={t.bg1} strokeWidth="1.2" opacity="0.6" />
        <line x1={205} y1={64} x2={205} y2={73} stroke={t.bg1} strokeWidth="1.2" opacity="0.6" />
      </g>

      {/* Machinery — interlocking gears (work in progress) */}
      <Gear cx={292} cy={202} r={20} teeth={11} color={acc} hub={panel} ink={ink} spin="mvSpin 9s linear infinite" />
      <Gear cx={321} cy={216} r={13} teeth={8} color={ink} hub={panel} ink={acc} spin="mvSpinR 6s linear infinite" />
    </svg>
  );
}

/** A clean gear: a ring of rounded teeth, a body, and a hollow hub. Spins via CSS. */
function Gear({ cx, cy, r, teeth, color, hub, ink, spin }: {
  cx: number; cy: number; r: number; teeth: number; color: string; hub: string; ink: string; spin: string;
}) {
  const tw = (2 * Math.PI * r) / teeth * 0.55; // tooth width
  const th = r * 0.3;                            // tooth height
  return (
    <g style={{ transformBox: "fill-box", transformOrigin: "center", animation: spin }}>
      {Array.from({ length: teeth }, (_, i) => (
        <rect
          key={i}
          x={cx - tw / 2} y={cy - r - th * 0.55} width={tw} height={th + 1} rx={tw * 0.25}
          fill={color} transform={`rotate(${(360 / teeth) * i} ${cx} ${cy})`}
        />
      ))}
      <circle cx={cx} cy={cy} r={r * 0.86} fill={color} />
      <circle cx={cx} cy={cy} r={r * 0.4} fill={hub} stroke={ink} strokeWidth={r * 0.07} />
    </g>
  );
}
