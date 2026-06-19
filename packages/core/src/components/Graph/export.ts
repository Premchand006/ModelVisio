// Export the graph SVG as a downloadable .svg or rasterized .png.

function download(url: string, filename: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function serialize(svg: SVGSVGElement): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  return new XMLSerializer().serializeToString(clone);
}

export function exportSvg(svg: SVGSVGElement, name: string): void {
  const blob = new Blob([serialize(svg)], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  download(url, `${name}.svg`);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export async function exportPng(svg: SVGSVGElement, name: string, bg: string, scale = 2): Promise<void> {
  const w = Number(svg.getAttribute("width")) || svg.clientWidth;
  const h = Number(svg.getAttribute("height")) || svg.clientHeight;
  const data = serialize(svg);
  const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(data);
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("Could not render graph to image."));
    img.src = url;
  });
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, w * scale);
  canvas.height = Math.max(1, h * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const out = URL.createObjectURL(blob);
    download(out, `${name}.png`);
    setTimeout(() => URL.revokeObjectURL(out), 2000);
  }, "image/png");
}
