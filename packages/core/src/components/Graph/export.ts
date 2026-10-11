// Export the graph SVG as a downloadable .svg or rasterized .png.
// Both go through downloadFile so shell save handlers (VS Code, Tauri) apply.
import { downloadFile } from "../../utils/download";

function serialize(svg: SVGSVGElement): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  return new XMLSerializer().serializeToString(clone);
}

export function exportSvg(svg: SVGSVGElement, name: string): void {
  downloadFile(serialize(svg), `${name}.svg`, "image/svg+xml;charset=utf-8");
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
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) return;
  downloadFile(new Uint8Array(await blob.arrayBuffer()), `${name}.png`, "image/png");
}
