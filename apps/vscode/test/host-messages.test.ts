import { describe, expect, it } from "vitest";
import { ScrapeError, scrapeUrl } from "@modelvisio/ai/scrape";
import { MAX_NOTIFY_CHARS, messageId, parseWebviewMessage, saveBytes, scrapeFailure } from "../src/messages";

describe("parseWebviewMessage", () => {
  it("rejects non-objects and unknown types", () => {
    for (const raw of [null, undefined, 1, "ready", [], {}, { type: "nope" }]) {
      expect(parseWebviewMessage(raw)).toBeNull();
    }
  });

  it("accepts ready", () => {
    expect(parseWebviewMessage({ type: "ready", extra: 1 })).toEqual({ type: "ready" });
  });

  it("keeps valid chat turns and drops empty or invalid ones", () => {
    const msg = parseWebviewMessage({
      type: "chat",
      id: 3,
      system: "sys",
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "" },
        { role: "system", content: "x" },
        { role: "user", content: 5 },
        null,
        { role: "assistant", content: "ok" },
      ],
    });
    expect(msg).toEqual({
      type: "chat",
      id: 3,
      system: "sys",
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "ok" },
      ],
    });
  });

  it("treats a beacon-shaped chat as an empty request, not a malformed one", () => {
    expect(parseWebviewMessage({ type: "chat", id: 1, messages: [] })).toEqual({
      type: "chat",
      id: 1,
      system: "",
      messages: [],
    });
  });

  it("requires a numeric id and a messages array for chat", () => {
    expect(parseWebviewMessage({ type: "chat", id: "1", messages: [] })).toBeNull();
    expect(parseWebviewMessage({ type: "chat", id: NaN, messages: [] })).toBeNull();
    expect(parseWebviewMessage({ type: "chat", id: 1 })).toBeNull();
  });

  it("validates scrape", () => {
    expect(parseWebviewMessage({ type: "scrape", id: 2, url: "https://x.dev" })).toEqual({
      type: "scrape",
      id: 2,
      url: "https://x.dev",
    });
    expect(parseWebviewMessage({ type: "scrape", id: 2, url: 7 })).toBeNull();
  });

  it("normalizes save payloads to string or Uint8Array", () => {
    const str = parseWebviewMessage({ type: "save", id: 1, filename: "a.txt", mime: "text/plain", data: "hé" });
    expect(str).toMatchObject({ data: "hé", mime: "text/plain" });

    const u8 = new Uint8Array([1, 2, 3]);
    expect(parseWebviewMessage({ type: "save", id: 1, filename: "a.bin", data: u8 })).toMatchObject({ data: u8, mime: "" });

    const ab = parseWebviewMessage({ type: "save", id: 1, filename: "a.bin", data: new Uint8Array([4, 5]).buffer });
    expect(ab && ab.type === "save" && Array.from(ab.data as Uint8Array)).toEqual([4, 5]);

    const view = new DataView(new Uint8Array([9, 8, 7, 6]).buffer, 1, 2);
    const dv = parseWebviewMessage({ type: "save", id: 1, filename: "a.bin", data: view });
    expect(dv && dv.type === "save" && Array.from(dv.data as Uint8Array)).toEqual([8, 7]);

    const arr = parseWebviewMessage({ type: "save", id: 1, filename: "a.bin", data: [0, 255] });
    expect(arr && arr.type === "save" && Array.from(arr.data as Uint8Array)).toEqual([0, 255]);

    expect(parseWebviewMessage({ type: "save", id: 1, filename: "a.bin", data: [256] })).toBeNull();
    expect(parseWebviewMessage({ type: "save", id: 1, filename: "a.bin", data: { 0: 1 } })).toBeNull();
    expect(parseWebviewMessage({ type: "save", id: 1, data: "x" })).toBeNull();
  });

  it("validates notify levels and caps the message length", () => {
    expect(parseWebviewMessage({ type: "notify", level: "warn", message: "m" })).toEqual({
      type: "notify",
      level: "warn",
      message: "m",
    });
    expect(parseWebviewMessage({ type: "notify", level: "debug", message: "m" })).toBeNull();
    const long = parseWebviewMessage({ type: "notify", level: "info", message: "x".repeat(MAX_NOTIFY_CHARS + 50) });
    expect(long && long.type === "notify" && long.message.length).toBe(MAX_NOTIFY_CHARS + 1);
  });
});

describe("messageId", () => {
  it("extracts type and id from malformed requests so they can be answered", () => {
    expect(messageId({ type: "save", id: 4, data: {} })).toEqual({ type: "save", id: 4 });
    expect(messageId({ type: "save" })).toBeNull();
    expect(messageId("x")).toBeNull();
  });
});

describe("saveBytes", () => {
  it("UTF-8 encodes strings and passes bytes through", () => {
    expect(Array.from(saveBytes("é"))).toEqual([0xc3, 0xa9]);
    const u8 = new Uint8Array([1]);
    expect(saveBytes(u8)).toBe(u8);
  });
});

describe("scrapeFailure", () => {
  it("relays ScrapeError status", () => {
    expect(scrapeFailure(new ScrapeError("blocked", 400))).toEqual({ error: "blocked", status: 400 });
    expect(scrapeFailure(new ScrapeError("timeout", 502))).toEqual({ error: "timeout", status: 502 });
  });

  it("maps anything else to 502", () => {
    expect(scrapeFailure(new Error("boom"))).toEqual({ error: "boom", status: 502 });
    expect(scrapeFailure("weird")).toEqual({ error: "Scrape failed", status: 502 });
  });

  it("names the modelvisio.scrapeAllowlist setting, not the server's env var", async () => {
    const e = new ScrapeError("Host not in MODELVISIO_SCRAPE_ALLOWLIST: evil.example (via redirect)");
    expect(scrapeFailure(e)).toEqual({
      error: "Host not in the modelvisio.scrapeAllowlist setting: evil.example (via redirect)",
      status: 400,
    });

    // The real scraper's message, so a reworded upstream error fails this test.
    const real = await scrapeUrl({
      url: "https://evil.example/page",
      allowlist: ["docs.nvidia.com"],
      resolveHost: async () => ["93.184.216.34"],
    }).catch((err: unknown) => err);
    const { error, status } = scrapeFailure(real);
    expect(status).toBe(400);
    expect(error).toContain("modelvisio.scrapeAllowlist");
    expect(error).not.toContain("MODELVISIO_SCRAPE_ALLOWLIST");
  });
});
