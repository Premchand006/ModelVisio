import { afterEach, describe, expect, it, vi } from "vitest";
import { detectOS, type NavigatorLike } from "../src/utils/platform";

const UA = {
  mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
  win: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
  linux: "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
  android: "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
  iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
  // iPadOS 13+ Safari requests the desktop site and claims to be a Mac.
  ipad: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
};

describe("detectOS", () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it("macOS via userAgentData (Chromium)", () => {
    expect(detectOS({ userAgentData: { platform: "macOS", mobile: false }, platform: "MacIntel", userAgent: UA.mac, maxTouchPoints: 0 })).toBe("mac");
  });

  it("macOS via legacy navigator.platform (Safari/Firefox)", () => {
    expect(detectOS({ platform: "MacIntel", userAgent: UA.mac, maxTouchPoints: 0 })).toBe("mac");
  });

  it("macOS from the user agent alone", () => {
    expect(detectOS({ userAgent: UA.mac })).toBe("mac");
  });

  it("Windows", () => {
    expect(detectOS({ userAgentData: { platform: "Windows", mobile: false }, platform: "Win32", userAgent: UA.win })).toBe("windows");
    expect(detectOS({ platform: "Win32", userAgent: UA.win })).toBe("windows");
  });

  it("Linux", () => {
    expect(detectOS({ userAgentData: { platform: "Linux", mobile: false }, platform: "Linux x86_64", userAgent: UA.linux })).toBe("linux");
    expect(detectOS({ platform: "Linux x86_64", userAgent: UA.linux })).toBe("linux");
  });

  it("iPad reporting MacIntel with touch support → other", () => {
    expect(detectOS({ platform: "MacIntel", userAgent: UA.ipad, maxTouchPoints: 5 })).toBe("other");
  });

  it("Android → other (even though its platform says Linux)", () => {
    expect(detectOS({ userAgentData: { platform: "Android", mobile: true }, platform: "Linux armv8l", userAgent: UA.android, maxTouchPoints: 5 })).toBe("other");
    expect(detectOS({ platform: "Linux armv8l", userAgent: UA.android })).toBe("other");
  });

  it("iPhone → other", () => {
    expect(detectOS({ platform: "iPhone", userAgent: UA.iphone, maxTouchPoints: 5 })).toBe("other");
  });

  it("ChromeOS and unknown platforms → other", () => {
    expect(detectOS({ userAgentData: { platform: "Chrome OS", mobile: false }, platform: "Linux x86_64", userAgent: "Mozilla/5.0 (X11; CrOS x86_64 14541.0.0)" })).toBe("other");
    expect(detectOS({ platform: "", userAgent: "" })).toBe("other");
  });

  it("no navigator / no window (SSR) → other", () => {
    expect(detectOS(null)).toBe("other");
    // Node exposes a global `navigator` describing the server; without a
    // browser `window` it must not be mistaken for the visitor's OS.
    expect(typeof window).toBe("undefined");
    expect(detectOS()).toBe("other");
  });

  it("reads window.navigator by default in a browser", () => {
    vi.stubGlobal("window", { navigator: { platform: "MacIntel", userAgent: UA.mac, maxTouchPoints: 0 } satisfies NavigatorLike });
    expect(detectOS()).toBe("mac");
  });
});
