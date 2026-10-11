// Tiny, dependency-free XML parser. Web Workers have no DOMParser, so the
// OpenVINO IR parser (and anything else that needs XML) uses this instead.
// Supports elements, attributes, text, self-closing tags, comments, CDATA,
// processing instructions / DOCTYPE (skipped) and the five predefined entities
// plus numeric character references. Not validating; throws on malformed tags.

export type XmlElement = {
  tag: string;
  attrs: Record<string, string>;
  children: XmlElement[];
  text: string;
};

const ENTITIES: Record<string, string> = { lt: "<", gt: ">", amp: "&", quot: '"', apos: "'" };

export function decodeEntities(s: string): string {
  if (s.indexOf("&") < 0) return s;
  return s.replace(/&(#x[0-9a-fA-F]+|#[0-9]+|[a-zA-Z]+);/g, (m, e: string) => {
    if (e[0] === "#") {
      const code = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : m;
    }
    return ENTITIES[e] ?? m;
  });
}

const NAME_END = /[\s/>]/;

/** Parse an XML document and return its root element. */
export function parseXml(text: string): XmlElement {
  // Strip a UTF-8 BOM.
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const root: XmlElement = { tag: "#document", attrs: {}, children: [], text: "" };
  const stack: XmlElement[] = [root];
  let i = 0;
  const n = text.length;

  while (i < n) {
    const lt = text.indexOf("<", i);
    const top = stack[stack.length - 1];
    if (lt < 0) {
      top.text += decodeEntities(text.slice(i));
      break;
    }
    if (lt > i) top.text += decodeEntities(text.slice(i, lt));

    if (text.startsWith("<!--", lt)) {
      const end = text.indexOf("-->", lt + 4);
      if (end < 0) throw new Error("XML: unterminated comment");
      i = end + 3;
    } else if (text.startsWith("<![CDATA[", lt)) {
      const end = text.indexOf("]]>", lt + 9);
      if (end < 0) throw new Error("XML: unterminated CDATA");
      top.text += text.slice(lt + 9, end);
      i = end + 3;
    } else if (text[lt + 1] === "?") {
      const end = text.indexOf("?>", lt + 2);
      if (end < 0) throw new Error("XML: unterminated processing instruction");
      i = end + 2;
    } else if (text[lt + 1] === "!") {
      // DOCTYPE (possibly with an internal subset in [...]).
      let j = lt + 2;
      let depth = 0;
      for (; j < n; j++) {
        const c = text[j];
        if (c === "[") depth++;
        else if (c === "]") depth--;
        else if (c === ">" && depth <= 0) break;
      }
      i = j + 1;
    } else if (text[lt + 1] === "/") {
      const end = text.indexOf(">", lt + 2);
      if (end < 0) throw new Error("XML: unterminated end tag");
      const tag = text.slice(lt + 2, end).trim();
      if (stack.length <= 1 || top.tag !== tag) throw new Error(`XML: mismatched </${tag}>`);
      stack.pop();
      i = end + 1;
    } else {
      // Start tag: name, attributes, optional self-close.
      let j = lt + 1;
      while (j < n && !NAME_END.test(text[j])) j++;
      const tag = text.slice(lt + 1, j);
      if (!tag) throw new Error(`XML: empty tag name at ${lt}`);
      const el: XmlElement = { tag, attrs: {}, children: [], text: "" };
      let selfClose = false;
      for (;;) {
        while (j < n && /\s/.test(text[j])) j++;
        if (j >= n) throw new Error(`XML: unterminated <${tag}>`);
        const c = text[j];
        if (c === ">") { j++; break; }
        if (c === "/" && text[j + 1] === ">") { selfClose = true; j += 2; break; }
        const eq = text.indexOf("=", j);
        if (eq < 0) throw new Error(`XML: malformed attribute in <${tag}>`);
        const key = text.slice(j, eq).trim();
        let k = eq + 1;
        while (k < n && /\s/.test(text[k])) k++;
        const q = text[k];
        if (q !== '"' && q !== "'") throw new Error(`XML: unquoted attribute ${key} in <${tag}>`);
        const close = text.indexOf(q, k + 1);
        if (close < 0) throw new Error(`XML: unterminated attribute ${key} in <${tag}>`);
        el.attrs[key] = decodeEntities(text.slice(k + 1, close));
        j = close + 1;
      }
      top.children.push(el);
      if (!selfClose) stack.push(el);
      i = j;
    }
  }
  if (stack.length > 1) throw new Error(`XML: unclosed <${stack[stack.length - 1].tag}>`);
  const docRoot = root.children[0];
  if (!docRoot) throw new Error("XML: no root element");
  return docRoot;
}

/** First direct child with the given tag. */
export function child(el: XmlElement | undefined, tag: string): XmlElement | undefined {
  return el?.children.find((c) => c.tag === tag);
}

/** All direct children with the given tag. */
export function children(el: XmlElement | undefined, tag: string): XmlElement[] {
  return el ? el.children.filter((c) => c.tag === tag) : [];
}
