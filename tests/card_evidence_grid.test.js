/**
 * Exercise frigate-clip-player's evidence-grid logic under a minimal DOM shim.
 *
 * Why a shim rather than a browser: this host has no headless browser, and the
 * parts most worth pinning are pure logic -- offset parsing, cell construction,
 * seek clamping, and the pending-seek path -- none of which need layout. The
 * shim is deliberately small: it records the calls the card makes and lets the
 * test assert on them, rather than pretending to be a rendering engine.
 */

const fs = require("fs");
const path = require("path");
const assert = require("assert");

// --- minimal DOM ---------------------------------------------------------

class ClassList {
  constructor() { this._set = new Set(); }
  add(...names) { names.forEach((n) => this._set.add(n)); }
  remove(...names) { names.forEach((n) => this._set.delete(n)); }
  contains(name) { return this._set.has(name); }
}

class Element {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.parentNode = null;
    this.hidden = false;
    this.textContent = "";
    this.classList = new ClassList();
    this._innerHTML = "";
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
  removeAttribute(name) { delete this.attributes[name]; }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  removeChild(child) {
    const i = this.children.indexOf(child);
    if (i >= 0) this.children.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  replaceChildren(...nodes) {
    this.children.forEach((c) => { c.parentNode = null; });
    this.children = [];
    nodes.forEach((n) => this.appendChild(n));
  }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type, event) { (this.listeners[type] || []).forEach((fn) => fn(event || {})); }
  querySelector(selector) { return this._query[selector] || null; }
  set innerHTML(html) { this._innerHTML = html; }
  get innerHTML() { return this._innerHTML; }
}

class ShadowRoot extends Element {}

const document = {
  createElement(tag) { return new Element(tag); },
  head: new Element("head"),
};

global.document = document;
global.window = { MediaSource: function () {}, Hls: undefined };
global.HTMLElement = Element;
global.customElements = {
  _defined: {},
  get(name) { return this._defined[name]; },
  define(name, cls) { this._defined[name] = cls; },
};

// --- load the card -------------------------------------------------------

const CARD = path.join(__dirname, "..", "www", "frigate-vision", "frigate-clip-player.js");
const source = fs.readFileSync(CARD, "utf8");

// The class is defined for its side effect on `customElements`; evaluate it in
// this context so the shim above is what the module sees.
// eslint-disable-next-line no-new-func
new Function("document", "window", "customElements", "HTMLElement", source)(
  document, global.window, global.customElements, Element
);
const Player = global.customElements.get("frigate-clip-player");
assert.ok(Player, "the card must register itself");

Player.prototype.attachShadow = function () {
  // The constructor calls this before anything else can be assigned, so the
  // root and the elements `_render()` will look up have to be created here.
  // A real element exposes the root as `shadowRoot` as well as returning it,
  // and the card uses both forms.
  const shadow = new ShadowRoot("shadow-root");
  const video = new Element("video");
  video.readyState = 0;
  video.duration = NaN;
  video.currentTime = 0;
  video.pause = () => { video.paused = true; };
  video.load = () => {};
  video.canPlayType = () => "";
  shadow._query = {
    video,
    ".empty": new Element("div"),
    ".evidence": new Element("div"),
    ".sheet": new Element("img"),
    ".cells": new Element("div"),
  };
  this.shadowRoot = shadow;
  return shadow;
};

/** Build a card instance with its shadow DOM wired like a real one. */
function makeCard(config) {
  const card = new Player();
  card.setConfig(config);
  // setConfig -> _render() replaced the shadow DOM; the card now holds the
  // elements it looked up, which are the ones the shim handed it.
  return {
    card,
    video: card._video,
    empty: card._empty,
    evidence: card._evidence,
    sheet: card._sheet,
    cells: card._cells,
  };
}

/** A store entity carrying one notification, as the card reads it. */
function storeWith(item) {
  return {
    states: {
      "sensor.notifications_store": { attributes: { items: item ? [item] : [] } },
    },
  };
}

// --- tests ---------------------------------------------------------------

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log("ok   " + name);
  } catch (error) {
    failures += 1;
    console.log("FAIL " + name + "\n     " + error.message);
  }
}

check("offsets are parsed from the pipe-separated string", () => {
  const { card } = makeCard({ notification_id: "alert_1" });
  assert.deepStrictEqual(card._parseOffsets("1.5|9.2|17.8"), [1.5, 9.2, 17.8]);
});

check("a comma-separated value is NOT split (it never arrives intact)", () => {
  const { card } = makeCard({ notification_id: "alert_1" });
  // HA parses a comma-separated result as a tuple and the notification fails,
  // so a comma here means a value from some other source; parseFloat takes the
  // leading number rather than silently inventing six offsets.
  assert.deepStrictEqual(card._parseOffsets("1.5,9.2,17.8"), [1.5]);
});

check("non-numeric entries are dropped, not turned into NaN", () => {
  const { card } = makeCard({ notification_id: "alert_1" });
  assert.deepStrictEqual(card._parseOffsets("1.5||oops|9.2"), [1.5, 9.2]);
});

check("a source with a sheet builds one button per offset", () => {
  const { card, cells, evidence } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  assert.strictEqual(evidence.hidden, false, "the sheet must be shown");
  assert.strictEqual(cells.children.length, 6, "six cells for six offsets");
  assert.strictEqual(cells.children[0].dataset.cell, "1");
  assert.strictEqual(cells.children[5].dataset.cell, "6");
});

check("a notification without the fields renders no sheet (old notifications)", () => {
  const { card, cells, evidence } = makeCard({ notification_id: "alert_old" });
  card.hass = storeWith({
    id: "alert_old",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
  });
  assert.strictEqual(evidence.hidden, true, "no sheet for an old notification");
  assert.strictEqual(cells.children.length, 0);
});

check("an image without offsets renders no sheet (half-present pair)", () => {
  const { card, evidence } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "",
  });
  assert.strictEqual(evidence.hidden, true, "a grid that cannot seek is worse than none");
});

check("tapping a cell seeks the video to that offset and pauses", () => {
  const { card, video, cells } = makeCard({ notification_id: "alert_1" });
  video.readyState = 1; // metadata available
  video.duration = 66.3;
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  cells.children[2].dispatch("click");
  assert.strictEqual(video.currentTime, 17.8, "third cell seeks to the third offset");
  assert.strictEqual(video.paused, true, "tapping a cell pauses for comparison");
});

check("a seek beyond the real duration is clamped", () => {
  const { card, video, cells } = makeCard({ notification_id: "alert_1" });
  video.readyState = 1;
  // The offsets are clamped into the planned window server-side, but the
  // stream's own duration can be shorter; seeking past it is discarded.
  video.duration = 10;
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|60",
  });
  cells.children[2].dispatch("click");
  assert.ok(video.currentTime <= 10, "must not seek past the end");
  assert.ok(video.currentTime > 9.8, "must land near the end, not at zero");
});

check("a tap before metadata is applied once metadata arrives", () => {
  const { card, video, cells } = makeCard({ notification_id: "alert_1" });
  video.readyState = 0; // no metadata yet: currentTime would be discarded
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8",
  });
  cells.children[1].dispatch("click");
  assert.strictEqual(video.currentTime, 0, "cannot seek without metadata");
  assert.strictEqual(card._pendingSeek, 9.2, "the target must be parked");
  video.readyState = 1;
  video.duration = 66.3;
  video.dispatch("loadedmetadata");
  assert.strictEqual(video.currentTime, 9.2, "the parked target must be applied");
});

check("a failed image hides the sheet instead of showing a broken icon", () => {
  const { card, sheet, evidence } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2",
  });
  assert.strictEqual(evidence.hidden, false);
  sheet.dispatch("error");
  assert.strictEqual(evidence.hidden, true, "an expired sheet must not show a broken image");
});

check("dragging the scrubber highlights the nearest cell", () => {
  const { card, video, cells } = makeCard({ notification_id: "alert_1" });
  video.readyState = 1;
  video.duration = 66.3;
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  video.currentTime = 18.0; // closest to 17.8, the third cell
  video.dispatch("timeupdate");
  assert.strictEqual(cells.children[2].getAttribute("aria-current"), "true");
  assert.strictEqual(cells.children[0].getAttribute("aria-current"), null);
  video.currentTime = 45.0; // closest to 44.7, the sixth cell
  video.dispatch("timeupdate");
  assert.strictEqual(cells.children[5].getAttribute("aria-current"), "true");
  assert.strictEqual(cells.children[2].getAttribute("aria-current"), null);
});

check("a source with no playable url shows the empty state and no sheet", () => {
  const { card, evidence, empty } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2",
  });
  assert.strictEqual(empty.hidden, false, "the empty state must show");
  assert.strictEqual(evidence.hidden, true, "the sheet is only for comparison with a video");
});

console.log(failures ? `\n${failures} FAILED` : "\nALL PASSED");
process.exit(failures ? 1 : 0);
