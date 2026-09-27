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

/**
 * A store entity carrying one notification, as the card reads it.
 *
 * Also carries a `callWS` that stands in for Home Assistant's `auth/sign_path`,
 * because the card re-signs before loading. It reports what it was asked to sign
 * on `hass.signed`, so a test can assert the card asked for the bare path.
 */
function storeWith(item) {
  const signed = [];
  return {
    signed,
    states: {
      "sensor.notifications_store": { attributes: { items: item ? [item] : [] } },
    },
    callWS(message) {
      signed.push(message.path);
      return Promise.resolve({ path: message.path + "?authSig=RESIGNED" });
    },
  };
}

// --- tests ---------------------------------------------------------------

let failures = 0;
const queued = [];

/**
 * Register a test. Runs after every `check()` in the file has been declared, so a
 * test may be async -- applying a source awaits a re-signing round trip now.
 */
function check(name, fn) {
  queued.push({ name, fn });
}

/** Let the card's pending async work settle before asserting on its result. */
function flush() {
  return new Promise((resolve) => setImmediate(resolve));
}

async function run() {
  for (const { name, fn } of queued) {
    try {
      await fn();
      console.log("ok   " + name);
    } catch (error) {
      failures += 1;
      console.log("FAIL " + name + "\n     " + error.message);
    }
  }
  console.log(failures ? `\n${failures} FAILED` : "\nALL PASSED");
  process.exit(failures ? 1 : 0);
}

check("offsets are parsed from the pipe-separated string", async () => {
  const { card } = makeCard({ notification_id: "alert_1" });
  assert.deepStrictEqual(card._parseOffsets("1.5|9.2|17.8"), [1.5, 9.2, 17.8]);
});

check("a signature-stripped path is what gets re-signed", async () => {
  // HA's signing secret lives in memory, so every signed URL a stored
  // notification holds stops working the moment HA restarts -- measured here as
  // 401 for a URL whose own expiry was still a day away. The card re-signs, and
  // it must ask for the bare path: the old `authSig` is not part of the path and
  // including it would sign a different URL than the one it then loads.
  const { card } = makeCard({ notification_id: "alert_1" });
  assert.strictEqual(
    card._pathWithoutSignature("/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=abc.def"),
    "/api/frigate/vod/cam/start/1/end/2/index.m3u8"
  );
  assert.strictEqual(
    card._pathWithoutSignature("/api/frigate_vision/media/e/a.jpg?authSig=x&height=360"),
    "/api/frigate_vision/media/e/a.jpg?height=360"
  );
  // A URL with no signature is already a path; returning it unchanged keeps the
  // re-sign idempotent.
  assert.strictEqual(
    card._pathWithoutSignature("/api/frigate/vod/cam/start/1/end/2/index.m3u8"),
    "/api/frigate/vod/cam/start/1/end/2/index.m3u8"
  );
  // An absolute URL still yields the path HA signs.
  assert.strictEqual(
    card._pathWithoutSignature("http://ha.local:8123/api/x.jpg?authSig=z"),
    "/api/x.jpg"
  );
});

check("a comma-separated value is NOT split (it never arrives intact)", async () => {
  const { card } = makeCard({ notification_id: "alert_1" });
  // HA parses a comma-separated result as a tuple and the notification fails,
  // so a comma here means a value from some other source; parseFloat takes the
  // leading number rather than silently inventing six offsets.
  assert.deepStrictEqual(card._parseOffsets("1.5,9.2,17.8"), [1.5]);
});

check("non-numeric entries are dropped, not turned into NaN", async () => {
  const { card } = makeCard({ notification_id: "alert_1" });
  assert.deepStrictEqual(card._parseOffsets("1.5||oops|9.2"), [1.5, 9.2]);
});

check("a nine-offset notification builds nine cells", async () => {
  // The sheet grew from 2x3 to 3x3 so the extra probed frames reach the popup.
  // The grid is declared as `grid-auto-rows: 1fr` with one button per offset, so
  // it should follow the count rather than assume two rows -- asserted here
  // because "should" is not evidence, and a hardcoded row count would clip the
  // third row without any error.
  const { card, cells, evidence } = makeCard({ notification_id: "alert_nine" });
  card.hass = storeWith({
    id: "alert_nine",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/nine.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7|52.3|61.8|70.2",
  });
  await flush();
  assert.strictEqual(evidence.hidden, false, "the sheet must be shown");
  assert.strictEqual(cells.children.length, 9, "nine cells for nine offsets");
  assert.strictEqual(cells.children[0].dataset.cell, "1");
  assert.strictEqual(cells.children[8].dataset.cell, "9");
});

check("an overlay still stops before a close-up beside the grid", async () => {
  // Older sheets, built before the close-up moved below the grid, may still be
  // stored and opened. That layout puts the close-up on the right, so the overlay
  // has to be narrower -- the card cannot know which layout it is looking at, so
  // both are handled from the image's own shape alone.
  //
  // 767x433 is a grid-plus-strip sheet; 1042x288 is the older grid-plus-column one
  // for six frames. Here the column is 275 wide, so the overlay is 767/1042 = 73.6%.
  const { card, cells } = makeCard({ notification_id: "alert_six" });
  cells.style = {};
  card.hass = storeWith({
    id: "alert_six",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/six.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  await flush();
  card._sheet.naturalWidth = 1042;
  card._sheet.naturalHeight = 288;
  card._insetFromExtraColumn();
  // Six 16:9 cells across two 144-tall rows span 767, so the overlay is
  // 767/1042 = 73.6% and the side column keeps the rest.
  const width = parseFloat(cells.style.width);
  assert.ok(
    Math.abs(width - 73.6) < 0.3,
    "a same-height sheet is the grid plus a side column; expected ~73.6%, got " +
      cells.style.width
  );
  assert.ok(
    Math.abs(parseFloat(cells.style.height) - 100) < 0.01,
    "the overlay spans the full height, got " + cells.style.height
  );
});

check("a grid-only sheet keeps the full overlay", async () => {
  // Both shapes have to stay correct: with no close-up the image is exactly the
  // frames, so the overlay covers all of it and no inset appears.
  for (const [offsets, width, height, label] of [
    ["1.5|9.2|17.8|26.4|35.1|44.7|52.3|61.8|70.2", 767, 431, "nine frames"],
    ["1.5|9.2|17.8|26.4|35.1|44.7", 767, 288, "six frames"],
  ]) {
    const { card, cells } = makeCard({ notification_id: "alert_nine" });
    cells.style = {};
    card.hass = storeWith({
      id: "alert_nine",
      hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
      evidence_image_url: "/api/frigate_vision/media/e/nine.jpg?authSig=Y",
      evidence_offsets: offsets,
    });
    await flush();
    card._sheet.naturalWidth = width;
    card._sheet.naturalHeight = height;
    card._insetFromExtraColumn();
    assert.ok(
      Math.abs(parseFloat(cells.style.width) - 100) < 0.01,
      label + " must keep the full width, got " + cells.style.width
    );
    assert.ok(
      Math.abs(parseFloat(cells.style.height) - 100) < 0.01,
      label + " must keep the full height, got " + cells.style.height
    );
  }
});

check("an overlay stops above a close-up strip below the grid", async () => {
  // The placement the owner asked for. The close-up sits UNDER the grid, so the
  // image is taller than the frames rather than wider, and the overlay has to stop
  // at the grid's bottom edge. Without that, the numbers are spread over the whole
  // image: a 433-tall sheet with a 288-tall grid made two 216-pixel rows, dropping
  // the second row's numbers 72 pixels below the frames they belong to and laying
  // invisible tap targets over the close-up.
  //
  // 767x433 is the real six-frame sheet: 288 of grid, a 1px seam, 144 of close-up.
  // The overlay must therefore be 288/433 = 66.5% tall.
  const { card, cells } = makeCard({ notification_id: "alert_six" });
  cells.style = {};
  card.hass = storeWith({
    id: "alert_six",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/six.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  // The offsets arrive with the notification, so the render must run before the
  // dimensions are faked in: `rows` comes from the offsets, and calling the sizing
  // with none loaded leaves it at zero and silently does nothing.
  return flush().then(() => {
    card._sheet.naturalWidth = 767;
    card._sheet.naturalHeight = 433;
    card._insetFromExtraColumn();
    assert.ok(
      Math.abs(parseFloat(cells.style.width) - 100) < 0.01,
      "the grid spans the full width, got " + cells.style.width
    );
    const height = parseFloat(cells.style.height);
    assert.ok(
      Math.abs(height - 66.5) < 0.5,
      "the overlay must stop above the strip, expected ~66.5%, got " +
        cells.style.height
    );
  });
});

check("a nine-frame sheet with a strip stops above it too", async () => {
  // 767x575: 432 of grid, 1px seam, 142 of close-up. Three rows, so the overlay is
  // 432/575 = 75.1% tall.
  const { card, cells } = makeCard({ notification_id: "alert_nine" });
  cells.style = {};
  card.hass = storeWith({
    id: "alert_nine",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/nine.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7|52.3|61.8|70.2",
  });
  await flush();
  card._sheet.naturalWidth = 767;
  card._sheet.naturalHeight = 575;
  card._insetFromExtraColumn();
  const height = parseFloat(cells.style.height);
  assert.ok(
    Math.abs(height - 75.1) < 0.5,
    "three rows of grid in a 575-tall sheet is 75.1%, got " + cells.style.height
  );
});

check("the overlay is sized once the image loads, when it is not yet cached", async () => {
  // The path the previous attempt got wrong. The card assigns `src` and sizes the
  // overlay in the same tick, so the dimensions are not there yet -- and a test
  // that sets `naturalWidth` by hand never notices. Here the dimensions arrive
  // only when the load event fires, which is what a first-time (uncached) view
  // does.
  const { card, cells } = makeCard({ notification_id: "alert_nine" });
  cells.style = {};
  card._sheet.complete = false;
  card._sheet.naturalWidth = 0;
  card._sheet.naturalHeight = 0;
  card.hass = storeWith({
    id: "alert_nine",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/nine.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7|52.3|61.8|70.2",
  });
  await flush();
  // The overlay cannot be sized yet, so it must at least not throw on the missing
  // dimensions, and it must have registered a listener for when they arrive.
  assert.strictEqual(card._sheet.listeners.load.length, 1, "must wait for load");
  // Now the image arrives.
  card._sheet.complete = true;
  card._sheet.naturalWidth = 1215;
  card._sheet.naturalHeight = 431;
  card._sheet.dispatch("load");
  const width = parseFloat(cells.style.width);
  assert.ok(
    Math.abs(width - 63.1) < 0.3,
    "sizing must happen on load, got " + cells.style.width
  );
});

check("the overlay is sized immediately when the image is already cached", async () => {
  // The other half of the race: a cached image can be complete before the sizing
  // runs, in which case a load listener would never fire and an earlier attempt
  // silently did nothing.
  const { card, cells } = makeCard({ notification_id: "alert_nine" });
  cells.style = {};
  card.hass = storeWith({
    id: "alert_nine",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/nine.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7|52.3|61.8|70.2",
  });
  await flush();
  card._sheet.complete = true;
  card._sheet.naturalWidth = 1215;
  card._sheet.naturalHeight = 431;
  card._insetFromExtraColumn();
  assert.ok(
    Math.abs(parseFloat(cells.style.width) - 63.1) < 0.3,
    "a cached image must be sized without waiting, got " + cells.style.width
  );
  assert.strictEqual(
    card._sheet.listeners.load.length,
    1,
    "a cached image needs no extra listener, but a harmless one may be kept"
  );
});

check("a source with a sheet builds one button per offset", async () => {
  const { card, cells, evidence } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  await flush();
  assert.strictEqual(evidence.hidden, false, "the sheet must be shown");
  assert.strictEqual(cells.children.length, 6, "six cells for six offsets");
  assert.strictEqual(cells.children[0].dataset.cell, "1");
  assert.strictEqual(cells.children[5].dataset.cell, "6");
});

check("a notification without the fields renders no sheet (old notifications)", async () => {
  const { card, cells, evidence } = makeCard({ notification_id: "alert_old" });
  card.hass = storeWith({
    id: "alert_old",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
  });
  await flush();
  assert.strictEqual(evidence.hidden, true, "no sheet for an old notification");
  assert.strictEqual(cells.children.length, 0);
});

check("an image without offsets renders no sheet (half-present pair)", async () => {
  const { card, evidence } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "",
  });
  await flush();
  assert.strictEqual(evidence.hidden, true, "a grid that cannot seek is worse than none");
});

check("tapping a cell seeks the video to that offset and pauses", async () => {
  const { card, video, cells } = makeCard({ notification_id: "alert_1" });
  video.readyState = 1; // metadata available
  video.duration = 66.3;
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  await flush();
  cells.children[2].dispatch("click");
  assert.strictEqual(video.currentTime, 17.8, "third cell seeks to the third offset");
  assert.strictEqual(video.paused, true, "tapping a cell pauses for comparison");
});

check("a seek beyond the real duration is clamped", async () => {
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
  await flush();
  cells.children[2].dispatch("click");
  assert.ok(video.currentTime <= 10, "must not seek past the end");
  assert.ok(video.currentTime > 9.8, "must land near the end, not at zero");
});

check("a tap before metadata is applied once metadata arrives", async () => {
  const { card, video, cells } = makeCard({ notification_id: "alert_1" });
  video.readyState = 0; // no metadata yet: currentTime would be discarded
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8",
  });
  await flush();
  cells.children[1].dispatch("click");
  assert.strictEqual(video.currentTime, 0, "cannot seek without metadata");
  assert.strictEqual(card._pendingSeek, 9.2, "the target must be parked");
  video.readyState = 1;
  video.duration = 66.3;
  video.dispatch("loadedmetadata");
  assert.strictEqual(video.currentTime, 9.2, "the parked target must be applied");
});

check("a failed image hides the sheet instead of showing a broken icon", async () => {
  const { card, sheet, evidence } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2",
  });
  await flush();
  assert.strictEqual(evidence.hidden, false);
  sheet.dispatch("error");
  assert.strictEqual(evidence.hidden, true, "an expired sheet must not show a broken image");
});

check("dragging the scrubber highlights the nearest cell", async () => {
  const { card, video, cells } = makeCard({ notification_id: "alert_1" });
  video.readyState = 1;
  video.duration = 66.3;
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2|17.8|26.4|35.1|44.7",
  });
  await flush();
  video.currentTime = 18.0; // closest to 17.8, the third cell
  video.dispatch("timeupdate");
  assert.strictEqual(cells.children[2].getAttribute("aria-current"), "true");
  assert.strictEqual(cells.children[0].getAttribute("aria-current"), null);
  video.currentTime = 45.0; // closest to 44.7, the sixth cell
  video.dispatch("timeupdate");
  assert.strictEqual(cells.children[5].getAttribute("aria-current"), "true");
  assert.strictEqual(cells.children[2].getAttribute("aria-current"), null);
});

check("a source with no playable url shows the empty state and no sheet", async () => {
  const { card, evidence, empty } = makeCard({ notification_id: "alert_1" });
  card.hass = storeWith({
    id: "alert_1",
    hls_url: "",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
    evidence_offsets: "1.5|9.2",
  });
  await flush();
  assert.strictEqual(empty.hidden, false, "the empty state must show");
  assert.strictEqual(evidence.hidden, true, "the sheet is only for comparison with a video");
});

// --- the `entity` path, which is what the deployed popup actually uses ------
// The popup passes `entity: input_text.frigate_clip_notification_id`, not
// `notification_id`. That helper holds an id, so the card has to resolve it
// through the store exactly like the id-configured form does. Testing only the
// `notification_id` form would leave the production path uncovered.

check("the entity path resolves an id through the store and shows the sheet", async () => {
  const { card, evidence, cells } = makeCard({
    entity: "input_text.frigate_clip_notification_id",
  });
  card.hass = {
    states: {
      "input_text.frigate_clip_notification_id": { state: "alert_9" },
      "sensor.notifications_store": {
        attributes: {
          items: [
            {
              id: "alert_9",
              hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=X",
              evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=Y",
              evidence_offsets: "4.1|7.4|38.9|46.3|55.9|58.9",
            },
          ],
        },
      },
    },
  };
  await flush();
  assert.strictEqual(evidence.hidden, false, "the sheet must be shown");
  assert.strictEqual(cells.children.length, 6, "six cells for six offsets");
  assert.ok(
    String(card._sheet.src).indexOf("authSig=Y") !== -1,
    "the signed image must be assigned to the img"
  );
});

check("the entity path still works when the helper is empty", async () => {
  const { card, evidence } = makeCard({
    entity: "input_text.frigate_clip_notification_id",
  });
  card.hass = {
    states: {
      "input_text.frigate_clip_notification_id": { state: "" },
      "sensor.notifications_store": { attributes: { items: [] } },
    },
  };
  await flush();
  assert.strictEqual(evidence.hidden, true, "no id means no sheet");
});

check("a bare URL in the entity still plays, with no sheet to look up", async () => {
  const { card, evidence } = makeCard({ entity: "input_text.some_url" });
  card.hass = {
    states: {
      "input_text.some_url": { state: "/api/frigate/vod/cam/x.m3u8?authSig=Z" },
    },
  };
  await flush();
  assert.strictEqual(evidence.hidden, true, "a bare URL has no notification behind it");
  assert.strictEqual(card._currentSource.indexOf("/api/frigate/vod/cam/x.m3u8"), 0);
});

check("a stored notification is re-signed before it loads", async () => {
  // The reported failure: "视频加载失败（链接可能已过期）" with the sheet missing too.
  // Both URLs in a stored notification were signed with a secret that lived in the
  // previous HA process, so a restart 401s all of them -- measured on this
  // deployment, where the tokens' own expiry was still a day away. The card has to
  // re-sign, or every notification older than the last restart is dead.
  const { card, sheet } = makeCard({ notification_id: "alert_1" });
  const hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=STALE",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=STALE",
    evidence_offsets: "1.5|9.2",
  });
  // Capture what the player is handed. Asserting on the element instead would
  // depend on which HLS path this environment happens to take, and the contract
  // that matters is what reaches `_applySource`.
  const original = card._applySource.bind(card);
  let applied = null;
  card._applySource = (source) => {
    applied = source;
    return original(source);
  };
  card.hass = hass;
  await flush();
  assert.deepStrictEqual(
    hass.signed,
    [
      "/api/frigate/vod/cam/start/1/end/2/index.m3u8",
      "/api/frigate_vision/media/e/a.jpg",
    ],
    "both URLs must be re-signed from their bare paths"
  );
  assert.ok(applied, "a source must be applied");
  assert.ok(
    applied.url.indexOf("authSig=RESIGNED") !== -1,
    "the clip must load the freshly signed URL, got " + applied.url
  );
  assert.ok(
    applied.evidence.image.indexOf("authSig=RESIGNED") !== -1,
    "the sheet must load the freshly signed URL, got " + applied.evidence.image
  );
  assert.deepStrictEqual(
    applied.evidence.offsets,
    [1.5, 9.2],
    "re-signing must not disturb the offsets"
  );
  // And the sheet really is rendered from that URL.
  assert.ok(
    String(sheet.src).indexOf("authSig=RESIGNED") !== -1,
    "the rendered sheet must use the signed URL, got " + sheet.src
  );
});

check("a failed re-sign falls back to the delivered URL", async () => {
  // A popup that cannot sign (not signed in, or a socket that is down) should
  // still try the URL it was given: a stale link sometimes still works, and an
  // unsigned one never does.
  const { card } = makeCard({ notification_id: "alert_1" });
  const hass = storeWith({
    id: "alert_1",
    hls_url: "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=STALE",
    evidence_image_url: "/api/frigate_vision/media/e/a.jpg?authSig=STALE",
    evidence_offsets: "1.5|9.2",
  });
  hass.callWS = () => Promise.reject(new Error("no auth"));
  const original = card._applySource.bind(card);
  let applied = null;
  card._applySource = (source) => {
    applied = source;
    return original(source);
  };
  card.hass = hass;
  await flush();
  assert.ok(applied, "a source must still be applied");
  assert.strictEqual(
    applied.url,
    "/api/frigate/vod/cam/start/1/end/2/index.m3u8?authSig=STALE",
    "the original URL is the fallback, got " + applied.url
  );
});

run();
