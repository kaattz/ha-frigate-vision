/**
 * frigate-clip-player — play one activity's HLS clip inside a Bubble Card popup,
 * with the model's own evidence sheet underneath for side-by-side comparison.
 *
 * Why a custom card rather than advanced-camera-card (which is installed):
 * that card's `frigate` engine fetches media by *event id*, and this project
 * delivers *activities*, which can span several Frigate events. Its `generic`
 * engine does not play clips at all. Neither can play an arbitrary playlist,
 * which is what an activity needs.
 *
 * Why HLS rather than the MP4 clip: measured on this deployment, every MP4
 * endpoint (the integration's own proxy, Frigate's start/end clip, Frigate's
 * per-event clip, and the HA Frigate proxy) answers `200` with
 * `Transfer-Encoding: chunked` and no `Accept-Ranges`. A mobile browser will
 * not stream a 17 MB video it cannot range-request, which is exactly the
 * "nothing happens" symptom this replaces. HLS segments answer `206`.
 *
 * Usage:
 *   type: custom:frigate-clip-player
 *   notification_id: alert_1790106601     # play this notification's clip
 *   # or, for testing:
 *   entity: input_text.some_url           # holds an id, or a URL directly
 *
 * Why the URL comes from the store rather than a helper: the signed URL is
 * longer than `input_text`'s 255-character state limit, but the store's
 * `items` attribute is not length-capped, so the notification id travels in
 * the helper and the URLs are looked up here.
 */

const HLS_JS_URL = "/local/frigate-vision/hls.min.js";

// Where the delivered URLs live, and the attribute that holds them.
const STORE_ENTITY = "sensor.notifications_store";
const STORE_ITEMS_ATTRIBUTE = "items";

// Helper states that mean "no clip to play", never a URL. HA reports
// "unknown" before the helper is first written and "unavailable" when the
// helper (or the integration owning it) is not loaded; the literal strings
// "None"/"null" appear when a Python None is serialised into the state.
// Matched case-insensitively, so "None" and "none" are both covered.
const EMPTY_STATE_TEXT = "没有可播放的视频";
const NOT_PLAYABLE_TEXT = "此浏览器无法播放该视频";
const NOT_PLAYABLE = ["unknown", "unavailable", "none", "null"];
// _currentSource starts (and is reset) at this sentinel, never at null: the
// first `hass` set must always apply its source, even when the helper is empty.
// Otherwise "nothing applied yet" and "helper holds no URL" compare equal, no
// apply ever runs, and the card shows a black <video> instead of the empty
// state until the helper happens to hold a real URL. A symbol can never equal
// the signature string, which is what keeps that property intact.
const UNAPPLIED = Symbol("unapplied");

// "No clip and no sheet", shared so the two absent cases return one object.
const EMPTY_SOURCE = { url: null, evidence: null };

// The sheet is always three cells across -- `build_contact_sheet` rejects any
// other column count -- so the overlay's geometry is fixed and can be declared
// here rather than measured from the image.
const SHEET_COLUMNS = 3;

class FrigateClipPlayer extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._entity = null;
    this._currentSource = UNAPPLIED;
    this._hls = null;
    this._hlsLoading = null;
    this._offsets = [];
    this._cellButtons = [];
    // A tap that arrives before the media has metadata cannot be applied yet:
    // assigning `currentTime` on an element with readyState 0 is discarded, so
    // the tap would silently do nothing. The target is parked here and applied
    // on `loadedmetadata` instead.
    this._pendingSeek = null;
    // Bumped by every source switch (and by setConfig / disconnect). An
    // in-flight _applySource re-checks it after each await and bails out when
    // it no longer matches, so a superseded continuation can never construct
    // an Hls instance nor attach media to an element nobody owns.
    this._gen = 0;
  }

  setConfig(config) {
    if (!config || (!config.entity && !config.notification_id)) {
      throw new Error(
        "frigate-clip-player: 'entity' or 'notification_id' is required"
      );
    }
    this._entity = config.entity || null;
    // A popup knows exactly which notification it belongs to, so it can be
    // told the id directly instead of having one written into a helper.
    this._notificationId = config.notification_id || null;
    // Stop whatever was bound to the element _render() is about to discard —
    // an Hls instance or a native <video> would otherwise keep playing (and
    // downloading) behind a shadow DOM that no longer shows it.
    if (this._hls) {
      this._hls.destroy();
      this._hls = null;
    }
    this._stopVideo();
    this._render();
    // _render() replaced the shadow DOM with a fresh <video>, so the source now
    // remembered no longer describes what is on screen. Reset it to UNAPPLIED
    // and bump the generation: the next `hass` set re-applies to the new
    // element even when the helper's value has not changed, and any
    // continuation still in flight is voided instead of attaching to the
    // discarded element.
    this._currentSource = UNAPPLIED;
    this._pendingSeek = null;
    this._gen++;
  }

  set hass(hass) {
    this._hass = hass;
    const source = this._resolveSource();
    const signature = this._sourceSignature(source);
    if (signature !== this._currentSource) {
      this._currentSource = signature;
      this._applySource(source);
    }
  }

  /**
   * Resolve where to play from, and which sheet to compare against.
   *
   * Two configurations reach here: a popup that knows its notification id, and
   * one pointed at a helper. Both end in the same place -- the delivered
   * notification entry -- because that is where the URLs live.
   */
  _resolveSource() {
    // A popup configured with a fixed notification id resolves from that id
    // alone -- it needs no helper, and it keeps working when the helper is
    // empty (which is its normal resting state).
    if (this._notificationId) {
      return this._sourceFromItem(this._findNotification(this._notificationId));
    }
    const state = this._hass.states[this._entity];
    const raw =
      state && state.state && !this._isNotPlayable(state.state)
        ? state.state
        : null;
    if (!raw) return EMPTY_SOURCE;
    const text = String(raw).trim();
    // A bare URL is still accepted so the card can be pointed at one directly,
    // and because that was the original interface. The two are told apart by
    // shape rather than by a second config key, which keeps one card config
    // correct in both cases. A bare URL has no notification behind it, so
    // there is no sheet to look up.
    if (text.startsWith("/") || text.includes("://")) {
      return { url: text, evidence: null };
    }
    return this._sourceFromItem(this._findNotification(text));
  }

  /** Find one delivered notification entry by id, or null. */
  _findNotification(notificationId) {
    const store = this._hass && this._hass.states[STORE_ENTITY];
    const items =
      store && store.attributes && store.attributes[STORE_ITEMS_ATTRIBUTE];
    if (!Array.isArray(items)) return null;
    // Newest first, and ids are unique, so the first match is the one the user
    // just tapped. Falling back to any match keeps older notifications playable
    // as long as their footage is still within Frigate's retention.
    for (const item of items) {
      if (item && item.id === notificationId) {
        return item;
      }
    }
    return null;
  }

  /** Turn one notification entry into a playable URL and an optional sheet. */
  _sourceFromItem(item) {
    if (!item) return EMPTY_SOURCE;
    const rawUrl = item.hls_url ? String(item.hls_url).trim() : "";
    const url = rawUrl && !this._isNotPlayable(rawUrl) ? rawUrl : null;

    // The sheet is optional at every level: notifications delivered before this
    // existed have neither field, and an old notification must keep playing.
    // Requiring both an image and offsets means a half-present pair degrades to
    // "no sheet" rather than to a grid that cannot seek.
    const image = item.evidence_image_url
      ? String(item.evidence_image_url).trim()
      : "";
    const offsets = this._parseOffsets(item.evidence_offsets);
    const evidence = image && offsets.length ? { image, offsets } : null;
    return { url, evidence };
  }

  /**
   * Parse the delivered offsets.
   *
   * They arrive as text because the value crosses seven layers (delivery event,
   * blueprint variable, automation action, script field, `to_json`, `jq`, and a
   * sensor attribute); a string stays a string in all of them.
   *
   * The separator is a pipe, not a comma: HA's native template parser reads a
   * comma-separated result as a tuple, which then fails to serialise and takes
   * the whole notification down. Splitting on commas here would also split a
   * value that never arrives intact.
   *
   * Non-numeric entries are dropped rather than turned into NaN, which would
   * make every subsequent comparison false and leave a cell that does nothing.
   */
  _parseOffsets(value) {
    if (value === null || value === undefined) return [];
    return String(value)
      .split("|")
      .map((part) => Number.parseFloat(part.trim()))
      .filter((offset) => Number.isFinite(offset) && offset >= 0);
  }

  /** A cheap identity for a source, used to detect a real change. */
  _sourceSignature(source) {
    if (!source || (!source.url && !source.evidence)) return "";
    const evidence = source.evidence
      ? source.evidence.image + "|" + source.evidence.offsets.join(",")
      : "";
    return (source.url || "") + "\n" + evidence;
  }

  getCardSize() {
    return 8;
  }

  _isNotPlayable(value) {
    return NOT_PLAYABLE.indexOf(String(value).trim().toLowerCase()) !== -1;
  }

  _render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; }
        /* The UA stylesheet's [hidden] { display: none } loses to any author
           rule that sets display, and the children below set it -- so the
           hidden attribute silently did nothing and the controls stayed on
           screen next to the empty-state text. Restore it explicitly.
           Keep this comment free of backticks: it lives inside a template
           literal, and one would end the literal early. */
        [hidden] { display: none !important; }
        .wrap { position: relative; background: #000; border-radius: 12px; overflow: hidden; }
        video { width: 100%; display: block; max-height: 60vh; background: #000; }
        .empty {
          padding: 24px 16px; text-align: center; color: var(--secondary-text-color);
          font-size: 14px;
        }
        /* The sheet is one image with a transparent grid laid over it, not six
           separate crops. The cells in the source JPEG are exactly equal, so a
           uniform grid lines up with them without measuring anything -- and
           nothing can drift out of alignment while the image is loading. */
        .evidence {
          position: relative; margin-top: 8px; border-radius: 12px;
          overflow: hidden; background: #000; line-height: 0;
        }
        .sheet { display: block; width: 100%; }
        .cells {
          position: absolute; inset: 0; display: grid;
          grid-template-columns: repeat(${SHEET_COLUMNS}, 1fr);
          grid-auto-rows: 1fr;
        }
        .cells button {
          appearance: none; -webkit-appearance: none; box-sizing: border-box;
          margin: 0; padding: 0; background: transparent; cursor: pointer;
          border: 2px solid transparent; position: relative;
        }
        .cells button::after {
          content: attr(data-cell);
          position: absolute; top: 4px; left: 4px;
          min-width: 16px; height: 16px; padding: 0 4px;
          border-radius: 8px; background: rgba(0, 0, 0, 0.55);
          color: #fff; font-size: 11px; line-height: 16px; text-align: center;
        }
        .cells button:hover { border-color: rgba(255, 255, 255, 0.5); }
        .cells button[aria-current="true"] {
          border-color: var(--primary-color, #03a9f4);
          background: rgba(3, 169, 244, 0.18);
        }
        .cells button:focus-visible {
          outline: 2px solid var(--primary-color, #03a9f4); outline-offset: -2px;
        }
      </style>
      <div class="wrap">
        <video controls playsinline preload="metadata"></video>
        <div class="empty" hidden>${EMPTY_STATE_TEXT}</div>
      </div>
      <div class="evidence" hidden>
        <img class="sheet" alt="活动证据图">
        <div class="cells"></div>
      </div>
    `;
    this._video = this.shadowRoot.querySelector("video");
    this._empty = this.shadowRoot.querySelector(".empty");
    this._evidence = this.shadowRoot.querySelector(".evidence");
    this._sheet = this.shadowRoot.querySelector(".sheet");
    this._cells = this.shadowRoot.querySelector(".cells");
    this._offsets = [];
    this._cellButtons = [];

    this._video.addEventListener("loadedmetadata", () =>
      this._applyPendingSeek()
    );
    // Keeps the highlighted cell in step with the picture, so dragging the
    // scrubber answers "which grid cell am I looking at" as well as tapping a
    // cell answers "where is this frame".
    this._video.addEventListener("timeupdate", () => this._updateActiveCell());
    // A sheet that has aged out of `media_retention_days` must not leave a
    // broken-image icon in the popup. The clip it accompanies is on its own
    // retention clock, so the video may well still play.
    this._sheet.addEventListener("error", () => {
      if (this._evidence) this._evidence.hidden = true;
    });
  }

  // Show the placeholder with a caller-supplied message. Each branch states its
  // own reason, and the default text is restored whenever a playable URL is
  // applied — otherwise a transient hls.js load failure would leave its
  // "cannot play this video" capability message on screen for every later
  // empty state.
  _showEmpty(message) {
    if (this._empty) this._empty.textContent = message || EMPTY_STATE_TEXT;
    if (this._empty) this._empty.hidden = false;
    if (this._video) this._video.hidden = true;
    // The sheet exists only to be compared with the video, so with no video
    // there is nothing to compare and it must not be shown on its own.
    this._hideEvidence();
  }

  _hideEvidence() {
    if (this._evidence) this._evidence.hidden = true;
    if (this._cells) this._cells.replaceChildren();
    if (this._sheet) this._sheet.removeAttribute("src");
    this._offsets = [];
    this._cellButtons = [];
  }

  // Stop whatever the <video> is doing and drop its source. Needed on every
  // teardown: on the native HLS path (Safari/iOS) playback is owned by the
  // element itself, so destroying an Hls instance is not enough — and closing
  // the popup would otherwise leave audio playing with no visible player.
  _stopVideo() {
    if (!this._video) return;
    try {
      this._video.pause();
    } catch (err) {
      /* not playing, or no media API */
    }
    try {
      this._video.removeAttribute("src");
      this._video.load();
    } catch (err) {
      /* detached element */
    }
  }

  async _applySource(source) {
    const gen = ++this._gen;
    if (!this._video) return;

    // Tear down any previous stream before switching, or the old fragments
    // keep being fetched and the element leaks a MediaSource.
    if (this._hls) {
      this._hls.destroy();
      this._hls = null;
    }
    this._stopVideo();
    // A target parked for the previous clip must not be applied to this one.
    this._pendingSeek = null;

    // Applied before the first await so the sheet appears with the video rather
    // than after hls.js has finished loading.
    this._renderEvidence(source.evidence);

    if (!source.url) {
      this._showEmpty(EMPTY_STATE_TEXT);
      return;
    }
    this._video.hidden = false;
    this._empty.hidden = true;
    // A playable URL is back: restore the default placeholder text so the next
    // genuine empty state does not inherit a stale message.
    this._empty.textContent = EMPTY_STATE_TEXT;

    // Prefer hls.js whenever the browser has Media Source Extensions, and fall
    // back to native HLS only when it does not (iPhone Safari, mainly).
    //
    // This order is the opposite of "native first", and the reversal is the
    // point: `canPlayType("application/vnd.apple.mpegurl")` answers "maybe" on
    // some Chromium-derived browsers -- Huawei's ArkWeb among them -- whose
    // native HLS path then renders the first frame and never advances. The
    // symptom is a still picture with a play button that does nothing, which
    // looks like a broken clip even though the stream is fine. MSE is the
    // dependable capability test on those browsers, so it decides first.
    const canUseMse = typeof window.MediaSource !== "undefined";
    const Hls = canUseMse ? await this._loadHls() : null;
    // Superseded while hls.js was loading — by a newer URL, or by a disconnect.
    // Constructing Hls here would start downloading a clip the state no longer
    // references, into a hidden player with no owner that nothing can destroy.
    if (gen !== this._gen) return;

    if (Hls && Hls.isSupported()) {
      this._hls = new Hls();
      // Surface what hls.js says instead of leaving a frozen frame on screen:
      // a stalled stream and a rejected stream look identical otherwise.
      this._hls.on(Hls.Events.ERROR, (_evt, data) => {
        if (!data || !data.fatal) return;
        this._showEmpty(
          data.type === "networkError"
            ? "视频加载失败（链接可能已过期）"
            : "视频无法解码"
        );
        if (this._hls) {
          this._hls.destroy();
          this._hls = null;
        }
      });
      this._hls.loadSource(source.url);
      this._hls.attachMedia(this._video);
      return;
    }

    if (this._video.canPlayType("application/vnd.apple.mpegurl")) {
      this._video.src = source.url;
      return;
    }

    // Neither MSE nor native HLS: say so rather than showing a dead player.
    this._showEmpty(NOT_PLAYABLE_TEXT);
  }

  /**
   * Draw the evidence sheet and its tap targets.
   *
   * One `<img>` with a grid of transparent buttons over it. Cell *i* of the
   * image corresponds to `offsets[i]`: the model validates sample times as
   * sorted and strictly increasing, and the sheet is built in that same order.
   */
  _renderEvidence(evidence) {
    if (!this._evidence || !this._cells || !this._sheet) return;
    if (!evidence || !evidence.image || !evidence.offsets.length) {
      this._hideEvidence();
      return;
    }
    this._offsets = evidence.offsets;
    this._sheet.src = evidence.image;
    const buttons = evidence.offsets.map((offset, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.cell = String(index + 1);
      button.title = "跳到第 " + (index + 1) + " 格画面";
      button.setAttribute(
        "aria-label",
        "跳到第 " + (index + 1) + " 格画面，第 " + offset + " 秒"
      );
      button.addEventListener("click", () => this._seekTo(offset));
      return button;
    });
    this._cells.replaceChildren(...buttons);
    this._cellButtons = buttons;
    this._evidence.hidden = false;
  }

  /**
   * Jump to one cell's moment.
   *
   * Clamped against the element's own `duration`, not just the delivered
   * offsets: the offsets are clamped into the planned window server-side, but
   * the stream's real duration can differ slightly from that window, and
   * seeking past the end is discarded by the browser rather than corrected.
   * Playback is paused because the point of tapping a cell is to study a still.
   */
  _seekTo(offset) {
    const video = this._video;
    if (!video) return;
    let target = offset;
    const duration = video.duration;
    if (Number.isFinite(duration) && duration > 0) {
      target = Math.min(Math.max(target, 0), Math.max(duration - 0.05, 0));
    }
    try {
      video.pause();
    } catch (err) {
      /* no media API */
    }
    if (video.readyState < 1) {
      // No metadata yet: `currentTime` would be discarded. Park it and apply
      // on `loadedmetadata`, so the tap still lands.
      this._pendingSeek = target;
      return;
    }
    this._pendingSeek = null;
    try {
      video.currentTime = target;
    } catch (err) {
      // Some browsers refuse a seek before the first frame is decodable; the
      // queued target is the same recovery as the readyState path.
      this._pendingSeek = target;
    }
    this._updateActiveCell();
  }

  _applyPendingSeek() {
    if (this._pendingSeek === null || this._pendingSeek === undefined) return;
    const target = this._pendingSeek;
    this._pendingSeek = null;
    this._seekTo(target);
  }

  /** Highlight whichever cell is closest to the current play position. */
  _updateActiveCell() {
    const buttons = this._cellButtons;
    if (!buttons || !buttons.length || !this._video || !this._offsets.length) {
      return;
    }
    const current = this._video.currentTime;
    let best = -1;
    let bestDistance = Infinity;
    for (let index = 0; index < this._offsets.length; index += 1) {
      const distance = Math.abs(this._offsets[index] - current);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    }
    buttons.forEach((button, index) => {
      if (index === best) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    });
  }

  _loadHls() {
    if (window.Hls) return Promise.resolve(window.Hls);
    if (this._hlsLoading) return this._hlsLoading;
    this._hlsLoading = new Promise((resolve) => {
      const script = document.createElement("script");
      script.src = HLS_JS_URL;
      script.onload = () => {
        if (window.Hls) resolve(window.Hls);
        else this._hlsLoadFailed(script, resolve);
      };
      script.onerror = () => this._hlsLoadFailed(script, resolve);
      document.head.appendChild(script);
    });
    return this._hlsLoading;
  }

  // A failed <script> load is transient (offline, a dropped request, a cache
  // miss right after a deploy), so it must not be cached: the cached promise
  // would resolve to null forever, every later attempt would reuse it, the
  // script would never be re-injected and window.Hls would never populate.
  // Drop the element too, so a retry starts from a clean <head>.
  _hlsLoadFailed(script, resolve) {
    if (script && script.parentNode) script.parentNode.removeChild(script);
    this._hlsLoading = null;
    resolve(null);
  }

  disconnectedCallback() {
    // Void any _applySource still waiting on hls.js: it would otherwise attach
    // media to this now-detached element, or start a download nobody can stop.
    this._gen++;
    this._pendingSeek = null;
    if (this._hls) {
      this._hls.destroy();
      this._hls = null;
    }
    this._stopVideo();
  }
}

if (!customElements.get("frigate-clip-player")) {
  customElements.define("frigate-clip-player", FrigateClipPlayer);
  window.customCards = window.customCards || [];
  window.customCards.push({
    type: "frigate-clip-player",
    name: "Frigate Vision Clip Player",
    description: "播放门口活动片段（HLS）并对照六宫格证据图",
  });
}
