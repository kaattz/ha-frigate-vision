/**
 * frigate-clip-player — play one activity's HLS clip inside a Bubble Card popup.
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
 *   entity: input_text.frigate_clip_url
 */

const HLS_JS_URL = "/local/frigate-vision/hls.min.js";

// Helper states that mean "no clip to play", never a URL. HA reports
// "unknown" before the helper is first written and "unavailable" when the
// helper (or the integration owning it) is not loaded; the literal strings
// "None"/"null" appear when a Python None is serialised into the state.
// Matched case-insensitively, so "None" and "none" are both covered.
const EMPTY_STATE_TEXT = "没有可播放的视频";
const NOT_PLAYABLE_TEXT = "此浏览器无法播放该视频";
const NOT_PLAYABLE = ["unknown", "unavailable", "none", "null"];
// _currentUrl starts (and is reset) at this sentinel, never at null: the first
// `hass` set must always apply its source, even when the helper is empty.
// Otherwise "nothing applied yet" and "helper holds no URL" compare equal, no
// apply ever runs, and the card shows a black <video> instead of the empty
// state until the helper happens to hold a real URL.
const UNAPPLIED = Symbol("unapplied");

class FrigateClipPlayer extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._entity = null;
    this._currentUrl = UNAPPLIED;
    this._hls = null;
    this._hlsLoading = null;
    // Bumped by every source switch (and by setConfig / disconnect). An
    // in-flight _applySource re-checks it after each await and bails out when
    // it no longer matches, so a superseded continuation can never construct
    // an Hls instance nor attach media to an element nobody owns.
    this._gen = 0;
  }

  setConfig(config) {
    if (!config || !config.entity) {
      throw new Error("frigate-clip-player: 'entity' is required");
    }
    this._entity = config.entity;
    // Stop whatever was bound to the element _render() is about to discard —
    // an Hls instance or a native <video> would otherwise keep playing (and
    // downloading) behind a shadow DOM that no longer shows it.
    if (this._hls) {
      this._hls.destroy();
      this._hls = null;
    }
    this._stopVideo();
    this._render();
    // _render() replaced the shadow DOM with a fresh <video>, so the URL now
    // remembered no longer describes what is on screen. Reset it to UNAPPLIED
    // and bump the generation: the next `hass` set re-applies the source to
    // the new element even when the helper's value has not changed, and any
    // continuation still in flight is voided instead of attaching to the
    // discarded element.
    this._currentUrl = UNAPPLIED;
    this._gen++;
  }

  set hass(hass) {
    this._hass = hass;
    const state = hass.states[this._entity];
    const url =
      state && state.state && !this._isNotPlayable(state.state)
        ? state.state
        : null;
    if (url !== this._currentUrl) {
      this._currentUrl = url;
      this._applySource(url);
    }
  }

  getCardSize() {
    return 6;
  }

  _isNotPlayable(value) {
    return NOT_PLAYABLE.indexOf(String(value).trim().toLowerCase()) !== -1;
  }

  _render() {
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; }
        .wrap { position: relative; background: #000; border-radius: 12px; overflow: hidden; }
        video { width: 100%; display: block; max-height: 60vh; background: #000; }
        .empty {
          padding: 24px 16px; text-align: center; color: var(--secondary-text-color);
          font-size: 14px;
        }
      </style>
      <div class="wrap">
        <video controls playsinline preload="metadata"></video>
        <div class="empty" hidden>${EMPTY_STATE_TEXT}</div>
      </div>
    `;
    this._video = this.shadowRoot.querySelector("video");
    this._empty = this.shadowRoot.querySelector(".empty");
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

  async _applySource(url) {
    const gen = ++this._gen;
    if (!this._video) return;

    // Tear down any previous stream before switching, or the old fragments
    // keep being fetched and the element leaks a MediaSource.
    if (this._hls) {
      this._hls.destroy();
      this._hls = null;
    }
    this._stopVideo();

    if (!url) {
      this._showEmpty(EMPTY_STATE_TEXT);
      return;
    }
    this._video.hidden = false;
    this._empty.hidden = true;
    // A playable URL is back: restore the default placeholder text so the next
    // genuine empty state does not inherit a stale message.
    this._empty.textContent = EMPTY_STATE_TEXT;

    // Safari (and iOS in particular) plays HLS natively; loading hls.js there
    // is wasted work and can be worse than the native path.
    if (this._video.canPlayType("application/vnd.apple.mpegurl")) {
      this._video.src = url;
      return;
    }

    const Hls = await this._loadHls();
    // Superseded while hls.js was loading — by a newer URL, or by a disconnect.
    // Constructing Hls here would start downloading a clip the state no longer
    // references, into a hidden player with no owner that nothing can destroy.
    if (gen !== this._gen) return;
    if (!Hls || !Hls.isSupported()) {
      // No MSE and no native HLS: surface it rather than showing a dead player.
      this._showEmpty(NOT_PLAYABLE_TEXT);
      return;
    }
    this._hls = new Hls();
    this._hls.loadSource(url);
    this._hls.attachMedia(this._video);
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
    description: "播放门口活动片段（HLS）",
  });
}
