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
 *   notification_id: alert_1790106601     # play this notification's clip
 *   # or, for testing:
 *   entity: input_text.some_url           # holds an id, or a URL directly
 *
 * Why `notification_id` is the normal form: the link that opens this popup is a
 * plain markdown link, because a markdown link is the only kind that sits
 * inline with the text above it -- every card-based button occupies a whole row.
 * A markdown link can only carry the id in the popup's hash, so each
 * notification gets its own popup and its own id, and no helper is involved.
 * The URL itself cannot travel in a hash either: it is unsigned-or-400-chars,
 * and the store's attributes, which are not length-capped, hold it instead.
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
    if (!config || (!config.entity && !config.notification_id)) {
      throw new Error(
        "frigate-clip-player: 'entity' or 'notification_id' is required"
      );
    }
    this._entity = config.entity || null;
    // A popup knows exactly which notification it belongs to, so it can be
    // told the id directly instead of having one written into a helper. That
    // keeps the link that opens it a plain markdown link, which is the only
    // kind of link that sits inline with the surrounding text.
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
    // A popup configured with a fixed notification id resolves from that id
    // alone -- it needs no helper, and it keeps working when the helper is
    // empty (which is its normal resting state).
    if (this._notificationId) {
      const url = this._lookupNotificationUrl(this._notificationId);
      if (url !== this._currentUrl) {
        this._currentUrl = url;
        this._applySource(url);
      }
      return;
    }
    const state = hass.states[this._entity];
    const raw =
      state && state.state && !this._isNotPlayable(state.state)
        ? state.state
        : null;
    const url = raw ? this._resolveUrl(raw) : null;
    if (url !== this._currentUrl) {
      this._currentUrl = url;
      this._applySource(url);
    }
  }

  /**
   * Turn the entity's state into a playable URL.
   *
   * The state is normally a notification id, because the URL is too long to
   * live in an `input_text` state. A bare URL is still accepted so the card can
   * be pointed at one directly, and because that was the original interface --
   * the two are told apart by shape rather than by a second config key, which
   * keeps one card config correct in both cases.
   */
  _resolveUrl(value) {
    const text = String(value).trim();
    if (text.startsWith("/") || text.includes("://")) {
      return text;
    }
    return this._lookupNotificationUrl(text);
  }

  /** Find the delivered play URL for one notification id, or null. */
  _lookupNotificationUrl(notificationId) {
    const store = this._hass && this._hass.states[STORE_ENTITY];
    const items = store && store.attributes && store.attributes[STORE_ITEMS_ATTRIBUTE];
    if (!Array.isArray(items)) return null;
    // Newest first, and ids are unique, so the first match is the one the user
    // just tapped. Falling back to any match keeps older notifications playable
    // as long as their footage is still within Frigate's retention.
    for (const item of items) {
      if (item && item.id === notificationId) {
        const url = item.hls_url ? String(item.hls_url).trim() : "";
        return url && !this._isNotPlayable(url) ? url : null;
      }
    }
    return null;
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
        /* The UA stylesheet's [hidden] { display: none } loses to any author
           rule that sets display, and both children below set it -- so the
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
      this._hls.loadSource(url);
      this._hls.attachMedia(this._video);
      return;
    }

    if (this._video.canPlayType("application/vnd.apple.mpegurl")) {
      this._video.src = url;
      return;
    }

    // Neither MSE nor native HLS: say so rather than showing a dead player.
    this._showEmpty(NOT_PLAYABLE_TEXT);
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
