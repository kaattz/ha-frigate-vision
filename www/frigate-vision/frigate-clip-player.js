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

class FrigateClipPlayer extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._entity = null;
    this._currentUrl = null;
    this._hls = null;
    this._hlsLoading = null;
  }

  setConfig(config) {
    if (!config || !config.entity) {
      throw new Error("frigate-clip-player: 'entity' is required");
    }
    this._entity = config.entity;
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    const state = hass.states[this._entity];
    const url =
      state && state.state && state.state !== "unknown" ? state.state : null;
    if (url !== this._currentUrl) {
      this._currentUrl = url;
      this._applySource(url);
    }
  }

  getCardSize() {
    return 6;
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
        <div class="empty" hidden>没有可播放的视频</div>
      </div>
    `;
    this._video = this.shadowRoot.querySelector("video");
    this._empty = this.shadowRoot.querySelector(".empty");
  }

  async _applySource(url) {
    if (!this._video) return;

    // Tear down any previous stream before switching, or the old fragments
    // keep being fetched and the element leaks a MediaSource.
    if (this._hls) {
      this._hls.destroy();
      this._hls = null;
    }
    this._video.removeAttribute("src");
    this._video.load();

    if (!url) {
      this._video.hidden = true;
      this._empty.hidden = false;
      return;
    }
    this._video.hidden = false;
    this._empty.hidden = true;

    // Safari (and iOS in particular) plays HLS natively; loading hls.js there
    // is wasted work and can be worse than the native path.
    if (this._video.canPlayType("application/vnd.apple.mpegurl")) {
      this._video.src = url;
      return;
    }

    const Hls = await this._loadHls();
    if (!Hls || !Hls.isSupported()) {
      // No MSE and no native HLS: surface it rather than showing a dead player.
      this._empty.textContent = "此浏览器无法播放该视频";
      this._video.hidden = true;
      this._empty.hidden = false;
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
      script.onload = () => resolve(window.Hls || null);
      script.onerror = () => resolve(null);
      document.head.appendChild(script);
    });
    return this._hlsLoading;
  }

  disconnectedCallback() {
    if (this._hls) {
      this._hls.destroy();
      this._hls = null;
    }
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
