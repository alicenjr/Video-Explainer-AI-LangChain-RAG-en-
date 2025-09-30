// Storage helpers
async function getStoredKeys() {
    return new Promise((resolve) => {
      try {
        const inExtension = typeof chrome !== "undefined" && chrome && chrome.runtime && chrome.runtime.id;
        if (!inExtension || !chrome.storage || !chrome.storage.local) {
          resolve({ cohere_api_key: "", openrouter_api_key: "" });
          return;
        }
        chrome.storage.local.get(["cohere_api_key", "openrouter_api_key"], (res) => {
          resolve({
            cohere_api_key: (res && res.cohere_api_key) ? res.cohere_api_key : "",
            openrouter_api_key: (res && res.openrouter_api_key) ? res.openrouter_api_key : "",
          });
        });
      } catch (_) {
        resolve({ cohere_api_key: "", openrouter_api_key: "" });
      }
    });
  }
  
  async function setStoredKeys({ cohere_api_key, openrouter_api_key }) {
    return new Promise((resolve) => {
      try {
        const inExtension = typeof chrome !== "undefined" && chrome && chrome.runtime && chrome.runtime.id;
        if (!inExtension || !chrome.storage || !chrome.storage.local) {
          resolve();
          return;
        }
        chrome.storage.local.set({ cohere_api_key, openrouter_api_key }, () => resolve());
      } catch (_) {
        resolve();
      }
    });
  }
  
  // Backend API helpers
  async function handleJsonResponse(res, context) {
    let bodyText = "";
    try {
      const data = await res.json();
      if (!res.ok) {
        const msg = (data && (data.detail || data.message)) ? (data.detail || data.message) : `${context}: HTTP ${res.status}`;
        throw new Error(msg);
      }
      return data;
    } catch (e) {
      // If JSON parsing failed, try to read text
      try {
        if (!bodyText) bodyText = await res.text();
      } catch {}
      if (!res.ok) {
        throw new Error(`${context}: HTTP ${res.status}${bodyText ? ` - ${bodyText}` : ""}`);
      }
      // If ok but json parsing failed
      throw new Error(`${context}: Invalid JSON response`);
    }
  }
  
  async function apiSetApiKeys(base, payload) {
    const res = await fetch(`${base}/extension/set-api-keys`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res, "set-api-keys");
  }
  
  async function apiSetVideoId(base, video_id) {
    const res = await fetch(`${base}/extension/set-video-id`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id }),
    });
    return handleJsonResponse(res, "set-video-id");
  }
  
  async function apiInitializeChat(base) {
    const res = await fetch(`${base}/extension/initialize-chat`, {
      method: "POST",
    });
    return handleJsonResponse(res, "initialize-chat");
  }
  
  async function apiChat(base, message) {
    const res = await fetch(`${base}/extension/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    return handleJsonResponse(res, "chat");
  }
  
  async function apiGetStatus(base) {
    const res = await fetch(`${base}/extension/status`);
    return handleJsonResponse(res, "status");
  }
  
  
  