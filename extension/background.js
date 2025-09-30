// Background service worker for auxiliary tasks if needed later
chrome.runtime.onInstalled.addListener(() => {
    // No-op for now
  });
  
  // Provide a simple ping for debugging
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === "PING") {
      sendResponse({ ok: true });
    }
  });
  
  
  