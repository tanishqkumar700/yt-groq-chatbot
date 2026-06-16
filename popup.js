let currentVideoId = null;
const BACKEND_URL = "http://127.0.0.1:8000";

const statusEl = document.getElementById("status");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const chatBox = document.getElementById("chat-box");
const clearBtn = document.getElementById("clear-btn");

// 1. URL Capture & Initialization
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  try {
    const url = tabs[0].url;
    const urlObj = new URL(url);
    
    if (urlObj.hostname.includes("youtube.com")) {
      const urlParams = new URLSearchParams(urlObj.search);
      currentVideoId = urlParams.get("v");
    }

    if (currentVideoId) {
      updateStatus("Analyzing...", "checking");
      
      fetch(`${BACKEND_URL}/initialize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: currentVideoId })
      })
      .then(res => {
        if (!res.ok) throw new Error();
        return res.json();
      })
      .then(() => {
        updateStatus("Ready", "ready");
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
      })
      .catch(() => {
        updateStatus("CC Error", "error");
      });
    } else {
      updateStatus("No Video", "error");
    }
  } catch (e) {
    updateStatus("Error", "error");
  }
});

function updateStatus(text, type) {
  statusEl.innerText = text;
  statusEl.className = `status-badge status-${type}`;
}

// 2. Chat Processing Logic
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

function sendMessage() {
  const query = userInput.value.trim();
  if (!query) return;

  // Clear welcome card text layout on first real chat interaction
  const welcomeMsg = document.getElementById("welcome-msg");
  if (welcomeMsg) welcomeMsg.remove();

  appendBubble(query, "user");
  userInput.value = "";

  fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: currentVideoId, question: query })
  })
  .then(res => res.json())
  .then(data => {
    appendBubble(data.answer, "ai");
  })
  .catch(() => {
    appendBubble("Connection to backend server lost.", "ai");
  });
}

function appendBubble(text, sender) {
  const wrapper = document.createElement("div");
  wrapper.className = `message-row ${sender}`;

  const label = document.createElement("span");
  label.className = "msg-label";
  label.innerText = sender === "user" ? "You" : "AI Assistant";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerText = text;

  wrapper.appendChild(label);
  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  
  // Smooth scroll activation
  chatBox.scrollTo({
    top: chatBox.scrollHeight,
    behavior: 'smooth'
  });
}

// 3. Clear Chat Trigger Event
clearBtn.addEventListener("click", () => {
  chatBox.innerHTML = `
    <div class="system-welcome" id="welcome-msg">
      <p>👋 Ask me anything about the video you are currently watching. I will answer using only the video's transcript context.</p>
    </div>
  `;
});