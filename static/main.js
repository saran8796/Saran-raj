const chatBox = document.getElementById("chat-box");
const inputField = document.getElementById("userInput");

function appendMessage(text, sender) {
  const message = document.createElement("div");
  message.className = sender === "user" ? "user-message" : "bot-message";
  message.textContent = text;
  chatBox.appendChild(message);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
  const text = inputField.value.trim();
  if (!text) return;

  appendMessage(text, "user");
  inputField.value = "";

  fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
    .then(res => res.json())
    .then(data => {
      const response = data.response;
      appendMessage(response, "bot");

      const speak = new SpeechSynthesisUtterance(response);
      speechSynthesis.speak(speak);
    });
}

function startVoice() {
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = "en-US";
  recognition.start();

  recognition.onresult = function (event) {
    const transcript = event.results[0][0].transcript;
    inputField.value = transcript;
    sendMessage();
  };
}
