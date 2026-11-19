(() => {
  if (document.getElementById("whisper-meet-captions")) return;

  const host = document.createElement("div");
  host.id = "whisper-meet-captions";
  document.documentElement.append(host);
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      :host { all: initial; }
      #panel {
        position: fixed; z-index: 2147483647; left: 50%; bottom: 24px;
        transform: translateX(-50%); width: min(680px, calc(100vw - 28px));
        box-sizing: border-box; padding: 14px 16px; border: 1px solid #53606a;
        border-radius: 10px; background: #10191e; color: #f3f6f7;
        box-shadow: 0 8px 32px #0008; font: 15px/1.45 "Malgun Gothic", sans-serif;
      }
      #header, #actions { display: flex; align-items: center; gap: 10px; }
      #header { justify-content: space-between; margin-bottom: 8px; color: #c3d0d5; font-size: 12px; }
      #status { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      #status::before { content: ""; display: inline-block; width: 8px; height: 8px;
        margin-right: 7px; border-radius: 50%; background: #e5a84b; }
      #status[data-connected="true"]::before { background: #69c69b; }
      #captions { max-height: 120px; overflow: auto; font-size: 20px; font-weight: 600; }
      .line { padding: 2px 0; overflow-wrap: anywhere; }
      #actions { justify-content: flex-end; margin-top: 10px; }
      button { min-height: 34px; padding: 5px 10px; border: 1px solid #718089;
        border-radius: 6px; background: #25343c; color: inherit; font: inherit; cursor: pointer; }
      button:hover { background: #344850; }
      button:focus-visible { outline: 2px solid #75c9a2; outline-offset: 2px; }
      #notice { min-height: 1.4em; margin-top: 4px; color: #b7c8ce; font-size: 12px; }
      @media (max-width: 480px) {
        #panel { bottom: 12px; padding: 11px 12px; }
        #captions { font-size: 17px; }
        button { font-size: 13px; }
      }
    </style>
    <section id="panel" aria-label="Whisper 실시간 자막">
      <div id="header"><span>Whisper 자막</span><span id="status" role="status">로컬 Whisper 연결 중...</span></div>
      <div id="captions" aria-live="polite"></div>
      <div id="actions">
        <button id="copy" type="button">최근 문장 복사</button>
        <button id="chat" type="button">채팅 입력창에 넣기</button>
      </div>
      <div id="notice" role="status"></div>
    </section>`;

  const status = root.querySelector("#status");
  const captions = root.querySelector("#captions");
  const notice = root.querySelector("#notice");
  let latestText = "";
  let socket;

  function connect() {
    socket = new WebSocket("ws://127.0.0.1:8765");
    socket.addEventListener("open", () => {
      status.textContent = "로컬 마이크 연결됨";
      status.dataset.connected = "true";
    });
    socket.addEventListener("message", (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === "status") {
        status.textContent = message.text;
      } else if (message.type === "transcript" && message.text?.trim()) {
        latestText = message.text.trim();
        const line = document.createElement("div");
        line.className = "line";
        line.textContent = latestText;
        captions.append(line);
        while (captions.childElementCount > 4) captions.firstElementChild.remove();
        captions.scrollTop = captions.scrollHeight;
      }
    });
    socket.addEventListener("close", () => {
      status.textContent = "연결 끊김, 다시 연결 중...";
      status.dataset.connected = "false";
      setTimeout(connect, 2000);
    });
    socket.addEventListener("error", () => socket.close());
  }

  async function copyLatest() {
    if (!latestText) {
      notice.textContent = "복사할 인식 문장이 아직 없습니다.";
      return;
    }
    try {
      await navigator.clipboard.writeText(latestText);
      notice.textContent = "최근 문장을 복사했습니다.";
    } catch {
      notice.textContent = "클립보드에 복사하지 못했습니다.";
    }
  }

  async function putLatestInChat() {
    if (!latestText) {
      notice.textContent = "채팅에 넣을 인식 문장이 아직 없습니다.";
      return;
    }
    const buttons = [...document.querySelectorAll('button[aria-label], [role="button"][aria-label]')];
    const chatButton = buttons.find((button) => /chat|채팅/i.test(button.getAttribute("aria-label")));
    if (chatButton) chatButton.click();

    const deadline = Date.now() + 4000;
    let composer;
    while (Date.now() < deadline) {
      composer = [...document.querySelectorAll(
        '[contenteditable="true"][role="textbox"], textarea[aria-label], input[aria-label]'
      )].find((element) => /message|메시지|채팅/i.test(
        `${element.getAttribute("aria-label") || ""} ${element.getAttribute("data-placeholder") || ""}`
      ));
      if (composer) break;
      await new Promise((resolve) => setTimeout(resolve, 150));
    }

    if (!composer) {
      notice.textContent = "채팅 입력창을 찾지 못했습니다. 최근 문장을 복사해 주세요.";
      return;
    }
    const existingText = composer.isContentEditable ? composer.innerText : composer.value;
    if (existingText?.trim()) {
      notice.textContent = "채팅 입력창에 작성 중인 내용이 있어 변경하지 않았습니다.";
      return;
    }

    composer.focus();
    if (composer.isContentEditable) {
      document.execCommand("insertText", false, latestText);
    } else {
      const prototype = composer instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(prototype, "value").set.call(composer, latestText);
      composer.dispatchEvent(new Event("input", { bubbles: true }));
    }
    notice.textContent = "채팅 입력창에 넣었습니다. 내용을 확인한 뒤 Meet에서 직접 전송하세요.";
  }

  root.querySelector("#copy").addEventListener("click", copyLatest);
  root.querySelector("#chat").addEventListener("click", putLatestInChat);
  connect();
})();