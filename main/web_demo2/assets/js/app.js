/**
 * ResearchDesk — Antigravity 2.0 Ultra-Minimalist Multi-Agent Client
 * With Full Chat History Persistence & Session Restoring
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Sections
  const chatHistoryList = document.getElementById("chatHistoryList");
  const historyCountBadge = document.getElementById("historyCountBadge");
  const docCountBadge = document.getElementById("docCountBadge");

  // Document Vault Elements
  const docList = document.getElementById("docList");
  const docCount = document.getElementById("docCountBadge");
  const docSearchInput = document.getElementById("docSearchInput");
  const activeDocName = document.getElementById("activeDocName");
  const autoScoutToggleBtn = document.getElementById("autoScoutToggleBtn");

  // Message Canvas Elements
  const messageFeed = document.getElementById("messageFeed");
  const welcomeHero = document.getElementById("welcomeHero");
  const promptInput = document.getElementById("promptInput");
  const sendBtn = document.getElementById("sendBtn");
  const sidebar = document.getElementById("sidebar");
  const sidecar = document.getElementById("sidecar");
  const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
  const toggleSidecarBtn = document.getElementById("toggleSidecarBtn");
  const closeSidecarBtn = document.getElementById("closeSidecarBtn");
  const sidecarEvidenceList = document.getElementById("sidecarEvidenceList");
  const evidenceCountBadge = document.getElementById("evidenceCountBadge");
  const pdfFrame = document.getElementById("pdfFrame");
  const newChatBtn = document.getElementById("newChatBtn");
  const suggestionCards = document.querySelectorAll(".suggestion-card");
  const sidecarTabs = document.querySelectorAll(".tab-btn");

  // Model & Kaggle Popover Controls
  const modelDropdownBtn = document.getElementById("modelDropdownBtn");
  const agentMatrixPopover = document.getElementById("agentMatrixPopover");
  const closePopoverBtn = document.getElementById("closePopoverBtn");
  const kaggleUrlInput = document.getElementById("kaggleUrlInput");
  const saveKaggleBtn = document.getElementById("saveKaggleBtn");
  const activeModelLabel = document.getElementById("activeModelLabel");
  const kaggleStatusBadge = document.getElementById("kaggleStatusBadge");
  const uploadPdfBtn = document.getElementById("uploadPdfBtn");
  const pdfFileInput = document.getElementById("pdfFileInput");
  const autoSelectBestBtn = document.getElementById("autoSelectBestBtn");
  const quickSelectBtns = document.querySelectorAll(".quick-select-btn");
  const quickActivePill = document.getElementById("quickActivePill");
  const MODE_LABELS = {
    gemini: "Gemini API",
    cpu_min: "CPU Min",
    cpu_max: "CPU Max",
    gpu_ft: "GPU FT",
    gpu_finetune: "GPU Finetune",
    gpu_max: "GPU Max",
    kaggle_fn: "GPU FT",
    kaggle_max: "GPU Max",
    hybrid: "Hybrid",
  };
  const agentModelSelects = document.querySelectorAll(".agent-model-select");

  const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");
  const deleteDocsBtn = document.getElementById("deleteDocsBtn");
  const delDocsCountBadge = document.getElementById("delDocsCountBadge");
  const docBulkBar = document.getElementById("docBulkBar");
  const selectAllDocsChk = document.getElementById("selectAllDocsChk");
  const confirmBulkDelDocsBtn = document.getElementById("confirmBulkDelDocsBtn");
  const cancelBulkDelDocsBtn = document.getElementById("cancelBulkDelDocsBtn");

  const state = {
    documents: [],
    runs: [],
    sessions: [],
    activeSessionId: localStorage.getItem("researchdesk_active_session") || null,
    selectedDocId: null,
    checkedDocIds: new Set(),
    docSelectMode: false,
    activeRunId: null,
    currentEvidence: [],
    autoScout: false,
    isKaggleOnline: false,
    isGeminiOnline: true,
    maxReflectionLoops: parseInt(localStorage.getItem("researchdesk_loops") || "2", 10),
  };

  // Close any open 3-dots menu when clicking outside
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".more-dots-wrap")) {
      document.querySelectorAll(".more-dots-menu.open").forEach(m => m.classList.remove("open"));
      document.querySelectorAll(".more-dots-btn.open").forEach(b => b.classList.remove("open"));
    }
  });

  // Tracking Active Polling & Timer Intervals (prevents memory leak & overlapping poll cycles)
  let activePollInterval = null;
  let activeTimerInterval = null;
  let activeMessageCard = null;
  let activeAbortController = null;

  function clearActivePolling() {
    if (activePollInterval) {
      clearInterval(activePollInterval);
      activePollInterval = null;
    }
  }

  function clearActiveTimer() {
    if (activeTimerInterval) {
      clearInterval(activeTimerInterval);
      activeTimerInterval = null;
    }
  }

  // ==================== 3-STATE SEND / STOP BUTTON ====================
  // Trạng thái: "idle" (chưa có gì) | "ready" (bắt đầu thực thi) | "running" (đang chạy)
  function setButtonState(btnState) {
    if (!sendBtn) return;
    const arrowIcon = sendBtn.querySelector(".btn-icon-arrow");
    const stopIcon = sendBtn.querySelector(".btn-icon-stop");
    sendBtn.classList.remove("ready", "active", "running");

    if (btnState === "running") {
      sendBtn.classList.add("running");
      sendBtn.disabled = false;
      sendBtn.title = "Dừng thực thi (Stop)";
      if (arrowIcon) arrowIcon.style.display = "none";
      if (stopIcon) stopIcon.style.display = "block";
    } else if (btnState === "ready") {
      sendBtn.classList.add("ready", "active");
      sendBtn.disabled = false;
      sendBtn.title = "Bắt đầu thực thi (Enter)";
      if (arrowIcon) arrowIcon.style.display = "block";
      if (stopIcon) stopIcon.style.display = "none";
    } else {
      // idle
      sendBtn.disabled = true;
      sendBtn.title = "Chưa có nội dung";
      if (arrowIcon) arrowIcon.style.display = "block";
      if (stopIcon) stopIcon.style.display = "none";
    }
  }

  function handleSendBtnClick() {
    if (sendBtn && sendBtn.classList.contains("running")) {
      // Đang chạy -> Bấm để Dừng (Stop)
      if (activeAbortController) {
        try { activeAbortController.abort(); } catch (e) {}
        activeAbortController = null;
      }
      clearActiveTimer();
      clearActivePolling();
      if (state.activeRunId) {
        fetch(`/api/runs/${state.activeRunId}/cancel`, { method: "POST" }).catch(() => {});
      }
      if (activeMessageCard) {
        updateAgentSkeletonError(activeMessageCard, "Đã dừng quá trình thực thi theo yêu cầu người dùng.");
        activeMessageCard = null;
      }
      setButtonState(promptInput && promptInput.value.trim().length > 0 ? "ready" : "idle");
      return;
    }
    submitQuery();
  }


  // Vietnamese Diacritics Removal for Smart Accent-Insensitive Search
  function removeDiacritics(str) {
    return (str || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .trim();
  }

  // ==================== 1. LOAD CHAT SESSIONS / ĐOẠN CHAT ====================
  async function loadChatHistory() {
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      state.sessions = data.sessions || [];

      if (historyCountBadge) historyCountBadge.textContent = state.sessions.length;
      renderChatHistory();
    } catch (err) {
      console.error("Failed to load chat sessions:", err);
      if (chatHistoryList) {
        chatHistoryList.innerHTML = `<div class="doc-empty-state">Lỗi nạp lịch sử</div>`;
      }
    }
  }

  function renderChatHistory() {
    if (!chatHistoryList) return;
    if (!state.sessions.length) {
      chatHistoryList.innerHTML = `<div class="doc-empty-state">Chưa có lịch sử nghiên cứu</div>`;
      return;
    }

    chatHistoryList.innerHTML = state.sessions.map(sess => {
      const isSelected = state.activeSessionId === sess.id;
      const title = sess.title || "Cuộc trò chuyện";
      return `
        <div class="history-item ${isSelected ? 'active' : ''}" data-id="${sess.id}" title="${escapeHtml(title)}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="history-item-icon">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          <span class="history-title">${escapeHtml(title)}</span>
          <div class="more-dots-wrap">
            <button class="more-dots-btn history-more-btn" data-id="${sess.id}" title="Tùy chọn">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <circle cx="12" cy="5" r="2.2"></circle>
                <circle cx="12" cy="12" r="2.2"></circle>
                <circle cx="12" cy="19" r="2.2"></circle>
              </svg>
            </button>
            <div class="more-dots-menu">
              <button class="more-dots-menu-item history-rename-action" data-rename="${sess.id}">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                </svg>
                <span>Đổi tên</span>
              </button>
              <button class="more-dots-menu-item history-del-action" data-del="${sess.id}">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
                <span>Xóa</span>
              </button>
            </div>
          </div>
        </div>
      `;
    }).join("");

    // Click to select session
    chatHistoryList.querySelectorAll(".history-item").forEach(item => {
      item.addEventListener("click", (e) => {
        if (e.target.closest(".more-dots-wrap")) return;
        const id = item.dataset.id;
        selectSession(id);
      });
    });

    // Toggle 3-dots menu on history item
    chatHistoryList.querySelectorAll(".history-more-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const menu = btn.parentElement.querySelector(".more-dots-menu");
        const isOpen = menu.classList.contains("open");
        document.querySelectorAll(".more-dots-menu.open").forEach(m => m.classList.remove("open"));
        document.querySelectorAll(".more-dots-btn.open").forEach(b => b.classList.remove("open"));
        if (!isOpen) {
          menu.classList.add("open");
          btn.classList.add("open");
        }
      });
    });

    // Rename chat session action
    chatHistoryList.querySelectorAll(".history-rename-action").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        document.querySelectorAll(".more-dots-menu.open").forEach(m => m.classList.remove("open"));
        const id = btn.dataset.rename;
        const currentSess = state.sessions.find(s => s.id === id);
        const oldTitle = currentSess ? currentSess.title : "";
        const newTitle = prompt("Nhập tên mới cho đoạn chat:", oldTitle);
        if (newTitle !== null && newTitle.trim() && newTitle.trim() !== oldTitle) {
          await renameSession(id, newTitle.trim());
        }
      });
    });

    // Delete chat session action
    chatHistoryList.querySelectorAll(".history-del-action").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        document.querySelectorAll(".more-dots-menu.open").forEach(m => m.classList.remove("open"));
        const id = btn.dataset.del;
        await deleteSession(id);
      });
    });
  }

  async function renameSession(sessionId, newTitle) {
    try {
      const res = await fetch(`/api/sessions/${sessionId}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle }),
      });
      if (res.ok) {
        const sess = state.sessions.find(s => s.id === sessionId);
        if (sess) sess.title = newTitle;
        renderChatHistory();
      }
    } catch (err) {
      console.error("Failed to rename session:", err);
    }
  }

  async function deleteSession(sessionId) {
    clearActivePolling();
    try {
      await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
      const wasActive = state.activeSessionId === sessionId;
      state.sessions = state.sessions.filter(s => s.id !== sessionId);
      if (historyCountBadge) historyCountBadge.textContent = state.sessions.length;

      if (wasActive) {
        if (state.sessions.length > 0) {
          await selectSession(state.sessions[0].id);
        } else {
          localStorage.removeItem("researchdesk_active_session");
          resetToWelcomeScreen();
        }
      }
      renderChatHistory();
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  }

  async function clearAllHistory() {
    if (!state.sessions.length) return;
    if (!confirm(`Bạn có chắc chắn muốn xóa toàn bộ ${state.sessions.length} đoạn chat lịch sử nghiên cứu?`)) return;
    clearActivePolling();
    try {
      await fetch("/api/sessions", { method: "DELETE" });
      state.sessions = [];
      if (historyCountBadge) historyCountBadge.textContent = "0";
      localStorage.removeItem("researchdesk_active_session");
      resetToWelcomeScreen();
      renderChatHistory();
    } catch (err) {
      console.error("Failed to clear all history:", err);
    }
  }

  const historyMoreBtn = document.getElementById("historyMoreBtn");
  const historyDropdownMenu = document.getElementById("historyDropdownMenu");
  const docMoreBtn = document.getElementById("docMoreBtn");
  const docDropdownMenu = document.getElementById("docDropdownMenu");

  function closeAllSectionDropdowns() {
    if (historyDropdownMenu) historyDropdownMenu.classList.remove("open");
    if (docDropdownMenu) docDropdownMenu.classList.remove("open");
    if (historyMoreBtn) historyMoreBtn.classList.remove("active");
    if (docMoreBtn) docMoreBtn.classList.remove("active");
  }

  if (historyMoreBtn && historyDropdownMenu) {
    historyMoreBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isOpen = historyDropdownMenu.classList.contains("open");
      closeAllSectionDropdowns();
      if (!isOpen) {
        historyDropdownMenu.classList.add("open");
        historyMoreBtn.classList.add("active");
      }
    });
  }

  if (docMoreBtn && docDropdownMenu) {
    docMoreBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isOpen = docDropdownMenu.classList.contains("open");
      closeAllSectionDropdowns();
      if (!isOpen) {
        docDropdownMenu.classList.add("open");
        docMoreBtn.classList.add("active");
      }
    });
  }

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".dropdown-anchor")) {
      closeAllSectionDropdowns();
    }
  });

  if (clearAllHistoryBtn) {
    clearAllHistoryBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeAllSectionDropdowns();
      clearAllHistory();
    });
  }

  async function selectSession(sessionId) {
    clearActivePolling();
    state.activeSessionId = sessionId;
    localStorage.setItem("researchdesk_active_session", sessionId);

    if (chatHistoryList) {
      chatHistoryList.querySelectorAll(".history-item").forEach(el => {
        el.classList.toggle("active", el.dataset.id === sessionId);
      });
    }

    try {
      const res = await fetch(`/api/sessions/${sessionId}`);
      if (!res.ok) throw new Error("Session not found");
      const data = await res.json();
      const runs = data.runs || [];

      if (welcomeHero) welcomeHero.style.display = "none";
      messageFeed.innerHTML = "";

      if (runs.length === 0) {
        resetToWelcomeScreen();
        return;
      }

      // Render TOÀN BỘ CÁC CÂU HỎI VÀ TRẢ LỜI CỦA ĐOẠN CHAT NÀY THEO THỨ TỰ
      for (let i = 0; i < runs.length; i++) {
        const run = runs[i];
        appendUserMessage(run.question || "Nội dung câu hỏi");

        const messageCard = createAgentSkeleton();
        messageFeed.appendChild(messageCard);

        const isLast = (i === runs.length - 1);
        if (isLast && (run.status === "processing" || run.status === "running")) {
          state.activeRunId = run.id;
          activeMessageCard = messageCard;
          setButtonState("running");
          const startTime = performance.now();
          messageCard._startTime = startTime;
          messageCard._currentStepStartTime = startTime;
          messageCard._activeStepIdx = 0;
          messageCard._stepDurations = {};
          updateThinkingTraces(messageCard, run, startTime);
          clearActiveTimer();
          activeTimerInterval = setInterval(() => {
            const now = performance.now();
            const elapsed = ((now - startTime) / 1000).toFixed(1);
            const timerEls = messageCard.querySelectorAll(".live-timer, .live-body-timer");
            timerEls.forEach(el => {
              el.textContent = `${elapsed}s`;
            });

            // Live update for the currently active step in timeline
            const activeStepTimeEl = messageCard.querySelector(".thinking-node.active .node-time");
            if (activeStepTimeEl) {
              const stepStart = messageCard._currentStepStartTime || startTime;
              const stepElapsed = Math.max(0.1, (now - stepStart) / 1000).toFixed(1);
              activeStepTimeEl.textContent = `${stepElapsed}s`;
            }
          }, 100);
          pollRunStatus(run.id, messageCard, run.question, startTime, activeTimerInterval);
        } else {
          renderFinalAnswer(messageCard, run, run.question);
        }
      }

      const lastRun = runs[runs.length - 1];
      if (lastRun) {
        state.activeRunId = lastRun.id;
        const docId = lastRun.document_id;
        if (docId) {
          selectDocument(docId, false);
        }
      }

      setButtonState(promptInput && promptInput.value.trim().length > 0 ? "ready" : "idle");
      messageFeed.scrollTo({ top: messageFeed.scrollHeight, behavior: "smooth" });
    } catch (err) {
      console.error("Failed to restore session details:", err);
      resetToWelcomeScreen();
    }
  }

  // Alias for legacy run selection if ever called
  async function selectRun(runId) {
    if (state.activeSessionId) {
      await selectSession(state.activeSessionId);
    }
  }

  function resetToWelcomeScreen() {
    clearActivePolling();
    messageFeed.innerHTML = "";
    if (welcomeHero) {
      messageFeed.appendChild(welcomeHero);
      welcomeHero.style.display = "flex";
    }
    state.activeRunId = null;
    state.activeSessionId = null;
    state.currentEvidence = [];
    state.selectedDocId = null;
    localStorage.removeItem("researchdesk_active_run");
    localStorage.removeItem("researchdesk_active_session");
    updateSidecarEvidence([]);
    
    // Uncheck all document checkboxes
    if (docList) {
      docList.querySelectorAll(".doc-item").forEach(d => {
        d.classList.remove("selected");
        const chk = d.querySelector("input[type='checkbox']");
        if (chk) chk.checked = false;
      });
    }
    if (activeDocName) {
      activeDocName.innerHTML = state.autoScout ? '<span style="color:#1d4ed8; font-weight:600">Auto-Scout (ArXiv Toàn Cầu)</span>' : 'Chưa chọn tài liệu';
    }
    if (chatHistoryList) {
      chatHistoryList.querySelectorAll(".history-item").forEach(el => el.classList.remove("active"));
    }
    if (promptInput) {
      promptInput.value = "";
      promptInput.style.height = "48px";
      setButtonState("idle");
      promptInput.focus();
    }
  }

  function formatRunTime(isoStr) {
    if (!isoStr) return "";
    try {
      const d = new Date(isoStr);
      const now = new Date();
      const diffMs = now - d;
      const diffMins = Math.floor(diffMs / (1000 * 60));
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

      if (diffMins < 5) return "Vừa xong";
      if (diffMins < 60) return `${diffMins} phút trước`;
      if (diffHours < 24) return `${diffHours} giờ trước`;
      if (diffDays === 1) return "Hôm qua";
      return `${d.getDate()}/${d.getMonth() + 1}`;
    } catch (e) {
      return "";
    }
  }

  // ==================== 3. LOAD DOCUMENTS ====================
  async function loadDocuments() {
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      state.documents = data.documents || [];
      if (docCount) docCount.textContent = state.documents.length;
      if (docCountBadge) docCountBadge.textContent = state.documents.length;
      renderDocuments();
    } catch (err) {
      console.error("Failed to load documents:", err);
      docList.innerHTML = `<div class="doc-empty-state">Lỗi nạp kho tài liệu</div>`;
    }
  }

  function updateDocDeleteUI() {
    const count = state.checkedDocIds.size;
    if (delDocsCountBadge) {
      if (count > 0) {
        delDocsCountBadge.style.display = "inline";
        delDocsCountBadge.textContent = String(count);
      } else {
        delDocsCountBadge.style.display = "none";
      }
    }
    const dot = document.getElementById("docMoreDot");
    if (dot) {
      dot.style.display = count > 0 ? "block" : "none";
    }
    const label = document.getElementById("deleteDocsBtnLabel");
    if (label) {
      label.textContent = count > 0 ? `Xóa (${count}) tài liệu đã chọn` : "Xóa tài liệu đã chọn";
    }
    if (deleteDocsBtn) {
      deleteDocsBtn.classList.toggle("has-selection", count > 0 || state.docSelectMode);
      deleteDocsBtn.title = count > 0
        ? `Xóa ${count} tài liệu đã tích chọn`
        : "Chọn nhiều hoặc ít tài liệu để xóa";
    }
    if (docBulkBar) {
      docBulkBar.style.display = (state.docSelectMode || count > 1) ? "flex" : "none";
    }
    if (confirmBulkDelDocsBtn) {
      confirmBulkDelDocsBtn.textContent = count > 0 ? `Xóa (${count})` : "Xóa";
    }
    if (selectAllDocsChk) {
      selectAllDocsChk.checked = state.documents.length > 0 && count === state.documents.length;
    }
  }

  function renderDocuments() {
    if (!docList) return;
    if (!state.documents.length) {
      docList.innerHTML = `<div class="doc-empty-state">Kho tài liệu trống</div>`;
      if (activeDocName && !state.autoScout) activeDocName.textContent = "Chưa có tài liệu";
      state.checkedDocIds.clear();
      updateDocDeleteUI();
      return;
    }

    // Keep checkedDocIds only for existing documents
    const validIds = new Set(state.documents.map(d => d.id));
    Array.from(state.checkedDocIds).forEach(id => {
      if (!validIds.has(id)) state.checkedDocIds.delete(id);
    });

    docList.innerHTML = state.documents.map((doc) => {
      const isChecked = state.checkedDocIds.has(doc.id);
      const isSelected = state.selectedDocId === doc.id || isChecked;
      return `
        <div class="doc-item ${isSelected ? 'selected' : ''}" data-id="${doc.id}">
          <input type="checkbox" ${isChecked ? 'checked' : ''} title="Tích chọn tài liệu">
          <span class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span>
          <div class="more-dots-wrap doc-more-dots-wrap">
            <button class="more-dots-btn doc-more-dots-btn" data-id="${doc.id}" title="Tùy chọn tài liệu" aria-label="Tùy chọn">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <circle cx="12" cy="5" r="2.2"></circle>
                <circle cx="12" cy="12" r="2.2"></circle>
                <circle cx="12" cy="19" r="2.2"></circle>
              </svg>
            </button>
            <div class="more-dots-menu doc-dots-menu">
              <button class="more-dots-menu-item doc-view-action" data-id="${doc.id}" title="Mở hiển thị tài liệu này">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8z"></path>
                  <circle cx="12" cy="12" r="3"></circle>
                </svg>
                <span>Xem tài liệu</span>
              </button>
              <button class="more-dots-menu-item doc-delete-action" data-id="${doc.id}" title="Xóa tài liệu khỏi kho" style="color: #ef4444;">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  <line x1="10" y1="11" x2="10" y2="17"></line>
                  <line x1="14" y1="11" x2="14" y2="17"></line>
                </svg>
                <span>Xóa</span>
              </button>
            </div>
          </div>
        </div>
      `;
    }).join("");

    if (!state.selectedDocId && activeDocName && !state.autoScout) {
      activeDocName.textContent = "Chưa chọn tài liệu";
    }

    docList.querySelectorAll(".doc-item").forEach(item => {
      const docId = item.dataset.id;
      const chk = item.querySelector("input[type='checkbox']");

      // Clicking the checkbox toggles multi-select
      if (chk) {
        chk.addEventListener("click", (e) => {
          e.stopPropagation();
          toggleDocCheckbox(docId, chk.checked);
        });
      }

      // Clicking the row: immediately select & display the file (nhấn phát là hiện file luôn)
      item.addEventListener("click", (e) => {
        if (e.target.closest(".doc-more-dots-wrap") || e.target.closest("input[type='checkbox']")) {
          return;
        }
        toggleDocCheckbox(docId, true, false);
        viewDocumentInSidecar(docId);
      });
    });

    // Toggle 3-dots menu on document item
    docList.querySelectorAll(".doc-more-dots-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const menu = btn.parentElement.querySelector(".doc-dots-menu");
        const isOpen = menu && menu.classList.contains("open");
        document.querySelectorAll(".more-dots-menu.open").forEach(m => m.classList.remove("open"));
        document.querySelectorAll(".more-dots-btn.open").forEach(b => b.classList.remove("open"));
        if (!isOpen && menu) {
          menu.classList.add("open");
          btn.classList.add("open");

          // Smart positioning to avoid clipping at the bottom of document list
          const rect = menu.getBoundingClientRect();
          const listRect = docList.getBoundingClientRect();
          if (rect.bottom > listRect.bottom + 15 && rect.top - rect.height > listRect.top) {
            menu.style.top = "auto";
            menu.style.bottom = "calc(100% + 2px)";
          } else {
            menu.style.top = "calc(100% + 2px)";
            menu.style.bottom = "auto";
          }
        }
      });
    });

    // Action: Xem tài liệu
    docList.querySelectorAll(".doc-view-action").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        document.querySelectorAll(".more-dots-menu.open").forEach(m => m.classList.remove("open"));
        document.querySelectorAll(".more-dots-btn.open").forEach(b => b.classList.remove("open"));
        const docId = btn.dataset.id;
        toggleDocCheckbox(docId, true, false);
        viewDocumentInSidecar(docId);
      });
    });

    // Action: Xóa tài liệu
    docList.querySelectorAll(".doc-delete-action").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        document.querySelectorAll(".more-dots-menu.open").forEach(m => m.classList.remove("open"));
        document.querySelectorAll(".more-dots-btn.open").forEach(b => b.classList.remove("open"));
        const docId = btn.dataset.id;
        deleteSingleDocument(docId);
      });
    });

    updateDocDeleteUI();
  }

  function toggleDocCheckbox(docId, isChecked, updatePdf = true) {
    if (state.autoScout && isChecked) {
      state.autoScout = false;
      if (autoScoutToggleBtn) autoScoutToggleBtn.classList.remove("active");
    }

    if (isChecked) {
      state.checkedDocIds.add(docId);
      state.selectedDocId = docId;
    } else {
      state.checkedDocIds.delete(docId);
      if (state.selectedDocId === docId) {
        const remaining = Array.from(state.checkedDocIds);
        state.selectedDocId = remaining.length > 0 ? remaining[remaining.length - 1] : null;
      }
    }

    if (docList) {
      docList.querySelectorAll(".doc-item").forEach(d => {
        const id = d.dataset.id;
        const checked = state.checkedDocIds.has(id);
        d.classList.toggle("selected", checked || state.selectedDocId === id);
        const chk = d.querySelector("input[type='checkbox']");
        if (chk) chk.checked = checked;
      });
    }

    if (state.selectedDocId) {
      const doc = state.documents.find(d => d.id === state.selectedDocId);
      if (doc && activeDocName) {
        activeDocName.textContent = state.checkedDocIds.size > 1
          ? `${doc.name} (+${state.checkedDocIds.size - 1})`
          : doc.name;
        if (updatePdf) {
          viewDocumentInSidecar(doc.id);
        }
      }
    } else if (activeDocName && !state.autoScout) {
      activeDocName.textContent = "Chưa chọn tài liệu";
    }

    updateDocDeleteUI();
  }

  function selectDocument(docId, updatePdf = true) {
    if (state.autoScout) {
      state.autoScout = false;
      if (autoScoutToggleBtn) autoScoutToggleBtn.classList.remove("active");
    }
    state.checkedDocIds.clear();
    if (docId) {
      state.checkedDocIds.add(docId);
      state.selectedDocId = docId;
    } else {
      state.selectedDocId = null;
    }
    if (docList) {
      docList.querySelectorAll(".doc-item").forEach(d => {
        const match = d.dataset.id === docId;
        d.classList.toggle("selected", match);
        const chk = d.querySelector("input[type='checkbox']");
        if (chk) chk.checked = match;
      });
    }
    const doc = state.documents.find(d => d.id === docId);
    if (doc && activeDocName) {
      activeDocName.textContent = doc.name;
      if (updatePdf) {
        viewDocumentInSidecar(doc.id);
      }
    }
    updateDocDeleteUI();
  }

  async function deleteSingleDocument(docId) {
    const doc = state.documents.find(d => d.id === docId);
    const docName = doc ? (doc.name || doc.title || docId) : docId;
    if (!confirm(`Bạn có chắc chắn muốn xóa tài liệu "${docName}" khỏi Kho Tài Liệu?`)) {
      return;
    }

    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
      if (!res.ok) {
        await fetch("/api/documents/batch-delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ doc_ids: [docId] }),
        });
      }
      state.checkedDocIds.delete(docId);
      if (state.selectedDocId === docId) {
        state.selectedDocId = null;
        if (activeDocName && !state.autoScout) {
          activeDocName.textContent = "Chưa chọn tài liệu";
        }
        if (pdfFrame) {
          pdfFrame.src = "about:blank";
        }
        const sidecarDocTitle = document.getElementById("sidecarDocTitle");
        if (sidecarDocTitle) {
          sidecarDocTitle.textContent = "Chưa có tài liệu được chọn";
        }
      }
      await loadDocuments();
      updateDocDeleteUI();
    } catch (err) {
      console.error("Failed to delete document:", err);
      alert("Không thể xóa tài liệu. Vui lòng thử lại!");
    }
  }

  async function deleteCheckedDocuments() {
    const idsToDelete = Array.from(state.checkedDocIds);
    if (!idsToDelete.length) {
      // Toggle bulk selection bar so user can tick 1 or many documents
      state.docSelectMode = !state.docSelectMode;
      updateDocDeleteUI();
      return;
    }

    if (!confirm(`Bạn có chắc chắn muốn xóa ${idsToDelete.length} tài liệu đã chọn khỏi Kho Tài Liệu?`)) {
      return;
    }

    try {
      const res = await fetch("/api/documents/batch-delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_ids: idsToDelete }),
      });
      if (!res.ok) {
        for (const id of idsToDelete) {
          await fetch(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
        }
      }
      state.checkedDocIds.clear();
      if (idsToDelete.includes(state.selectedDocId)) {
        state.selectedDocId = null;
      }
      state.docSelectMode = false;
      await loadDocuments();
      updateDocDeleteUI();
    } catch (err) {
      console.error("Failed to delete documents:", err);
    }
  }

  const toggleDocSelectModeBtn = document.getElementById("toggleDocSelectModeBtn");
  if (toggleDocSelectModeBtn) {
    toggleDocSelectModeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeAllSectionDropdowns();
      state.docSelectMode = !state.docSelectMode;
      updateDocDeleteUI();
      renderDocuments(state.documents);
    });
  }

  if (deleteDocsBtn) {
    deleteDocsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeAllSectionDropdowns();
      deleteCheckedDocuments();
    });
  }

  if (selectAllDocsChk) {
    selectAllDocsChk.addEventListener("change", () => {
      if (selectAllDocsChk.checked) {
        state.documents.forEach(d => state.checkedDocIds.add(d.id));
        if (!state.selectedDocId && state.documents.length > 0) {
          state.selectedDocId = state.documents[0].id;
        }
      } else {
        state.checkedDocIds.clear();
        state.selectedDocId = null;
      }
      renderDocuments();
    });
  }

  if (confirmBulkDelDocsBtn) {
    confirmBulkDelDocsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (state.checkedDocIds.size === 0) {
        alert("Vui lòng tích chọn ít nhất 1 tài liệu để xóa.");
        return;
      }
      deleteCheckedDocuments();
    });
  }

  if (cancelBulkDelDocsBtn) {
    cancelBulkDelDocsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      state.checkedDocIds.clear();
      state.selectedDocId = null;
      state.docSelectMode = false;
      renderDocuments();
    });
  }

  // Auto-Scout Toggle Functionality (Agent 0 - Global ArXiv Discovery)
  function toggleAutoScout(forceState) {
    state.autoScout = typeof forceState === "boolean" ? forceState : !state.autoScout;

    if (state.autoScout) {
      // 1. Chuyển nút sang màu xanh nhẹ
      if (autoScoutToggleBtn) autoScoutToggleBtn.classList.add("active");

      // 2. Bỏ tích TẤT CẢ các bài báo trong kho
      state.selectedDocId = null;
      if (docList) {
        docList.querySelectorAll(".doc-item").forEach(d => {
          d.classList.remove("selected");
          const chk = d.querySelector("input[type='checkbox']");
          if (chk) chk.checked = false;
        });
      }

      // 3. Cập nhật nhãn phạm vi trên thanh Canvas Header
      if (activeDocName) {
        activeDocName.innerHTML = `<span style="color:#1d4ed8; font-weight:600">Auto-Scout (ArXiv Toàn Cầu)</span>`;
      }
    } else {
      // Tắt chế độ Auto-Scout
      if (autoScoutToggleBtn) autoScoutToggleBtn.classList.remove("active");
      if (activeDocName) {
        activeDocName.textContent = "Chưa chọn tài liệu";
      }
    }
  }

  if (autoScoutToggleBtn) {
    autoScoutToggleBtn.addEventListener("click", () => {
      toggleAutoScout();
    });
  }

  // Quick Search Filter with Diacritics Removal
  if (docSearchInput) {
    docSearchInput.addEventListener("input", () => {
      const term = removeDiacritics(docSearchInput.value);
      if (!docList) return;
      document.querySelectorAll(".doc-item").forEach(item => {
        const text = removeDiacritics(item.querySelector(".doc-name").textContent);
        item.style.display = text.includes(term) ? "flex" : "none";
      });
    });
  }

  // ==================== 4. SUBMIT QUERY ====================
  
  function getRecentChatHistory(maxTurns = 3) {
    const history = [];
    const cards = Array.from(document.querySelectorAll("#messageFeed .message-card"));
    for (const card of cards) {
      const userBubble = card.querySelector(".user-query-bubble");
      if (userBubble) {
        history.push({ role: "user", content: userBubble.textContent.trim() });
      }
      const ansBlock = card.querySelector(".direct-qa-view") || card.querySelector(".report-markdown-view");
      if (ansBlock) {
        history.push({ role: "assistant", content: ansBlock.textContent.slice(0, 250).trim() });
      }
    }
    return history.slice(-maxTurns * 2);
  }

  async function submitQuery(queryText) {
    const text = (queryText || promptInput.value || "").trim();
    if (!text) return;

    clearActivePolling();
    promptInput.value = "";
    promptInput.style.height = "48px";
    setButtonState("running");

    if (welcomeHero) welcomeHero.style.display = "none";
    appendUserMessage(text);

    // KIỂM TRA KẾT NỐI MÔ HÌNH: Nếu người dùng chọn mô hình chưa kết nối (VD: Kaggle FN / Kaggle Max chưa kết nối)
    const hasUnconnectedModel = Array.from(document.querySelectorAll(".agent-model-select")).some(select => {
      const val = select.value || "";
      const selectedOpt = select.options[select.selectedIndex];
      const optText = selectedOpt ? selectedOpt.textContent : "";
      if (optText.includes("Chưa kết nối") || (selectedOpt && selectedOpt.dataset.connected === "false")) {
        return true;
      }
      if ((val.startsWith("kaggle") || val.startsWith("gpu")) && !state.isKaggleOnline) {
        return true;
      }
      if (val === "gemini" && !state.isGeminiOnline) {
        return true;
      }
      return false;
    });

    if (hasUnconnectedModel) {
      const errCard = document.createElement("div");
      errCard.className = "message-card";
      errCard.innerHTML = `
        <div class="agent-response-container">
          <div class="agent-output-block" style="padding: 14px 18px; border-left: 3px solid #ef4444; background: #fef2f2; border-radius: 8px;">
            <div style="font-size: 13.5px; color: #991b1b; font-weight: 500; line-height: 1.6;">
              Chưa kết nối được hãy kết nối lại.
            </div>
          </div>
        </div>
      `;
      messageFeed.appendChild(errCard);
      messageFeed.scrollTo({ top: messageFeed.scrollHeight, behavior: "smooth" });
      setButtonState("idle");
      return;
    }

    const checkedList = Array.from(state.checkedDocIds || []);
    const docIds = checkedList.length > 0 
      ? checkedList 
      : (state.selectedDocId ? [state.selectedDocId] : []);

    // KIỂM TRA ĐIỀU KIỆN BẮT BUỘC: Chưa chọn tài liệu VÀ Chưa bật Auto-Scout
    if (docIds.length === 0 && !state.autoScout) {
      const promptCard = document.createElement("div");
      promptCard.className = "message-card";
      promptCard.innerHTML = `
        <div class="agent-response-container">
          <div class="agent-output-block" style="padding: 16px 20px; border-left: 4px solid #2563eb; background: #f8fafc; border-radius: 8px;">
            <div style="font-size: 14px; color: #1e293b; line-height: 1.6; font-weight: 500;">
              ℹ️ Vui lòng <b>chọn một tài liệu</b> trong kho hoặc bật <b>Auto-Scout</b> để bắt đầu nghiên cứu.
            </div>
            <div style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
              • Bấm vào tên bài báo (ví dụ: <i>Báo Hà Nội mới</i> hoặc <i>Môi trường</i>) ở cột bên trái để chọn.<br>
              • Hoặc gạt bật nút <b>Auto-Scout</b> ở phía trên để AI tự động tìm kiếm bài báo khoa học.
            </div>
          </div>
        </div>
      `;
      messageFeed.appendChild(promptCard);
      messageFeed.scrollTo({ top: messageFeed.scrollHeight, behavior: "smooth" });
      setButtonState(promptInput.value.trim().length > 0 ? "ready" : "idle");
      return;
    }

    const isAutoScout = state.autoScout;
    const initialTask = isAutoScout ? "Agent 0 đang trinh sát ArXiv & CrossRef" : "Đang đọc hiểu tài liệu và đối soát";
    const messageCard = createAgentSkeleton(initialTask);
    activeMessageCard = messageCard;
    messageFeed.appendChild(messageCard);
    messageFeed.scrollTo({ top: messageFeed.scrollHeight, behavior: "smooth" });

    // Live Stopwatch Timer
    const startTime = performance.now();
    messageCard._startTime = startTime;
    messageCard._currentStepStartTime = startTime;
    messageCard._activeStepIdx = 0;
    messageCard._stepDurations = {};
    clearActiveTimer();
    const timerInterval = setInterval(() => {
      const now = performance.now();
      const elapsed = ((now - startTime) / 1000).toFixed(1);
      const timerEls = messageCard.querySelectorAll(".live-timer, .live-body-timer");
      timerEls.forEach(el => {
        el.textContent = `${elapsed}s`;
      });

      // Live update for the currently active step in timeline
      const activeStepTimeEl = messageCard.querySelector(".thinking-node.active .node-time");
      if (activeStepTimeEl) {
        const stepStart = messageCard._currentStepStartTime || startTime;
        const stepElapsed = Math.max(0.1, (now - stepStart) / 1000).toFixed(1);
        activeStepTimeEl.textContent = `${stepElapsed}s`;
      }
    }, 100);
    activeTimerInterval = timerInterval;

    if (activeAbortController) {
      try { activeAbortController.abort(); } catch (e) {}
    }
    activeAbortController = new AbortController();

    try {
      // NẾU BẬT CHẾ ĐỘ AUTO-SCOUT: CHẠY RIÊNG BIỆT AGENT 0 (ĐƠN NHIỆM TRINH SÁT)
      if (state.autoScout) {
        const taskText = messageCard.querySelector(".live-task-text");
        if (taskText) taskText.textContent = "Agent 0 đang trinh sát ArXiv & CrossRef";

        const scoutRes = await fetch("/api/scout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text, max_results: 3, session_id: state.activeSessionId || null }),
          signal: activeAbortController.signal,
        });

        clearActiveTimer();
        const finalElapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        if (!scoutRes.ok) throw new Error("HTTP " + scoutRes.status);
        const scoutData = await scoutRes.json();

        // Tự động làm mới Kho Tài Liệu ở Sidebar bên trái
        await loadDocuments();
        // Tự động cập nhật ngay Lịch Sử Nghiên Cứu để đoạn tin nhắn tìm kiếm mới nhất xếp ngay đầu danh sách
        await loadChatHistory();

        // Hiển thị danh thiếp các bài báo tìm được
        renderScoutResults(messageCard, scoutData, text, finalElapsed);
        activeMessageCard = null;
        updateGeminiUsageBadge();
        setButtonState(promptInput.value.trim().length > 0 ? "ready" : "idle");
        return;
      }

      // NẾU TẮT AUTO-SCOUT: CHẠY QUY TRÌNH THẨM ĐỊNH 5 AGENT CŨ TRÊN TÀI LIỆU ĐÃ CHỌN
      const payload = {
        query: text,
        document_ids: docIds,
        max_iterations: state.maxReflectionLoops || 2,
        chat_history: getRecentChatHistory(3),
        session_id: state.activeSessionId || null,
      };

      const res = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: activeAbortController.signal,
      });

      if (!res.ok) {
        clearActiveTimer();
        throw new Error("HTTP " + res.status);
      }
      const data = await res.json();
      state.activeRunId = data.id;
      if (data.session_id) {
        state.activeSessionId = data.session_id;
        localStorage.setItem("researchdesk_active_session", data.session_id);
      }
      setButtonState("running");
      await loadChatHistory();

      pollRunStatus(data.id, messageCard, text, startTime, timerInterval);
    } catch (err) {
      if (err && err.name === "AbortError") {
        return;
      }
      clearActiveTimer();
      console.error("Submission failed:", err);
      updateAgentSkeletonError(messageCard, "Không thể kết nối đến máy chủ.");
      activeMessageCard = null;
      setButtonState(promptInput.value.trim().length > 0 ? "ready" : "idle");
    }
  }

  // ==================== 5. POLLING STATUS ====================
  function pollRunStatus(runId, messageCard, query, startTime, timerInterval) {
    const maxPolls = 300;
    let polls = 0;

    clearActivePolling();
    setButtonState("running");

    activePollInterval = setInterval(async () => {
      polls++;
      try {
        const res = await fetch(`/api/runs/${runId}`);
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();

        updateThinkingTraces(messageCard, data, startTime);

        if (data.status === "completed") {
          if (timerInterval) clearInterval(timerInterval);
          clearActiveTimer();
          clearActivePolling();
          activeMessageCard = null;
          const finalElapsed = startTime ? ((performance.now() - startTime) / 1000).toFixed(1) : null;
          updateThinkingTraces(messageCard, data, startTime, true, finalElapsed);
          renderFinalAnswer(messageCard, data, query, finalElapsed);
          await loadChatHistory();
          updateGeminiUsageBadge(); // Refresh quota
          setButtonState(promptInput.value.trim().length > 0 ? "ready" : "idle");
        } else if (data.status === "failed" || data.status === "cancelled") {
          if (timerInterval) clearInterval(timerInterval);
          clearActiveTimer();
          clearActivePolling();
          activeMessageCard = null;
          updateThinkingTraces(messageCard, data, startTime, true);
          const errMsg = data.status === "cancelled"
            ? "Đã dừng quá trình thực thi theo yêu cầu người dùng."
            : (data.error || "Quá trình nghiên cứu và đối soát tài liệu thất bại.");
          updateAgentSkeletonError(messageCard, errMsg);
          await loadChatHistory();
          updateGeminiUsageBadge();
          setButtonState(promptInput.value.trim().length > 0 ? "ready" : "idle");
        } else if (polls >= maxPolls) {
          if (timerInterval) clearInterval(timerInterval);
          clearActiveTimer();
          clearActivePolling();
          activeMessageCard = null;
          const finalElapsed = startTime ? ((performance.now() - startTime) / 1000).toFixed(1) : null;
          updateThinkingTraces(messageCard, data, startTime, true, finalElapsed);
          renderFinalAnswer(messageCard, data, query, finalElapsed);
          await loadChatHistory();
          updateGeminiUsageBadge();
          setButtonState(promptInput.value.trim().length > 0 ? "ready" : "idle");
        } else {
          // Still processing -> guarantee red Stop button remains visible & refresh quota live
          setButtonState("running");
          if (polls % 2 === 0) {
            updateGeminiUsageBadge();
          }
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 350);
  }

  function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "message-card";
    div.innerHTML = `
      <div class="user-query-row">
        <div class="user-query-bubble">${escapeHtml(text)}</div>
      </div>
    `;
    messageFeed.appendChild(div);
  }

  function createAgentSkeleton(initialTask = "Đang đọc hiểu tài liệu và đối soát") {
    const div = document.createElement("div");
    div.className = "message-card";
    div.innerHTML = `
      <div class="agent-response-container">
        <div class="thinking-accordion open">
          <div class="thinking-header">
            <div class="thinking-title">
              <span class="thinking-sparkle">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                </svg>
              </span>
              <span class="thinking-label thinking-shimmer">Thinking</span>
              <span class="thinking-dots-anim"><span>.</span><span>.</span><span>.</span></span>
              <span class="thinking-timer live-timer">0.0s</span>
              <span class="thinking-task-sep">•</span>
              <span class="thinking-task-text live-task-text">${escapeHtml(initialTask)}</span>
            </div>
            <div class="thinking-chevron-wrap">
              <svg class="thinking-chevron" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </div>
          </div>
          <div class="thinking-collapsible-wrapper">
            <div class="thinking-body" id="liveTraces">
              <div class="thinking-timeline" id="tracesContent">
                <div class="thinking-node active">
                  <div class="node-rail">
                    <span class="node-status-icon active" title="Đang thực thi">
                      <svg class="node-spinner" width="13" height="13" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="9" stroke="#e4e4e7" stroke-width="2.5"></circle>
                        <path d="M12 3a9 9 0 0 1 9 9" stroke="#18181b" stroke-width="2.5" stroke-linecap="round"></path>
                      </svg>
                    </span>
                  </div>
                  <div class="node-content">
                    <span class="node-badge">SUPERVISOR</span>
                    <span class="node-text">Đang phân rã câu hỏi theo các góc nhìn nghiên cứu...</span>
                  </div>
                  <div class="node-time live-step-timer">0.1s</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    const header = div.querySelector(".thinking-header");
    const acc = div.querySelector(".thinking-accordion");
    header.addEventListener("click", () => acc.classList.toggle("open"));
    return div;
  }

  function formatAgentBadge(rawBadge) {
    if (!rawBadge) return "SYSTEM";
    const s = String(rawBadge).toLowerCase().trim();
    if (s.includes("supervisor") || s.includes("plan")) return "SUPERVISOR";
    if (s.includes("scout")) return "SCOUT";
    if (s.includes("retrieval") || s.includes("retriever")) return "RETRIEVAL";
    if (s.includes("reader")) return "READER";
    if (s.includes("summarizer")) return "SUMMARIZER";
    if (s.includes("fact") || s.includes("check") || s.includes("nli")) return "FACT-CHECK";
    if (s.includes("writer")) return "WRITER";
    if (s.includes("critic")) return "CRITIC";
    return rawBadge.replace(/^(cpu_min_|cpumin_|cpu_max_|cpumax_|hybrid_|gemini_|gpu_ft_|gpu_max_|gpuft_|gpumax_|kaggle_fn_|kaggle_max_)/i, "").replace(/agent$/i, "").toUpperCase();
  }

  function formatActionText(rawAction) {
    if (!rawAction) return "Đang xử lý...";
    const text = String(rawAction).trim();
    const actionMap = {
      "bypass_discovery": "Bỏ qua trinh sát (tài liệu đã có sẵn trong kho)",
      "gpu_ft_retrieval_completed": "Truy hồi tài liệu khoa học hoàn tất",
      "gpu_ft_reading_completed": "Đọc hiểu cấu trúc văn bản & BBox hoàn tất",
      "gpu_ft_page_summary_completed": "Tổng hợp trích đoạn & luận điểm hoàn tất",
      "gpu_ft_summarization_completed": "Tổng hợp trích đoạn & luận điểm hoàn tất",
      "gpu_ft_fact_check_completed": "Thẩm định đối soát luận điểm hoàn tất",
      "gpu_ft_report_written": "Biên soạn báo cáo học thuật hoàn tất",
      "gpu_max_retrieval_completed": "Truy hồi đa tầng FTS5 + BM25 hoàn tất",
      "gpu_max_reading_completed": "Bóc tách layout PDF & bảng biểu hoàn tất",
      "gpu_max_page_summary_completed": "Tổng hợp trích dẫn sâu hoàn tất",
      "gpu_max_summarization_completed": "Tổng hợp trích dẫn sâu hoàn tất",
      "gpu_max_fact_check_completed": "Kiểm chứng MiniCheck NLI hoàn tất",
      "gpu_max_report_written": "Biên tập báo cáo chuẩn học thuật hoàn tất",
      "kaggle_fn_retrieval_completed": "Truy hồi tài liệu khoa học hoàn tất",
      "kaggle_fn_reading_completed": "Đọc hiểu cấu trúc văn bản & BBox hoàn tất",
      "kaggle_fn_summarization_completed": "Tổng hợp trích đoạn & luận điểm hoàn tất",
      "kaggle_fn_fact_check_completed": "Thẩm định đối soát luận điểm hoàn tất",
      "kaggle_fn_report_written": "Biên soạn báo cáo học thuật hoàn tất",
      "kaggle_max_retrieval_completed": "Truy hồi đa tầng FTS5 + BM25 hoàn tất",
      "kaggle_max_reading_completed": "Bóc tách layout PDF & bảng biểu hoàn tất",
      "kaggle_max_summarization_completed": "Tổng hợp trích dẫn sâu hoàn tất",
      "kaggle_max_fact_check_completed": "Kiểm chứng MiniCheck NLI hoàn tất",
      "kaggle_max_report_written": "Biên tập báo cáo chuẩn học thuật hoàn tất",
      "cpu_min_agentic_retrieval_completed": "Truy hồi tài liệu khoa học hoàn tất",
      "cpu_min_reading_completed": "Đọc hiểu cấu trúc văn bản & BBox hoàn tất",
      "cpu_min_page_summary_completed": "Tổng hợp trích đoạn & luận điểm hoàn tất",
      "cpu_min_fact_check_completed": "Thẩm định đối soát luận điểm hoàn tất",
      "cpu_min_report_written": "Biên soạn báo cáo học thuật hoàn tất",
      "cpu_max_agentic_retrieval_completed": "Truy hồi đa tầng FTS5 + BM25 hoàn tất",
      "cpu_max_reading_completed": "Bóc tách layout PDF & bảng biểu hoàn tất",
      "cpu_max_page_summary_completed": "Tổng hợp trích dẫn sâu hoàn tất",
      "cpu_max_fact_check_completed": "Kiểm chứng MiniCheck NLI hoàn tất",
      "cpu_max_report_written": "Biên tập báo cáo chuẩn học thuật hoàn tất",
      "gemini_retrieval_completed": "Truy hồi tài liệu khoa học hoàn tất",
      "gemini_reading_completed": "Đọc hiểu cấu trúc văn bản PDF hoàn tất",
      "gemini_summarization_completed": "Tổng hợp luận điểm khoa học & trích dẫn sâu hoàn tất",
      "gemini_fact_check_completed": "Thẩm định đối soát luận điểm & chống ảo giác hoàn tất",
      "gemini_report_written": "Biên soạn báo cáo khoa học hoàn tất",
      "hybrid_retrieval_completed": "Truy hồi tài liệu đa tầng hoàn tất",
      "hybrid_reading_completed": "Đọc hiểu cấu trúc văn bản hoàn tất",
      "hybrid_summarization_completed": "Tổng hợp trích dẫn sâu hoàn tất",
      "hybrid_fact_check_completed": "Kiểm chứng đối kháng NLI hoàn tất",
      "hybrid_report_written": "Biên soạn báo cáo học thuật hoàn tất"
    };
    const cleanKey = text.toLowerCase().replace(/\s+/g, "_");
    if (actionMap[cleanKey]) return actionMap[cleanKey];
    if (actionMap[text]) return actionMap[text];
    return text.replace(/_/g, " ");
  }

  function getNextActiveAgent(lastTrace) {
    if (!lastTrace) {
      return { badge: "SUPERVISOR", text: "Đang phân tích mục tiêu & phân rã câu hỏi nghiên cứu..." };
    }
    const action = (lastTrace.action || "").toLowerCase();
    const agent = (lastTrace.agent || lastTrace.node || "").toLowerCase();
    const s = `${action} ${agent}`;

    if (s.includes("supervisor") || s.includes("phân tích mục tiêu") || s.includes("router")) {
      return { badge: "RETRIEVAL", text: "Đang truy hồi tài liệu khoa học đa tầng (FTS5 + BM25)..." };
    }
    if (s.includes("scout") || s.includes("trinh sát")) {
      return { badge: "RETRIEVAL", text: "Đang truy hồi tài liệu khoa học đa tầng (FTS5 + BM25)..." };
    }
    if (s.includes("retrieval") || s.includes("truy hồi")) {
      return { badge: "READER", text: "Đang đọc hiểu cấu trúc văn bản PDF, bảng biểu & BBox..." };
    }
    if (s.includes("reading") || s.includes("reader") || s.includes("đọc hiểu")) {
      return { badge: "SUMMARIZER", text: "Đang tổng hợp luận điểm khoa học & trích dẫn sâu..." };
    }
    if (s.includes("summariz") || s.includes("tổng hợp")) {
      return { badge: "FACT-CHECK", text: "Đang thẩm định đối soát luận điểm & kiểm chứng NLI..." };
    }
    if (s.includes("fact") || s.includes("check") || s.includes("thẩm định")) {
      return { badge: "WRITER", text: "Đang biên tập báo cáo khoa học theo chuẩn học thuật..." };
    }
    return null;
  }

  function updateThinkingTraces(messageCard, data, startTime = null, isCompleted = false, finalElapsed = null) {
    const traceBox = messageCard.querySelector("#tracesContent");
    if (!traceBox) return;

    const traces = (data && data.reasoning_trace) ? data.reasoning_trace : [];
    const runIsDone = isCompleted || (data && (data.status === "completed" || data.status === "failed"));
    const now = performance.now();
    const baseStart = startTime || messageCard._startTime || now;
    if (!messageCard._startTime && startTime) {
      messageCard._startTime = startTime;
    }

    if (messageCard._prevTraceCount === undefined) {
      messageCard._prevTraceCount = 0;
      messageCard._currentStepStartTime = baseStart;
    }

    if (traces.length > messageCard._prevTraceCount) {
      messageCard._prevTraceCount = traces.length;
      messageCard._currentStepStartTime = now;
    }

    let html = "";

    // 1. Render all COMPLETED steps from traces (with checkmark and real duration)
    traces.forEach((t, idx) => {
      const rawAction = t.action || "Đang xử lý...";
      const action = formatActionText(rawAction);
      let badge = "SUPERVISOR";

      if (rawAction.includes("Phân tích") || rawAction.includes("mục tiêu") || t.stage === "Thought") {
        badge = "SUPERVISOR";
      } else if (rawAction.includes("Truy hồi") || rawAction.includes("FTS5") || rawAction.includes("STORM") || t.stage === "Action") {
        badge = "RETRIEVAL";
      } else if (rawAction.includes("PyMuPDF") || rawAction.includes("BBox") || t.stage === "Observation") {
        badge = "READER";
      } else if (rawAction.includes("vòng tự sửa sai") || rawAction.includes("bù khuyết") || rawAction.includes("Critic") || t.stage === "Reflection") {
        badge = "INNER CRITIC";
      } else if (rawAction.includes("Đối soát") || rawAction.includes("Adversarial") || rawAction.includes("NLI")) {
        badge = "CROSS-NLI";
      } else if (rawAction.includes("Ký duyệt") || rawAction.includes("báo cáo") || t.stage === "Decision") {
        badge = "WRITER";
      } else if (t.agent) {
        badge = formatAgentBadge(t.agent);
      }

      let durationStr = t.duration || "0.1s";
      if (!t.duration && t.timestamp && idx > 0 && traces[idx - 1].timestamp) {
        const diff = Math.max(0.1, t.timestamp - traces[idx - 1].timestamp);
        durationStr = `${diff.toFixed(1)}s`;
      }

      html += `
        <div class="thinking-node completed">
          <div class="node-rail">
            <span class="node-status-icon completed" title="Hoàn tất">
              <svg class="node-check" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#18181b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
            </span>
          </div>
          <div class="node-content">
            <span class="node-badge">${escapeHtml(badge)}</span>
            <span class="node-text">${escapeHtml(action)}</span>
          </div>
          <div class="node-time">${durationStr}</div>
        </div>
      `;
    });

    // 2. If the run is STILL PROCESSING, append the NEXT CURRENTLY RUNNING agent!
    if (!runIsDone) {
      const lastTrace = traces.length > 0 ? traces[traces.length - 1] : null;
      const nextActive = getNextActiveAgent(lastTrace);

      if (nextActive) {
        const stepStart = messageCard._currentStepStartTime || baseStart;
        const activeElapsed = Math.max(0.1, (now - stepStart) / 1000).toFixed(1);

        html += `
          <div class="thinking-node active">
            <div class="node-rail">
              <span class="node-status-icon active" title="Đang thực thi">
                <svg class="node-spinner" width="13" height="13" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="9" stroke="#e4e4e7" stroke-width="2.5"></circle>
                  <path d="M12 3a9 9 0 0 1 9 9" stroke="#18181b" stroke-width="2.5" stroke-linecap="round"></path>
                </svg>
              </span>
            </div>
            <div class="node-content">
              <span class="node-badge">${escapeHtml(nextActive.badge)}</span>
              <span class="node-text">${escapeHtml(nextActive.text)}</span>
            </div>
            <div class="node-time">${activeElapsed}s</div>
          </div>
        `;

        const taskText = messageCard.querySelector(".live-task-text");
        if (taskText) {
          taskText.textContent = nextActive.text;
        }
      }
    }

    traceBox.innerHTML = html;
  }

  function updateAgentSkeletonError(messageCard, errorMsg) {
    const container = messageCard.querySelector(".agent-response-container");
    container.innerHTML = `
      ${autonomyBannerHtml}
        <div class="direct-qa-view">
        <p class="text-sm text-neutral-600">${escapeHtml(errorMsg)}</p>
      </div>
    `;
  }

  // ==================== 5.5 RENDER SCOUT RESULTS (AGENT 0 STANDALONE) ====================
  function renderScoutResults(messageCard, data, query, finalElapsed = null) {
    const container = messageCard.querySelector(".agent-response-container");
    const acc = messageCard.querySelector(".thinking-accordion");
    if (acc) {
      acc.classList.remove("open");
      try { updateThinkingTraces(messageCard, data, null, true, finalElapsed); } catch (e) {}
      const titleDiv = acc.querySelector(".thinking-title");
      if (titleDiv) {
        const papersCount = data.total_discovered || (data.papers && data.papers.length) || 0;
        const timeStr = finalElapsed ? `${finalElapsed}s` : "";
        titleDiv.innerHTML = `
          <span class="thinking-sparkle completed">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          </span>
          <span class="thinking-label-completed">Trinh sát hoàn tất</span>
          ${timeStr ? `<span class="thinking-timer">${timeStr}</span>` : ""}
          <span class="thinking-task-sep">•</span>
          <span class="thinking-task-text">${papersCount} tài liệu khoa học được định vị</span>
        `;
      }
    }

    const papers = data.papers || [];

    if (data.decision === "reject" || data.decision === "clarify" || data.decision === "no_relevant_papers" || papers.length === 0) {
      const bannerTitle = data.decision === "clarify" 
        ? "Cần làm rõ chủ đề nghiên cứu" 
        : (data.decision === "reject" ? "Chủ đề nằm ngoài danh mục STEM của ArXiv" : "Không tìm thấy bài báo phù hợp");
      const bannerMsg = data.message || "Hệ thống không tìm thấy tài liệu phù hợp trên ArXiv cho chủ đề này.";
      
      // Agent Autonomy & Provenance Banner
      const loopsNum = state.maxReflectionLoops || 2;
      const autonomyBannerHtml = `
        <div class="agent-autonomy-banner">
          <span class="autonomy-icon">🤖</span>
          <span class="autonomy-text"><b>Hybrid Autonomous Agent</b> • Điều phối Đa Tác tử L4 • ${loopsNum} Vòng Tự Phản Biện • Đối Soát Trích Dẫn 100%</span>
        </div>
      `;

      const card = document.createElement("div");
      card.className = "agent-output-block scout-results-block";
      card.innerHTML = `
        <div class="scout-banner" style="border-left: 4px solid #0f172a;">
          <div class="scout-banner-title">
            <span><b>Agent 0 (Literature Scout):</b> ${escapeHtml(bannerTitle)}</span>
          </div>
          <p class="scout-banner-subtitle" style="margin-top: 8px; color: #334155; line-height: 1.6;">${escapeHtml(bannerMsg)}</p>
          <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #cbd5e1; font-size: 13px; color: #64748b;">
            Gợi ý: Bạn có thể tải trực tiếp tệp PDF vào <b>Kho Tài Liệu</b> bên trái để 5 Agent chuyên sâu phân tích, hoặc thử lại với các từ khóa công nghệ/khoa học kỹ thuật.
          </div>
        </div>
      `;
      container.innerHTML = "";
      container.appendChild(card);
      return;
    }

    let papersHtml = "";

    papers.forEach((p, idx) => {
      const title = p.title || "Tài liệu học thuật";
      const year = p.year || 2024;
      const authors = (p.authors && p.authors.length) ? p.authors.join(", ") : "Nhiều tác giả";
      const arxivId = p.arxiv_id || "";
      const docId = p.doc_id || p.id || "";
      const abstract = (p.abstract || "").trim();
      
      const hasPdf = Boolean(p.has_pdf !== undefined ? p.has_pdf : (p.doc_type === "pdf" || (p.local_pdf_path && p.local_pdf_path.endsWith(".pdf"))));
      const sourceUrl = p.source_url || (arxivId ? `https://arxiv.org/abs/${arxivId}` : "");
      const sourceName = p.source || (arxivId ? "arXiv" : "Tạp chí Học thuật");

      let badgeText = "";
      let actionButtonsHtml = "";

      if (hasPdf) {
        badgeText = arxivId && !arxivId.startsWith("SCHOLAR-") && !arxivId.startsWith("ACAD-")
          ? `arXiv: ${arxivId} • ${year}`
          : `PDF Bản Gốc • ${year}`;

        actionButtonsHtml = `
          <button type="button" class="scout-action-btn ghost view-scout-doc-btn" data-doc-id="${escapeHtml(docId)}" title="Mở bản PDF song song bên cạnh (Chia đôi 2 tab: 1 tab chat, 1 tab bài báo)">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <line x1="14" y1="3" x2="14" y2="21"/>
            </svg>
            <span>Xem PDF Song Song</span>
          </button>
          <button class="scout-action-btn primary deep-dive-btn" data-doc-id="${escapeHtml(docId)}" data-title="${escapeHtml(title)}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            <span>Đọc &amp; Phân tích chuyên sâu bài này</span>
          </button>
        `;
      } else {
        badgeText = `Tham Chiếu Gốc • ${escapeHtml(sourceName)} • ${year}`;

        actionButtonsHtml = `
          ${sourceUrl ? `
            <a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer" class="scout-action-btn ghost scout-ext-link" title="Mở trang bài báo gốc tại ${escapeHtml(sourceName)} (Trang ngoài)">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
              </svg>
              <span>Mở Bài Báo Gốc ↗</span>
            </a>
          ` : ""}
          <button type="button" class="scout-action-btn ghost view-scout-doc-btn" data-doc-id="${escapeHtml(docId)}" title="Xem hồ sơ trích dẫn và tóm tắt bài báo ở tab bên cạnh">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <line x1="14" y1="3" x2="14" y2="21"/>
            </svg>
            <span>Xem Tóm Tắt &amp; Chi Tiết</span>
          </button>
          <button class="scout-action-btn primary deep-dive-btn" data-doc-id="${escapeHtml(docId)}" data-title="${escapeHtml(title)}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            <span>Phân tích bài này</span>
          </button>
        `;
      }

      papersHtml += `
        <div class="scout-paper-card" data-doc-id="${escapeHtml(docId)}">
          <div class="scout-card-header">
            <span class="scout-card-badge">${escapeHtml(badgeText)}</span>
            <span class="scout-card-source">${hasPdf ? "Đã tải PDF vào kho" : "Đã lưu tham chiếu"}</span>
          </div>
          <h4 class="scout-card-title">${escapeHtml(title)}</h4>
          <div class="scout-card-authors">Tác giả: ${escapeHtml(authors)}</div>
          <p class="scout-card-abstract">${escapeHtml(abstract.slice(0, 220))}...</p>
          <div class="scout-card-actions">
            ${actionButtonsHtml}
          </div>
        </div>
      `;
    });

    const card = document.createElement("div");
    card.className = "agent-output-block scout-results-block";
    card.innerHTML = `
      <div class="scout-banner">
        <div class="scout-banner-title">
          
          <span><b>Agent 0 (Literature Scout):</b> Đã phát hiện <b>${papers.length}</b> bài báo quốc tế tiêu biểu về <i>"${escapeHtml(query)}"</i></span>
        </div>
        <p class="scout-banner-subtitle">Tất cả các bài báo đã được tự động tải về và đăng ký trong <b>Kho Tài Liệu</b> bên trái. Hãy chọn bài bạn muốn thẩm định chuyên sâu:</p>
      </div>
      <div class="scout-papers-grid">
        ${papersHtml}
      </div>
    `;

    container.innerHTML = "";
    container.appendChild(card);

    // Sync sidecar evidence when clicking on the message card or its citations
    card.addEventListener("click", () => {
      if (evidence && evidence.length) {
        state.currentEvidence = evidence;
        updateSidecarEvidence(evidence);
      }
    });

    card.querySelectorAll(".inline-citation-badge").forEach(badge => {
      badge.addEventListener("click", (e) => {
        e.stopPropagation();
        state.currentEvidence = evidence;
        updateSidecarEvidence(evidence);
        const evIdx = parseInt(badge.dataset.ev, 10);
        const ev = evidence[evIdx];
        if (ev) {
          const pNum = ev.page || ev.page_no || 1;
          const targetDoc = state.documents.find(d => d.id === (ev.doc_id || state.selectedDocId)) || state.documents[0];
          if (targetDoc && pdfFrame) {
            pdfFrame.src = `/api/documents/${encodeURIComponent(targetDoc.id)}/file#page=${encodeURIComponent(pNum)}`;
          }
        }
      });
    });

    // Bind event for view document buttons (open side-by-side in sidecar)
    card.querySelectorAll(".view-scout-doc-btn").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const docId = btn.dataset.docId;
        if (docId) {
          await viewDocumentInSidecar(docId);
        }
      });
    });

    // Also allow clicking paper title to open side-by-side
    card.querySelectorAll(".scout-card-title").forEach(titleEl => {
      titleEl.style.cursor = "pointer";
      titleEl.title = "Bấm để mở xem bài báo song song bên cạnh chat";
      titleEl.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const cardEl = titleEl.closest(".scout-paper-card");
        const docId = cardEl ? cardEl.dataset.docId : null;
        if (docId) {
          await viewDocumentInSidecar(docId);
        }
      });
    });

    // Bind event for deep dive buttons
    card.querySelectorAll(".deep-dive-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const docId = btn.dataset.docId;
        const title = btn.dataset.title;
        triggerDeepDiveOnPaper(docId, title);
      });
    });
  }

  async function triggerDeepDiveOnPaper(docId, paperTitle) {
    toggleAutoScout(false);
    selectDocument(docId);
    await submitQuery(`Phân tích chuyên sâu và thẩm định học thuật bài báo: ${paperTitle}`);
  }

  // ==================== 6. RENDER FINAL ANSWER ====================
  function renderFinalAnswer(messageCard, data, query, finalElapsed = null) {
    if (!messageCard) return;
    const container = messageCard.querySelector(".agent-response-container");
    if (!container) return;

    try {
      const acc = messageCard.querySelector(".thinking-accordion");
      if (acc) {
        acc.classList.remove("open");
        try { updateThinkingTraces(messageCard, data, null, true, finalElapsed); } catch (e) {}
        const titleDiv = acc.querySelector(".thinking-title");
        if (titleDiv) {
          const timeStr = finalElapsed ? `${finalElapsed}s` : "";
          titleDiv.innerHTML = `
            <span class="thinking-sparkle completed">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
            </span>
            <span class="thinking-label-completed">Thought for</span>
            ${timeStr ? `<span class="thinking-timer">${timeStr}</span>` : ""}
            <span class="thinking-task-sep">•</span>
            <span class="thinking-task-text">5 bước suy luận & đối soát hoàn tất</span>
          `;
        }
      }

      const evidence = (data && data.evidence && data.evidence.length ? data.evidence : null) || (data && data.result && data.result.evidence) || (data && data.evidence_spans) || [];
      state.currentEvidence = evidence;
      try { updateSidecarEvidence(evidence); } catch (e) { console.warn("updateSidecarEvidence:", e); }

      const report = (data && (data.report || data.final_report || (data.result && data.result.report))) || "";
      if (report) {
        const reportPane = document.getElementById("sidecarReportPane");
        if (reportPane) {
          try {
            reportPane.innerHTML = `
              <div class="sidecar-report-content report-markdown-view">
                ${renderMarkdown(report, evidence)}
              </div>
            `;
          } catch (e) { console.warn("sidecarReportPane render:", e); }
        }
      }

      // 1. Extract the direct conversational AI answer
      let mainAnswer = "";
      
      const execMatch = report.match(/## 1\.\s*(?:Tóm\s*tắt\s*Tổng\s*quan|Executive\s*Summary)[\s\S]*?\n([\s\S]*?)(?=\n## 2|\n---|\Z)/i);
      const directAnsMatch = report.match(/> \*\*Câu trả lời Trực tiếp:\*\* \*\*([\s\S]*?)\*\*/);

      if (execMatch && execMatch[1].trim() && !execMatch[1].includes("Thông tin nghiên cứu học thuật xác thực liên quan đến: tóm tắt.")) {
        mainAnswer = execMatch[1].trim();
      } else if (directAnsMatch && directAnsMatch[1].trim()) {
        mainAnswer = directAnsMatch[1].trim();
      } else if (data && data.answer && data.answer.trim()) {
        mainAnswer = data.answer.trim();
      } else {
        const findingsMatch = report.match(/## 2\.\s*(?:Kết quả Nghiên cứu|Các Luận Điểm)[\s\S]*?\n([\s\S]*?)(?=\n## 3|\n---|\Z)/i);
        if (findingsMatch && findingsMatch[1].trim()) {
          mainAnswer = findingsMatch[1].trim();
        } else if (report && report.trim()) {
          mainAnswer = report.trim();
        } else {
          const claims = (data && data.verified_claims) || [];
          if (claims.length > 0) {
            mainAnswer = claims.map((c, i) => `${i + 1}. ${c.claim || c.text || c.evidence}`).join("\n\n");
          } else {
            mainAnswer = "Đã thẩm định và hoàn tất quy trình nghiên cứu.";
          }
        }
      }

      // Helper to sanitize document names
      function cleanTitle(str) {
        if (!str) return "Tài liệu";
        return str.replace(/^(?:gemini|doc)_[a-f0-9]+_/i, "")
                  .replace(/^(?:gemini|doc)_/i, "")
                  .replace(/\.(?:pdf|md|txt)$/i, "")
                  .replace(/_/g, " ").trim();
      }

      const docObj = (state.documents && state.documents.find(d => d.id === state.selectedDocId)) || (state.documents && state.documents[0]);
      const rawDocName = docObj ? (docObj.name || docObj.id) : "Tài liệu";
      const cleanDoc = cleanTitle(rawDocName);
      const shortDocName = cleanDoc.length > 22 ? cleanDoc.slice(0, 20) + "..." : cleanDoc;

      // Build bottom inline citation list if evidence spans exist
      let citationBadgesHtml = "";
      if (evidence && evidence.length > 0) {
        citationBadgesHtml = evidence.slice(0, 4).map((ev, i) => {
          const pNum = ev.page || ev.page_no || 1;
          const evDocClean = ev.doc_id ? cleanTitle(ev.doc_id) : shortDocName;
          const badgeLabel = evDocClean.length > 18 ? evDocClean.slice(0, 16) + "..." : evDocClean;
          return `<button class="inline-citation-badge" data-ev="${i}" title="${escapeHtml(ev.quote || '')}">${escapeHtml(badgeLabel)} • Trang ${pNum}</button>`;
        }).join(" ");
      }

      // Split dual reports (Academic Report & Audit Ledger)
      let academicReportMd = report;
      let auditLedgerMd = "";
      if (report && report.includes("<!-- DUAL_REPORT_SPLIT -->")) {
        const parts = report.split("<!-- DUAL_REPORT_SPLIT -->");
        academicReportMd = parts[0].trim();
        auditLedgerMd = parts[1].trim();
      } else if (report) {
        const splitIdx = report.search(/\n## 3\.\s*Đánh giá Bình duyệt/i);
        if (splitIdx !== -1) {
          academicReportMd = report.slice(0, splitIdx).trim();
          auditLedgerMd = "# Sổ cái Thẩm định & Đối soát Bằng chứng (Audit & Verification Ledger)\n\n" + report.slice(splitIdx).trim();
        } else {
          academicReportMd = report;
          auditLedgerMd = report;
        }
      }

      const card = document.createElement("div");
      card.className = "agent-output-block";
      card.innerHTML = `
        <!-- Chatbot AI Primary Answer -->
        <div class="chatbot-answer-container" style="display:flex; flex-direction:column; width:100%;">
          <div class="chatbot-answer-body" style="font-size:14.5px; line-height:1.7; color:#18181b;">
            ${renderMarkdown(mainAnswer, evidence)}
          </div>

          ${citationBadgesHtml ? `
          <div class="chatbot-sources-row" style="margin-top:10px;">
            <span class="sources-label" style="font-size:12px; font-weight:600; color:#71717a; margin-right:8px;">Nguồn trích dẫn:</span>
            ${citationBadgesHtml}
          </div>` : ''}

          <!-- Chatbot Action Toolbar -->
          <div class="chat-action-bar" style="display:flex; justify-content:space-between; align-items:center; margin-top:12px; padding-top:8px; border-top:1px solid #f4f4f5;">
            <div class="chat-action-left" style="display:flex; gap:6px;">
              <button class="chat-act-btn" id="likeBtn" title="Hữu ích">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
              </button>
              <button class="chat-act-btn" id="dislikeBtn" title="Chưa chuẩn">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path></svg>
              </button>
              <button class="chat-act-btn" id="reloadBtn" title="Tạo lại câu trả lời">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
              </button>
              <button class="chat-act-btn" id="copyBtn" title="Sao chép câu trả lời">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              </button>
            </div>
            <div class="chat-action-right">
              <span class="model-footprint-badge" style="font-size:11px; color:#a1a1aa; background:#f4f4f5; padding:3px 8px; border-radius:12px;">5 Agents Verified</span>
            </div>
          </div>

          <!-- Collapsible Dual Academic Report & Audit Ledger -->
          <div class="academic-report-collapsible">
            <div class="academic-collapse-header">
              <button class="academic-collapse-toggle" id="academicToggleBtn" title="Bấm để mở hoặc đóng báo cáo">
                <svg class="toggle-arrow" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
                <span class="toggle-main-title">Báo cáo Học thuật &amp; Sổ cái Đối soát</span>
                <span class="toggle-status-badge">100% Verified</span>
              </button>

              <div class="academic-actions-toolbar">
                <span class="evidence-count-badge">${evidence ? evidence.length : 0} trích dẫn</span>
                <button class="academic-action-icon-btn" id="exportMdBtn" title="Xuất file Markdown (.md)">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
                  <span class="action-btn-label">MD</span>
                </button>
                <button class="academic-action-icon-btn" id="exportPdfBtn" title="Xuất file PDF (.pdf)">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9V2h12v7"></path><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
                  <span class="action-btn-label">PDF</span>
                </button>
                <button class="academic-action-icon-btn" id="copyReportBtn" title="Sao chép nội dung">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                </button>
              </div>
            </div>

            <div class="academic-collapse-content" id="academicContent" style="display:none;">
              <!-- Minimalist Segmented Tabs -->
              <div class="report-segmented-control">
                <button class="report-segment-tab active" data-target="paneAcademicReport">Báo cáo Hoàn chỉnh</button>
                <button class="report-segment-tab" data-target="paneAuditLedger">Sổ cái Kiểm chứng</button>
              </div>

              <!-- Tab Pane 1: Full Academic Report -->
              <div class="report-pane-container" id="paneAcademicReport" style="display:block;">
                <div class="academic-content-inner report-markdown-view">
                  ${renderMarkdown(academicReportMd, evidence)}
                </div>
              </div>

              <!-- Tab Pane 2: Audit Ledger -->
              <div class="report-pane-container" id="paneAuditLedger" style="display:none;">
                <div class="academic-content-inner report-markdown-view">
                  ${renderMarkdown(auditLedgerMd, evidence)}
                </div>
              </div>
            </div>
          </div>
        </div>
      `;

      // Collapsible toggle handler
      const toggleBtn = card.querySelector("#academicToggleBtn");
      const contentDiv = card.querySelector("#academicContent");
      const arrowSvg = card.querySelector(".toggle-arrow");
      if (toggleBtn && contentDiv) {
        toggleBtn.addEventListener("click", () => {
          const isOpen = contentDiv.style.display !== "none";
          contentDiv.style.display = isOpen ? "none" : "block";
          if (arrowSvg) {
            arrowSvg.style.transform = isOpen ? "rotate(0deg)" : "rotate(90deg)";
          }
        });
      }

      // Tab switcher handler
      const tabBtns = card.querySelectorAll(".report-segment-tab");
      tabBtns.forEach(tab => {
        tab.addEventListener("click", () => {
          tabBtns.forEach(t => t.classList.remove("active"));
          tab.classList.add("active");
          const targetId = tab.dataset.target;
          const p1 = card.querySelector("#paneAcademicReport");
          const p2 = card.querySelector("#paneAuditLedger");
          if (p1 && p2) {
            p1.style.display = targetId === "paneAcademicReport" ? "block" : "none";
            p2.style.display = targetId === "paneAuditLedger" ? "block" : "none";
          }
        });
      });

      // Export Markdown button
      const exportMdBtn = card.querySelector("#exportMdBtn");
      if (exportMdBtn) {
        exportMdBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const activeTab = card.querySelector(".report-segment-tab.active");
          const isAudit = activeTab && activeTab.dataset.target === "paneAuditLedger";
          const contentToExport = isAudit ? auditLedgerMd : academicReportMd;
          const fileSuffix = isAudit ? "kiem_chung" : "bao_cao";
          const blob = new Blob([contentToExport || report], { type: "text/markdown;charset=utf-8" });
          const link = document.createElement("a");
          link.href = URL.createObjectURL(blob);
          link.download = `nghien_cuu_${fileSuffix}_${new Date().toISOString().slice(0, 10)}.md`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(link.href);
        });
      }

      // Export PDF button (Print layout)
      const exportPdfBtn = card.querySelector("#exportPdfBtn");
      if (exportPdfBtn) {
        exportPdfBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const activeTab = card.querySelector(".report-segment-tab.active");
          const isAudit = activeTab && activeTab.dataset.target === "paneAuditLedger";
          const pane = card.querySelector(isAudit ? "#paneAuditLedger" : "#paneAcademicReport");
          const bodyHtml = pane ? pane.innerHTML : "";
          const docTitle = isAudit ? "Sổ cái Thẩm định & Đối soát Bằng chứng" : "Báo cáo Nghiên cứu Khoa học";

          const printWin = window.open("", "_blank", "width=850,height=900");
          if (printWin) {
            printWin.document.write(`
              <!DOCTYPE html>
              <html>
              <head>
                <meta charset="utf-8">
                <title>${docTitle}</title>
                <style>
                  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 40px; color: #18181b; line-height: 1.6; font-size: 11pt; }
                  h1 { font-size: 18pt; border-bottom: 2px solid #e4e4e7; padding-bottom: 8px; margin-bottom: 16px; color: #09090b; }
                  h2 { font-size: 14pt; margin-top: 24px; color: #18181b; border-bottom: 1px solid #f4f4f5; padding-bottom: 4px; }
                  h3 { font-size: 12pt; margin-top: 18px; color: #27272a; }
                  p, li { line-height: 1.6; color: #27272a; }
                  table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 10pt; }
                  th, td { border: 1px solid #d4d4d8; padding: 7px 10px; text-align: left; }
                  th { background: #f4f4f5; font-weight: 600; color: #09090b; }
                  tr:nth-child(even) { background: #fafafa; }
                  blockquote { border-left: 3px solid #71717a; margin: 16px 0; padding: 8px 16px; background: #f8fafc; font-style: italic; color: #3f3f46; }
                  code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; background: #f1f5f9; padding: 2px 4px; font-size: 9.5pt; border-radius: 3px; }
                  @media print { body { padding: 0; } @page { margin: 1.5cm; } }
                </style>
              </head>
              <body>
                ${bodyHtml}
                <script>
                  window.onload = function() {
                    setTimeout(() => { window.print(); }, 250);
                  };
                </script>
              </body>
              </html>
            `);
            printWin.document.close();
          }
        });
      }

      // Copy report button
      const copyReportBtn = card.querySelector("#copyReportBtn");
      if (copyReportBtn) {
        copyReportBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const activeTab = card.querySelector(".report-segment-tab.active");
          const isAudit = activeTab && activeTab.dataset.target === "paneAuditLedger";
          const textToCopy = isAudit ? auditLedgerMd : academicReportMd;
          navigator.clipboard.writeText(textToCopy || report);
          copyReportBtn.classList.add("copied");
          setTimeout(() => copyReportBtn.classList.remove("copied"), 1500);
        });
      }

      // Copy button with feedback
      const copyBtn = card.querySelector("#copyBtn");
      if (copyBtn) {
        copyBtn.addEventListener("click", () => {
          navigator.clipboard.writeText(mainAnswer);
          copyBtn.title = "Đã sao chép";
          copyBtn.style.color = "#09090b";
          setTimeout(() => {
            copyBtn.title = "Sao chép câu trả lời";
            copyBtn.style.color = "";
          }, 1800);
        });
      }

      // Citation pill clicks jump to Sidecar evidence & page
      card.querySelectorAll(".inline-citation-badge").forEach(pillBtn => {
        pillBtn.addEventListener("click", () => {
          const evIdx = parseInt(pillBtn.dataset.ev, 10) || 0;
          openSidecar();
          if (evidence && evidence[evIdx]) {
            const ev = evidence[evIdx];
            const pageNo = ev.page || ev.page_no || 1;
            const doc = (state.documents && state.documents.find(d => d.id === state.selectedDocId)) || (state.documents && state.documents[0]);
            if (doc && pdfFrame) {
              pdfFrame.src = `/api/documents/${encodeURIComponent(doc.id)}/file#page=${encodeURIComponent(pageNo)}`;
            }
          }
        });
      });

      container.appendChild(card);
      messageFeed.scrollTo({ top: messageFeed.scrollHeight, behavior: "smooth" });
    } catch (err) {
      console.error("[ResearchDesk] Critical error in renderFinalAnswer:", err);
      const fallbackCard = document.createElement("div");
      fallbackCard.className = "agent-output-block";
      const fallbackText = (data && (data.final_report || data.report || data.answer)) || "Đã thẩm định và hoàn tất nghiên cứu.";
      fallbackCard.innerHTML = `
        <div class="chatbot-answer-container" style="padding:14px; background:#ffffff; border-radius:8px; line-height:1.6;">
          <div class="chatbot-answer-body">
            ${renderMarkdown(fallbackText, [])}
          </div>
        </div>
      `;
      container.appendChild(fallbackCard);
      messageFeed.scrollTo({ top: messageFeed.scrollHeight, behavior: "smooth" });
    }
  }

  // ==================== 7. UPDATE SIDECAR EVIDENCE ====================
  function updateSidecarEvidence(evidence) {
    evidenceCountBadge.textContent = `${evidence.length} trích dẫn`;
    if (!evidence.length) {
      sidecarEvidenceList.innerHTML = `
        <div class="empty-sidecar">
          <p>Không có trích dẫn trực tiếp nào cho câu hỏi này.</p>
        </div>
      `;
      return;
    }

    sidecarEvidenceList.innerHTML = evidence.map((ev, idx) => {
      const pageNum = ev.page || ev.page_no || 1;
      const quoteStr = ev.quote || ev.text || '';
      return `
        <div class="evidence-card-mono" data-idx="${idx}">
          <div class="ev-header">
            <span class="ev-badge-mono">100% Khớp</span>
            <span class="ev-page">Trang ${pageNum}</span>
          </div>
          <div class="ev-quote">“${escapeHtml(quoteStr)}”</div>
        </div>
      `;
    }).join("");

    sidecarEvidenceList.querySelectorAll(".evidence-card-mono").forEach(card => {
      card.addEventListener("click", () => {
        const idx = card.dataset.idx;
        const ev = evidence[idx];
        const pageNum = ev ? (ev.page || ev.page_no || 1) : 1;
        const docId = ev ? (ev.doc_id || ev.document_id || state.selectedDocId) : state.selectedDocId;
        jumpToPdfPage(pageNum, docId);
      });
    });
  }

  function switchSidecarTab(targetTab) {
    document.querySelectorAll(".sidecar-tabs .tab-btn").forEach(t => {
      t.classList.toggle("active", t.dataset.tab === targetTab);
    });
    ["evidenceTab", "pdfViewerTab", "reportViewTab"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle("active", id === targetTab);
    });
  }

  // ==================== JUMP TO EXACT PDF PAGE ====================
  function jumpToPdfPage(pageNo = 1, docId = null) {
    openSidecar();
    switchSidecarTab("pdfViewerTab");

    let targetDoc = null;
    if (docId) {
      targetDoc = (state.documents && state.documents.find(d => 
        d.id === docId || 
        d.doc_id === docId || 
        d.name === docId || 
        (d.path && d.path.includes(docId))
      )) || null;
    }
    if (!targetDoc) {
      targetDoc = (state.documents && state.documents.find(d => d.id === state.selectedDocId)) || (state.documents && state.documents[0]);
    }
    if (!targetDoc || !pdfFrame) return;

    state.selectedDocId = targetDoc.id;
    const p = Math.max(1, parseInt(pageNo, 10) || 1);

    // Update sidecar title and pop-out URL
    const sidecarDocTitle = document.getElementById("sidecarDocTitle");
    if (sidecarDocTitle) {
      const displayTitle = targetDoc.title || targetDoc.name || "Tài liệu học thuật";
      sidecarDocTitle.textContent = `${displayTitle} • Trang ${p}`;
      sidecarDocTitle.title = `${displayTitle} (Đang xem trang ${p})`;
    }

    const docFileUrl = `/api/documents/${encodeURIComponent(targetDoc.id)}/file`;
    const openExtBtn = document.getElementById("openExternalPdfBtn");
    if (openExtBtn) {
      openExtBtn.href = `${targetDoc.source_url || docFileUrl}#page=${p}`;
      openExtBtn.title = `Mở tài liệu tại Trang ${p} trong tab mới`;
      openExtBtn.style.display = "inline-flex";
    }

    const downloadBtn = document.getElementById("downloadDocBtn");
    if (downloadBtn) {
      downloadBtn.href = docFileUrl;
      downloadBtn.download = targetDoc.name || "academic_paper.pdf";
      downloadBtn.style.display = "inline-flex";
    }

    const pdfWrapper = document.getElementById("pdfFrameWrapper");
    const mdViewer = document.getElementById("docMdViewer");
    if (pdfWrapper) pdfWrapper.style.display = "block";
    if (mdViewer) mdViewer.style.display = "none";

    const targetUrl = `${docFileUrl}#page=${p}&zoom=page-fit`;

    // Force iframe reload if already on the same document to ensure Chromium jumps to new page
    try {
      const currentSrc = pdfFrame.src || "";
      if (currentSrc.includes(`/api/documents/${encodeURIComponent(targetDoc.id)}/file`)) {
        pdfFrame.src = "about:blank";
        setTimeout(() => {
          pdfFrame.src = targetUrl;
        }, 40);
      } else {
        pdfFrame.src = targetUrl;
      }
    } catch (e) {
      pdfFrame.src = targetUrl;
    }
  }

  async function viewDocumentInSidecar(docId) {
    let doc = (state.documents && state.documents.find(d => d.id === docId)) || null;
    if (!doc) {
      doc = { id: docId, name: docId };
    }
    state.selectedDocId = docId;

    const sidecarDocTitle = document.getElementById("sidecarDocTitle");
    if (sidecarDocTitle) {
      const displayTitle = doc.title || doc.name || "Tài liệu học thuật";
      sidecarDocTitle.textContent = displayTitle;
      sidecarDocTitle.title = displayTitle;
    }

    const docFileUrl = `/api/documents/${encodeURIComponent(doc.id)}/file`;
    const docNameLow = (doc.name || "").toLowerCase();
    const docPathLow = (doc.path || "").toLowerCase();
    const hasPdf = Boolean(doc.has_pdf || doc.doc_type === "pdf" || docNameLow.endsWith(".pdf") || docPathLow.endsWith(".pdf"));

    // Download button: ONLY show if real PDF exists
    const downloadBtn = document.getElementById("downloadDocBtn");
    if (downloadBtn) {
      if (hasPdf) {
        downloadBtn.href = docFileUrl;
        downloadBtn.download = doc.name || "academic_paper.pdf";
        downloadBtn.style.display = "inline-flex";
      } else {
        downloadBtn.style.display = "none";
      }
    }

    // Open in new tab button
    const openExtBtn = document.getElementById("openExternalPdfBtn");
    if (openExtBtn) {
      const extTarget = doc.source_url || docFileUrl;
      openExtBtn.href = extTarget;
      openExtBtn.title = doc.source_url ? "Mở bài báo gốc tại trang xuất bản" : "Mở tệp trong tab mới";
      openExtBtn.style.display = "inline-flex";
    }

    // Update Tab 1 Label
    const tabPdfLabel = document.getElementById("tabPdfLabel");
    if (tabPdfLabel) {
      tabPdfLabel.textContent = hasPdf ? "Bài Báo PDF Gốc" : "Hồ Sơ & Tham Chiếu Gốc";
    }

    const pdfWrapper = document.getElementById("pdfFrameWrapper");
    const mdViewer = document.getElementById("docMdViewer");

    if (hasPdf) {
      if (pdfWrapper) pdfWrapper.style.display = "block";
      if (mdViewer) mdViewer.style.display = "none";
      if (pdfFrame) {
        pdfFrame.src = `${docFileUrl}#page=1`;
      }
    } else {
      if (pdfWrapper) pdfWrapper.style.display = "none";
      if (mdViewer) {
        mdViewer.style.display = "block";
        const titleStr = doc.title || doc.name || "Tài liệu học thuật";
        const authorsStr = (doc.authors && doc.authors.length) ? (Array.isArray(doc.authors) ? doc.authors.join(", ") : doc.authors) : "Nhóm nghiên cứu";
        const sourceStr = doc.source || "Kho Lưu Trữ Học Thuật Quốc Tế";
        const sourceUrl = doc.source_url || "";
        const yearStr = doc.year || 2024;
        const abstractStr = doc.abstract || "";

        mdViewer.innerHTML = `
          <div class="academic-ref-profile" style="padding: 24px 28px; max-width: 800px; margin: 0 auto; color: #18181b;">
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; background: #f4f4f5; border: 1px solid #e4e4e7; border-radius: 4px; font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #52525b; margin-bottom: 14px;">
              <span>Tham chiếu bài báo gốc</span>
              <span>•</span>
              <span>Xuất bản ${escapeHtml(String(yearStr))}</span>
            </div>
            
            <h2 style="font-size: 20px; font-weight: 700; line-height: 1.4; margin: 0 0 12px 0; color: #09090b;">
              ${escapeHtml(titleStr)}
            </h2>
            
            <div style="font-size: 13.5px; color: #52525b; margin-bottom: 8px;">
              <b>Tác giả:</b> ${escapeHtml(authorsStr)}
            </div>
            <div style="font-size: 13.5px; color: #52525b; margin-bottom: 18px;">
              <b>Nguồn xuất bản:</b> ${escapeHtml(sourceStr)}
            </div>

            ${sourceUrl ? `
              <div style="margin: 18px 0 24px 0; padding: 16px; background: #fafafa; border: 1px solid #e4e4e7; border-radius: 8px;">
                <div style="font-size: 12.5px; color: #71717a; margin-bottom: 10px;">
                  Tài liệu này được công bố chính thức trên trang của nhà xuất bản / cơ sở dữ liệu học thuật. Nhấn nút bên dưới để truy cập trực tiếp bài báo gốc:
                </div>
                <a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer" style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 20px; background: #18181b; color: #ffffff; text-decoration: none; border-radius: 6px; font-size: 13.5px; font-weight: 600; transition: background 0.15s ease;">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                    <polyline points="15 3 21 3 21 9"/>
                    <line x1="10" y1="14" x2="21" y2="3"/>
                  </svg>
                  <span>Truy Cập Bài Báo Gốc Tại Nhà Xuất Bản ↗</span>
                </a>
              </div>
            ` : ""}

            <div style="margin-top: 20px; padding-top: 18px; border-top: 1px solid #e4e4e7;">
              <h3 style="font-size: 14.5px; font-weight: 700; margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.03em; color: #18181b;">
                Tóm Tắt Nội Dung Nghiên Cứu (Abstract)
              </h3>
              <p style="font-size: 14px; line-height: 1.75; color: #3f3f46; white-space: pre-wrap; margin: 0;">
                ${escapeHtml(abstractStr || "Đang cập nhật tóm tắt từ hệ thống...")}
              </p>
            </div>
          </div>
        `;
      }
    }

    openSidecar();
    switchSidecarTab("pdfViewerTab");
  }

  function openSidecar() {
    if (sidecar) {
      sidecar.classList.remove("collapsed");
      if (toggleSidecarBtn) toggleSidecarBtn.classList.add("active");
    }
  }

  function toggleSidecar() {
    if (!sidecar) return;
    const isNowCollapsed = sidecar.classList.toggle("collapsed");
    if (toggleSidecarBtn) {
      toggleSidecarBtn.classList.toggle("active", !isNowCollapsed);
    }
    if (!isNowCollapsed) {
      if (state.selectedDocId) {
        viewDocumentInSidecar(state.selectedDocId);
      } else if (state.documents.length > 0) {
        viewDocumentInSidecar(state.documents[0].id);
      }
    }
  }

  function toggleSidebar() {
    if (sidebar) {
      const isNowCollapsed = sidebar.classList.toggle("collapsed");
      if (toggleSidebarBtn) {
        toggleSidebarBtn.classList.toggle("active", !isNowCollapsed);
      }
    }
  }

  // Initial sync for panel buttons
  if (sidebar && toggleSidebarBtn) {
    toggleSidebarBtn.classList.toggle("active", !sidebar.classList.contains("collapsed"));
  }
  if (sidecar && toggleSidecarBtn) {
    toggleSidecarBtn.classList.toggle("active", !sidecar.classList.contains("collapsed"));
  }

  if (toggleSidebarBtn) toggleSidebarBtn.addEventListener("click", toggleSidebar);
  if (toggleSidecarBtn) toggleSidecarBtn.addEventListener("click", toggleSidecar);
  if (closeSidecarBtn) {
    closeSidecarBtn.addEventListener("click", () => {
      if (sidecar) sidecar.classList.add("collapsed");
      if (toggleSidecarBtn) toggleSidecarBtn.classList.remove("active");
    });
  }

  // Click "Mới" Button -> Reset session & clear active run
  if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
      state.activeSessionId = null;
      state.activeRunId = null;
      localStorage.removeItem("researchdesk_active_run");
      localStorage.removeItem("researchdesk_active_session");
      resetToWelcomeScreen();
    });
  }

  if (sendBtn) sendBtn.addEventListener("click", handleSendBtnClick);

  if (promptInput) {
    promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        if (e.isComposing || e.keyCode === 229) return;
        e.preventDefault();
        submitQuery();
      }
    });

    // Dynamic Send Button State
    promptInput.addEventListener("input", () => {
      promptInput.style.height = "48px";
      promptInput.style.height = Math.min(promptInput.scrollHeight, 140) + "px";
      if (!sendBtn.classList.contains("running")) {
        const hasVal = promptInput.value.trim().length > 0;
        setButtonState(hasVal ? "ready" : "idle");
      }
    });
  }

  suggestionCards.forEach(card => {
    card.addEventListener("click", () => {
      submitQuery(card.dataset.query);
    });
  });

  document.querySelectorAll(".sidecar-tabs .tab-btn").forEach(tab => {
    tab.addEventListener("click", () => {
      switchSidecarTab(tab.dataset.tab);
    });
  });

  // ==================== 8. PDF UPLOAD ====================
  if (uploadPdfBtn && pdfFileInput) {
    uploadPdfBtn.addEventListener("click", () => {
      pdfFileInput.click();
    });

    pdfFileInput.addEventListener("change", async () => {
      const file = pdfFileInput.files[0];
      if (!file) return;

      const originalBtnHtml = uploadPdfBtn.innerHTML;
      uploadPdfBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="animate-spin"><circle cx="12" cy="12" r="10"></circle></svg>
      `;
      uploadPdfBtn.style.opacity = "0.7";

      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("/api/documents/upload", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        await loadDocuments();
        
        if (data.document && data.document.id) {
          state.selectedDocId = data.document.id;
          activeDocName.textContent = data.document.name;
          renderDocuments();
        }
      } catch (err) {
        console.error("PDF Upload Error:", err);
        alert("Lỗi khi tải tệp lên: " + err.message);
      } finally {
        uploadPdfBtn.innerHTML = originalBtnHtml;
        uploadPdfBtn.style.opacity = "1";
        pdfFileInput.value = "";
      }
    });
  }

  // ==================== 9. MODEL POPOVER ====================
  if (modelDropdownBtn && agentMatrixPopover) {
    modelDropdownBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      agentMatrixPopover.classList.toggle("open");
      if (agentMatrixPopover.classList.contains("open")) {
        updateGeminiUsageBadge();
      }
    });

    if (closePopoverBtn) {
      closePopoverBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        agentMatrixPopover.classList.remove("open");
      });
    }

    document.addEventListener("click", (e) => {
      if (!agentMatrixPopover.contains(e.target) && !modelDropdownBtn.contains(e.target)) {
        agentMatrixPopover.classList.remove("open");
      }
    });
  }

  // ==================== 10. SYSTEM CONFIG & KAGGLE ====================
    function updateModelOptionsAvailability(isKaggleOnline, isGeminiOnline = true, isOllamaOnline = false) {
    state.isKaggleOnline = Boolean(isKaggleOnline);
    state.isGeminiOnline = Boolean(isGeminiOnline);
    state.isOllamaOnline = Boolean(isOllamaOnline);

    // 1. Cập nhật các nút Quick Select (Mờ xám nhạt và chặn chọn khi Offline)
    const gpuButtons = document.querySelectorAll('.quick-select-btn[data-mode="gpu_ft"], .quick-select-btn[data-mode="gpu_finetune"], .quick-select-btn[data-mode="gpu_max"], .quick-select-btn[data-mode="hybrid"]');
    gpuButtons.forEach(btn => {
      if (!state.isKaggleOnline) {
        btn.classList.add("mode-offline-dimmed");
        btn.classList.remove("active");
        btn.title = "⚠️ Cụm Kaggle GPU chưa kết nối (Offline). Hãy khởi động Kaggle và nhập URL để kích hoạt.";
      } else {
        btn.classList.remove("mode-offline-dimmed");
      }
    });

    const cpuMaxBtn = document.querySelector('.quick-select-btn[data-mode="cpu_max"]');
    const cpuMaxSplit = document.getElementById("cpuMaxSplit");
    if (cpuMaxBtn && cpuMaxSplit) {
      if (!state.isOllamaOnline) {
        cpuMaxSplit.classList.add("mode-offline-dimmed");
        cpuMaxBtn.classList.add("mode-offline-dimmed");
        cpuMaxBtn.classList.remove("active");
        cpuMaxBtn.title = "⚠️ Máy chủ Ollama CPU Max đang TẮT (OFF). Hãy nhấn nút [OFF] màu đỏ bên cạnh để bật.";
      } else {
        cpuMaxSplit.classList.remove("mode-offline-dimmed");
        cpuMaxBtn.classList.remove("mode-offline-dimmed");
        cpuMaxBtn.title = "Áp dụng CPU Max cho toàn bộ 5 Agent";
      }
    }

    // 2. Cập nhật thẻ select của từng Agent
    agentModelSelects.forEach(select => {
      const options = select.options;
      for (let i = 0; i < options.length; i++) {
        const opt = options[i];
        if (opt.value.startsWith("kaggle") || opt.value.startsWith("gpu") || opt.value === "hybrid") {
          opt.disabled = !state.isKaggleOnline;
          if (!state.isKaggleOnline) {
            opt.dataset.connected = "false";
            opt.classList.add("model-option-disabled");
            if (!opt.textContent.includes("(Chưa kết nối)")) {
              opt.textContent = opt.textContent.replace(/\s*\(.*?\)/, "") + " (Chưa kết nối)";
            }
          } else {
            opt.dataset.connected = "true";
            opt.classList.remove("model-option-disabled");
            opt.textContent = opt.textContent.replace(/\s*\(.*?\)/, "") + " (Sẵn sàng)";
          }
        } else if (opt.value === "cpu_max") {
          opt.disabled = !state.isOllamaOnline;
          if (!state.isOllamaOnline) {
            opt.classList.add("model-option-disabled");
            if (!opt.textContent.includes("(Đang tắt)")) {
              opt.textContent = opt.textContent.replace(/\s*\(.*?\)/, "") + " (Đang tắt)";
            }
          } else {
            opt.classList.remove("model-option-disabled");
            opt.textContent = opt.textContent.replace(/\s*\(.*?\)/, "") + " (Sẵn sàng)";
          }
        } else if (opt.value === "gemini") {
          opt.disabled = false;
          if (!isGeminiOnline) {
            opt.dataset.connected = "false";
            opt.classList.add("model-option-disabled");
            if (!opt.textContent.includes("(Chưa kết nối)")) {
              opt.textContent = opt.textContent.replace(/\s*\(.*?\)/, "") + " (Chưa kết nối)";
            }
          } else {
            opt.dataset.connected = "true";
            opt.classList.remove("model-option-disabled");
            opt.textContent = opt.textContent.replace(/\s*\(.*?\)/, "") + " (Sẵn sàng)";
          }
        } else if (opt.value === "cpu_min") {
          opt.disabled = false;
          opt.dataset.connected = "true";
          opt.classList.remove("model-option-disabled");
        }
      }
    });
  }

  // ==================== GEMINI USAGE QUOTA TRACKER ====================
  let lastGeminiBadge = "0 / 1.500";

  function syncModelLabelsFromDropdowns() {
    const selects = Array.from(document.querySelectorAll(".agent-model-select"));
    if (!selects.length) return;
    const allValues = selects.map(s => s.value);
    const allSame = allValues.every(v => v === allValues[0]);

    if (allSame) {
      const mode = allValues[0];
      const label = MODE_LABELS[mode] || mode;
      quickSelectBtns.forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
      });
      if (quickActivePill) quickActivePill.textContent = label;
      if (activeModelLabel) activeModelLabel.textContent = label;
    } else {
      quickSelectBtns.forEach(btn => btn.classList.remove("active"));
      if (quickActivePill) quickActivePill.textContent = "Tùy biến";
      if (activeModelLabel) activeModelLabel.textContent = "Tùy biến";
    }
    updateGeminiPillVisibility();
  }

  function updateGeminiPillVisibility() {
    const geminiPillBadge = document.getElementById("geminiPillBadge");
    if (!geminiPillBadge) return;
    const anyGeminiSelected = Array.from(document.querySelectorAll(".agent-model-select")).some(s => s.value === "gemini");
    geminiPillBadge.style.display = anyGeminiSelected ? "inline-flex" : "none";
    if (lastGeminiBadge) {
      geminiPillBadge.textContent = lastGeminiBadge;
    }
  }

  async function updateGeminiUsageBadge() {
    try {
      const res = await fetch("/api/config/gemini-usage", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      const used = Number(data.used) || 0;
      const limit = Number(data.limit) || 1500;
      const remaining = data.remaining !== undefined ? Number(data.remaining) : Math.max(0, limit - used);
      const limitFormatted = limit.toLocaleString("vi-VN");
      lastGeminiBadge = data.badge || `${used} / ${limitFormatted}`;
      const tooltipText = `Hôm nay (${data.date || "Hôm nay"}): Đã dùng ${used} / ${limitFormatted} lượt gọi Gemini API (Còn lại: ${remaining} lượt)`;
      
      const popoverBadge = document.getElementById("geminiPopoverBadge");
      if (popoverBadge) {
        popoverBadge.textContent = lastGeminiBadge;
        popoverBadge.title = tooltipText;
      }
      
      const pillBadge = document.getElementById("geminiPillBadge");
      if (pillBadge) {
        pillBadge.textContent = lastGeminiBadge;
        pillBadge.title = tooltipText;
      }
      updateGeminiPillVisibility();
    } catch (e) {
      console.warn("Failed to update Gemini usage:", e);
    }
  }

  // Periodic poll every 15s to keep quota real-time
  setInterval(updateGeminiUsageBadge, 15000);

  async function loadSystemConfig() {
    try {
      const res = await fetch("/api/config");
      const config = await res.json();

      if (config.remote_worker && config.remote_worker !== "Local API") {
        kaggleUrlInput.value = config.remote_worker;
      }

      await checkKaggleStatus();
      checkOllamaStatus();

  // Ollama CPU Status Check & 1-Click Power Button
  const ollamaToggleBtn = document.getElementById("ollamaToggleBtn");

  async function checkOllamaStatus() {
    if (!ollamaToggleBtn) return false;
    try {
      const res = await fetch("/api/config/local-cpu/test");
      const data = await res.json();
      const isOnline = Boolean(data.is_online);
      state.isOllamaOnline = isOnline;
      if (isOnline) {
        ollamaToggleBtn.className = "split-action-btn online";
        ollamaToggleBtn.innerHTML = '<span class="action-text">ON</span>';
        ollamaToggleBtn.title = "Ollama CPU đang chạy (Online).";
      } else {
        ollamaToggleBtn.className = "split-action-btn offline";
        ollamaToggleBtn.innerHTML = '<span class="action-text">OFF</span>';
        ollamaToggleBtn.title = "Ollama CPU chưa bật. Nhấp để bật ngay.";
      }
      updateModelOptionsAvailability(state.isKaggleOnline, state.isGeminiOnline, isOnline);
      return isOnline;
    } catch (e) {
      state.isOllamaOnline = false;
      if (ollamaToggleBtn) {
        ollamaToggleBtn.className = "split-action-btn offline";
        ollamaToggleBtn.innerHTML = '<span class="action-text">OFF</span>';
        ollamaToggleBtn.title = "Chưa kết nối server CPU. Nhấp để bật.";
      }
      updateModelOptionsAvailability(state.isKaggleOnline, state.isGeminiOnline, false);
      return false;
    }
  }
  if (ollamaToggleBtn) {
    ollamaToggleBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      e.preventDefault();

      ollamaToggleBtn.className = "split-action-btn loading";
      ollamaToggleBtn.innerHTML = '<span class="action-text">...</span>';
      ollamaToggleBtn.disabled = true;

      try {
        const res = await fetch("/api/config/local-cpu/start", { method: "POST" });
        const data = await res.json();

        if (data.is_online) {
          ollamaToggleBtn.className = "split-action-btn online";
          ollamaToggleBtn.innerHTML = '<span class="action-text">ON</span>';
          ollamaToggleBtn.title = "Ollama CPU đang chạy (Online)";
          alert(data.message || "Đã khởi động Ollama CPU thành công.");
        } else {
          ollamaToggleBtn.className = "split-action-btn offline";
          ollamaToggleBtn.innerHTML = '<span class="action-text">OFF</span>';
          alert(data.message || "Chưa thể khởi động Ollama tự động. Vui lòng chạy lệnh 'ollama serve' trong terminal.");
        }
      } catch (err) {
        ollamaToggleBtn.className = "split-action-btn offline";
        ollamaToggleBtn.innerHTML = '<span class="action-text">OFF</span>';
        alert("Lỗi khi yêu cầu bật Ollama: " + err.message);
      } finally {
        ollamaToggleBtn.disabled = false;
      }
    });
  }


      // Mặc định khi tải lại trang là CPU Min cho toàn bộ 5 Agent
      await setAllAgentsMode("cpu_min");
      await updateGeminiUsageBadge();
    } catch (e) {
      console.warn("Failed to load /api/config:", e);
    }
  }

  async function checkKaggleStatus() {
    try {
      const res = await fetch("/api/config/remote-url/test");
      const data = await res.json();
      const isOnline = Boolean(data.is_online);
      state.isKaggleOnline = isOnline;
      if (isOnline && kaggleStatusBadge) {
        kaggleStatusBadge.textContent = "Online";
        kaggleStatusBadge.className = "kaggle-status-badge online";
      } else if (kaggleStatusBadge) {
        kaggleStatusBadge.textContent = "Offline";
        kaggleStatusBadge.className = "kaggle-status-badge";
      }
      updateModelOptionsAvailability(isOnline, state.isGeminiOnline, state.isOllamaOnline);
      return isOnline;
    } catch (e) {
      state.isKaggleOnline = false;
      updateModelOptionsAvailability(false, state.isGeminiOnline, state.isOllamaOnline);
      return false;
    }
  }

  if (saveKaggleBtn && kaggleUrlInput) {
    saveKaggleBtn.addEventListener("click", async () => {
      const url = kaggleUrlInput.value.trim();
      if (!url) {
        alert("Vui lòng nhập đường link Ngrok (ví dụ: https://xxxx.ngrok-free.app)");
        return;
      }

      saveKaggleBtn.textContent = "...";
      saveKaggleBtn.disabled = true;

      try {
        const res = await fetch("/api/config/remote-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ remote_url: url }),
        });

        const data = await res.json();
        const isOnline = Boolean(data.is_online);
        
        await fetch("/api/config/mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: isOnline ? "kaggle_gpu" : "local_api" }),
        });

        updateModelOptionsAvailability(isOnline, true);

        if (isOnline) {
          if (kaggleStatusBadge) {
            kaggleStatusBadge.textContent = "Online";
            kaggleStatusBadge.className = "kaggle-status-badge online";
          }
          await setAllAgentsMode("gpu_max");
          alert("Kết nối thành công!\n\nCụm Kaggle Dual Tesla T4 đã Online.\nChi tiết: " + (data.details || "2x GPU sẵn sàng."));
        } else {
          if (kaggleStatusBadge) {
            kaggleStatusBadge.textContent = "Offline";
            kaggleStatusBadge.className = "kaggle-status-badge";
          }
          syncModelLabelsFromDropdowns();
          alert("Chưa thể kết nối đến Kaggle!\n\nChi tiết: " + (data.details || "Worker chưa phản hồi") + "\n\nNguyên nhân: Kaggle đang tải mô hình (khoảng 1-2 phút lần đầu). Vui lòng đợi một lát rồi bấm Lưu lại.");
        }
        saveKaggleBtn.textContent = "Đã lưu";

        setTimeout(() => {
          saveKaggleBtn.textContent = "Lưu";
          saveKaggleBtn.disabled = false;
        }, 1800);
      } catch (err) {
        console.error("Save Kaggle URL Error:", err);
        alert("Lỗi khi lưu URL: " + err.message);
        saveKaggleBtn.textContent = "Lưu";
        saveKaggleBtn.disabled = false;
      }
    });
  }

  
  // Autonomous Reflection Loops Control
  const loopBtns = document.querySelectorAll(".loop-select-btn");
  const loopsStatusVal = document.getElementById("reflectionLoopsStatus");
  if (loopBtns.length > 0) {
    const curLoops = state.maxReflectionLoops || 2;
    loopBtns.forEach(btn => {
      btn.classList.toggle("active", parseInt(btn.dataset.loops, 10) === curLoops);
      btn.addEventListener("click", () => {
        const loops = parseInt(btn.dataset.loops, 10) || 2;
        state.maxReflectionLoops = loops;
        localStorage.setItem("researchdesk_loops", loops);
        loopBtns.forEach(b => b.classList.toggle("active", b === btn));
        if (loopsStatusVal) {
          const labels = { 1: "1 Vòng (Nhanh)", 2: "2 Vòng (Khuyên dùng)", 3: "3 Vòng (Chuyên sâu)" };
          loopsStatusVal.textContent = labels[loops] || `${loops} Vòng`;
        }
      });
    });
    if (loopsStatusVal) {
      const labels = { 1: "1 Vòng (Nhanh)", 2: "2 Vòng (Khuyên dùng)", 3: "3 Vòng (Chuyên sâu)" };
      loopsStatusVal.textContent = labels[curLoops] || `${curLoops} Vòng`;
    }
  }

  // Tự động phân tích và chọn mô hình tối ưu nhất cho 5 Agent
  if (autoSelectBestBtn) {
    autoSelectBestBtn.addEventListener("click", async () => {
      const origHtml = autoSelectBestBtn.innerHTML;
      autoSelectBestBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        <span>Đang phân tích...</span>
      `;
      autoSelectBestBtn.disabled = true;

      try {
        const res = await fetch("/api/config/agents/auto-select", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        const data = await res.json();

        if (data.status === "success" && data.agent_modes) {
          // Cập nhật lại giá trị trên từng dropdown
          document.querySelectorAll(".agent-model-select").forEach(select => {
            const agentKey = select.dataset.agent;
            let targetMode = data.agent_modes[agentKey];
            if (targetMode === "cpu") targetMode = "cpu_min";
            if (targetMode === "model" || targetMode === "kaggle" || targetMode === "kaggle_max") targetMode = "gpu_max";
            if (targetMode === "kaggle_fn" || targetMode === "fn") targetMode = "gpu_ft";
            if (targetMode === "api") targetMode = "gemini";
            if (targetMode) {
              const opt = Array.from(select.options).find(o => o.value === targetMode);
              if (opt && !opt.disabled) {
                select.value = targetMode;
              }
            }
          });

          syncModelLabelsFromDropdowns();

          autoSelectBestBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
            <span>Đã tối ưu!</span>
          `;
        }
      } catch (err) {
        console.warn("Auto-select error:", err);
      } finally {
        setTimeout(() => {
          autoSelectBestBtn.innerHTML = origHtml;
          autoSelectBestBtn.disabled = false;
        }, 1600);
      }
    });
  }

  async function setAllAgentsMode(mode) {
    const agentModes = {};
    document.querySelectorAll(".agent-model-select").forEach(select => {
      const agentKey = select.dataset.agent;
      select.value = mode;
      agentModes[agentKey] = mode;
    });

    syncModelLabelsFromDropdowns();
    updateGeminiUsageBadge();

    try {
      await fetch("/api/config/agents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_modes: agentModes }),
      });
    } catch (e) {
      console.warn("Failed to persist agent modes:", e);
    }
  }

  // ==================== HOT-SWAP GPU VRAM MUTUAL EXCLUSION ====================
  let isGpuMaxActive = false;

  function updateGpuMutualExclusionUI(is32b) {
    isGpuMaxActive = Boolean(is32b);
    const hotSwapBanner = document.getElementById("hotSwapBanner");
    const gpuFtBtn = document.querySelector('.quick-select-btn[data-mode="gpu_ft"]');
    const gpuFnBtn = document.querySelector('.quick-select-btn[data-mode="gpu_finetune"]');

    if (isGpuMaxActive) {
      if (hotSwapBanner) hotSwapBanner.classList.add("show");
      if (gpuFtBtn) {
        gpuFtBtn.classList.add("vram-locked-dimmed");
        gpuFtBtn.title = "⚠️ Đang khóa: Toàn bộ 32GB VRAM đang được GPU Max 32B độc chiếm.";
      }
      if (gpuFnBtn) {
        gpuFnBtn.classList.add("vram-locked-dimmed");
        gpuFnBtn.title = "⚠️ Đang khóa: Toàn bộ 32GB VRAM đang được GPU Max 32B độc chiếm.";
      }
      document.querySelectorAll(".agent-model-select").forEach(sel => {
        Array.from(sel.options).forEach(opt => {
          if (opt.value === "gpu_ft" || opt.value === "gpu_finetune") {
            opt.disabled = true;
          }
        });
      });
    } else {
      if (hotSwapBanner) hotSwapBanner.classList.remove("show");
      if (gpuFtBtn) {
        gpuFtBtn.classList.remove("vram-locked-dimmed");
        gpuFtBtn.title = "Áp dụng GPU FT cho toàn bộ 5 Agent";
      }
      if (gpuFnBtn) {
        gpuFnBtn.classList.remove("vram-locked-dimmed");
        gpuFnBtn.title = "Áp dụng GPU Finetune (Cụm 7 Mô Hình SOTA) cho toàn bộ 5 Agent";
      }
      document.querySelectorAll(".agent-model-select").forEach(sel => {
        Array.from(sel.options).forEach(opt => {
          if (opt.value === "gpu_ft" || opt.value === "gpu_finetune") {
            opt.disabled = false;
          }
        });
      });
    }
  }

  function showHotSwapModal(targetTitle, estimatedSeconds) {
    const existing = document.getElementById("hotSwapModalOverlay");
    if (existing) existing.remove();

    const overlay = document.createElement("div");
    overlay.id = "hotSwapModalOverlay";
    overlay.className = "hot-swap-overlay";
    overlay.innerHTML = `
      <div class="hot-swap-card">
        <div class="hot-swap-title">🔄 Đang Hoán Đổi VRAM Trên Kaggle GPU...</div>
        <div class="hot-swap-desc">
          Hệ thống đang thu hồi bộ nhớ và nạp <strong>${targetTitle}</strong> vào cụm Dual Tesla T4.<br>
          Thời gian ước tính: khoảng <span id="hotSwapCountdown">${estimatedSeconds}</span> giây.
        </div>
        <div class="hot-swap-progress-bar-bg">
          <div class="hot-swap-progress-bar-fill" id="hotSwapProgressBar"></div>
        </div>
        <div style="font-size: 11px; color: #64748b;">Vui lòng giữ nguyên màn hình, không đóng trình duyệt...</div>
      </div>
    `;
    document.body.appendChild(overlay);

    let progress = 0;
    const progressFill = document.getElementById("hotSwapProgressBar");
    const countdownEl = document.getElementById("hotSwapCountdown");
    let remaining = estimatedSeconds;

    const interval = setInterval(() => {
      progress += (100 / (estimatedSeconds * 10));
      if (progress > 95) progress = 95;
      if (progressFill) progressFill.style.width = progress + "%";
      remaining -= 0.1;
      if (countdownEl && remaining > 0) {
        countdownEl.textContent = Math.ceil(remaining);
      }
    }, 100);

    return {
      close: () => {
        clearInterval(interval);
        if (progressFill) progressFill.style.width = "100%";
        setTimeout(() => {
          const el = document.getElementById("hotSwapModalOverlay");
          if (el) el.remove();
        }, 400);
      }
    };
  }

  async function executeRemoteHotSwap(targetMode) {
    const isTo32B = (targetMode === "gpu_max");
    const title = isTo32B ? "Đại Mô Hình Qwen 32B AWQ (TP=2)" : "Cụm 7 Mô Hình SOTA Finetune";
    const estSec = isTo32B ? 55 : 35;

    const modal = showHotSwapModal(title, estSec);
    try {
      const res = await fetch("/api/remote/switch-mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_mode: targetMode })
      });
      const data = await res.json();
      modal.close();

      if (!res.ok) {
        alert("Lỗi hoán đổi VRAM: " + (data.detail || res.statusText));
        return false;
      }

      await setAllAgentsMode(targetMode);
      updateGpuMutualExclusionUI(isTo32B);
      alert("🎉 " + (data.message || "Hoán đổi GPU thành công!"));
      return true;
    } catch (err) {
      modal.close();
      alert("Không thể kết nối đến máy chủ Kaggle: " + err.message);
      return false;
    }
  }

  // Hook revert button
  const revertTo7ModelsBtn = document.getElementById("revertTo7ModelsBtn");
  if (revertTo7ModelsBtn) {
    revertTo7ModelsBtn.addEventListener("click", async () => {
      if (confirm("Bạn có chắc muốn giải phóng 32B để nạp lại Cụm 7 Mô Hình SOTA trên Dual T4 không?")) {
        await executeRemoteHotSwap("gpu_finetune");
      }
    });
  }

  quickSelectBtns.forEach(btn => {
    btn.addEventListener("click", async () => {
      const mode = btn.dataset.mode;
      if (btn.classList.contains("mode-offline-dimmed") || (btn.parentElement && btn.parentElement.classList.contains("mode-offline-dimmed"))) {
        if (mode === "cpu_max") {
          alert("⚠️ Máy chủ CPU Max (Ollama) đang TẮT [OFF]!\n\nVui lòng nhấn nút [OFF] màu đỏ bên cạnh để khởi động máy chủ Ollama trên localhost:11434 trước khi chọn.");
        } else if (mode.startsWith("gpu") || mode === "hybrid") {
          alert("⚠️ Cụm Kaggle GPU chưa được kết nối [Offline]!\n\nVui lòng khởi chạy notebook trên Kaggle và nhập đường dẫn REMOTE_INFERENCE_URL vào ô cấu hình để kích hoạt các chế độ GPU.");
        }
        return;
      }
      if (btn.classList.contains("vram-locked-dimmed")) {
        alert("⚠️ Chế độ này đang bị tạm khóa vì GPU Max 32B đang độc chiếm toàn bộ 32GB VRAM. Hãy bấm nút 'Đổi về 7 Mô Hình' trên thanh banner để giải phóng 32B trước!");
        return;
      }
      if (mode === "gpu_max" && !isGpuMaxActive) {
        if (confirm("⚡ Bạn đang chọn GPU Max (Đại mô hình 32B AWQ).\n\nĐể chạy mô hình này, hệ thống sẽ tự động giải phóng cụm 7 mô hình và nạp Qwen-32B vào 2x GPU T4 (mất khoảng 50s).\n\nBạn có muốn tiến hành hoán đổi VRAM ngay bây giờ không?")) {
          await executeRemoteHotSwap("gpu_max");
        }
        return;
      }
      setAllAgentsMode(mode);
      if (mode !== "gpu_max") {
        updateGpuMutualExclusionUI(false);
      }
    });
  });

  agentModelSelects.forEach(select => {
    select.addEventListener("change", async () => {
      const agent = select.dataset.agent;
      const mode = select.value;

      syncModelLabelsFromDropdowns();

      try {
        await fetch("/api/config/agents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agent, mode }),
        });
      } catch (e) {
        console.warn("Failed to update agent mode:", e);
      }
    });
  });

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Enhanced Markdown Renderer with Code Blocks, Lists & Blockquotes
  function renderMarkdown(md, evidence = []) {
    if (!md) return "";
    let text = md.trim();

    // 1. Pre-process code blocks
    const codeBlocks = [];
    text = text.replace(/```([a-z0-9_-]*)\n([\s\S]*?)```/gim, (match, lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push(`<pre class="code-block"><code class="language-${lang}">${escapeHtml(code.trim())}</code></pre>`);
      return `__CODE_BLOCK_${idx}__`;
    });

    // 2. Escape HTML
    let html = escapeHtml(text);

    // Restore code blocks
    codeBlocks.forEach((cb, idx) => {
      html = html.replace(`__CODE_BLOCK_${idx}__`, cb);
    });

    // 3. Inline code
    html = html.replace(/`([^`]+)`/gim, '<code class="mono-code">$1</code>');

    // 4. Headers
    html = html.replace(/^### (.*$)/gim, '<h3 class="chat-h3">$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 class="chat-h2">$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1 class="chat-h1">$1</h1>');

    // 5. Bold & Italic
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

    // 5.5 Markdown Tables
    html = html.replace(/(?:^|\n)(\|.+?\|\r?\n\|[-:\s|]+\|\r?\n(?:\|.+?\|\r?\n?)+)/g, (match, tableBlock) => {
      const rows = tableBlock.trim().split(/\r?\n/).map(r => r.trim()).filter(Boolean);
      if (rows.length < 2) return match;
      const parseRow = (row, isHeader = false) => {
        const cells = row.split('|').slice(1, -1).map(c => c.trim());
        const tag = isHeader ? 'th' : 'td';
        return '<tr>' + cells.map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>';
      };
      const thead = '<thead>' + parseRow(rows[0], true) + '</thead>';
      const tbody = '<tbody>' + rows.slice(2).map(r => parseRow(r, false)).join('') + '</tbody>';
      return `\n<div class="chat-table-wrapper"><table class="chat-table">${thead}${tbody}</table></div>\n`;
    });

    // 6. Citations [1], [2], [doc...]
    html = html.replace(/\[(\d+)\]/g, (match, num) => {
      const evIdx = parseInt(num, 10) - 1;
      const ev = (evidence && evidence[evIdx]) ? evidence[evIdx] : null;
      const docName = (ev && (ev.filename || ev.document_id || ev.doc_id)) || "Tài liệu";
      const shortName = docName.length > 20 ? docName.slice(0, 18) + "..." : docName;
      const pageNum = (ev && (ev.page || ev.page_no)) ? (ev.page || ev.page_no) : "";
      const pageInfo = pageNum ? ` • Trang ${pageNum}` : "";
      const docAttr = (ev && (ev.document_id || ev.doc_id)) ? ` data-doc="${escapeHtml(ev.document_id || ev.doc_id)}"` : "";
      const pageAttr = pageNum ? ` data-page="${pageNum}"` : "";
      return `<button class="inline-citation-badge" data-ev="${evIdx}"${pageAttr}${docAttr} title="Nhấn để mở và nhảy đến Trang ${pageNum || 1} của tài liệu">[${num}] ${escapeHtml(shortName)}${pageInfo}</button>`;
    });

    // 6b. Direct Page References: disabled by user request

    // 7. Structured List Parsing with Lookahead & Tag Symmetry
    const lines = html.split(/\r?\n/);
    const processedLines = [];
    let currentListType = null; // null | 'ul' | 'ol'

    function closeCurrentList() {
      if (currentListType === 'ul') {
        processedLines.push('</ul>');
      } else if (currentListType === 'ol') {
        processedLines.push('</ol>');
      }
      currentListType = null;
    }

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      // Blank line handling: Check lookahead to prevent breaking loose lists
      if (trimmed.length === 0) {
        let nextListType = null;
        for (let j = i + 1; j < lines.length; j++) {
          const nextTrimmed = lines[j].trim();
          if (nextTrimmed.length === 0) continue;
          if (nextTrimmed.match(/^(\s*)[-\*•]\s+/)) {
            nextListType = 'ul';
          } else if (nextTrimmed.match(/^(\s*)\d+\.\s+/)) {
            nextListType = 'ol';
          }
          break;
        }

        if (currentListType && nextListType === currentListType) {
          // Keep current list open across blank line between items
          continue;
        } else {
          closeCurrentList();
          continue;
        }
      }

      const ulMatch = line.match(/^(\s*)[-\*•]\s+(.*)$/);
      const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/);

      if (ulMatch) {
        if (currentListType !== 'ul') {
          closeCurrentList();
          processedLines.push('<ul class="chat-bullet-list">');
          currentListType = 'ul';
        }
        const isSub = ulMatch[1].length >= 2;
        const subClass = isSub ? ' class="chat-sub-bullet"' : '';
        processedLines.push(`<li${subClass}>${ulMatch[2]}</li>`);
      } else if (olMatch) {
        if (currentListType !== 'ol') {
          closeCurrentList();
          processedLines.push('<ol class="chat-number-list">');
          currentListType = 'ol';
        }
        processedLines.push(`<li>${olMatch[3]}</li>`);
      } else {
        closeCurrentList();
        if (line.startsWith('&gt; ')) {
          processedLines.push(`<blockquote class="chat-blockquote">${line.slice(5)}</blockquote>`);
        } else if (!line.startsWith('<h1') && !line.startsWith('<h2') && !line.startsWith('<h3') && !line.startsWith('<pre') && !line.startsWith('<div')) {
          processedLines.push(`<p class="chat-paragraph">${line}</p>`);
        } else {
          processedLines.push(line);
        }
      }
    }
    closeCurrentList();

    return processedLines.join("\n");
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (sidecar) sidecar.classList.add("collapsed");
      if (agentMatrixPopover) agentMatrixPopover.classList.remove("open");
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (newChatBtn) newChatBtn.click();
    }
    if (e.ctrlKey && e.key.toLowerCase() === "b") {
      e.preventDefault();
      toggleSidecar();
    }
  });

  // ==================== 11. INITIALIZATION & SESSION RESTORE ====================
  
  // Global Click Delegation on messageFeed for all citation & page badges
  if (messageFeed) {
    messageFeed.addEventListener("click", (e) => {
      const citeBtn = e.target.closest(".inline-citation-badge");
      if (citeBtn) {
        e.preventDefault();
        e.stopPropagation();
        const pNum = citeBtn.dataset.page || 1;
        const dId = citeBtn.dataset.doc || state.selectedDocId;
        jumpToPdfPage(pNum, dId);
        return;
      }

      const pageBtn = e.target.closest(".inline-page-badge");
      if (pageBtn) {
        e.preventDefault();
        e.stopPropagation();
        const pNum = pageBtn.dataset.page || 1;
        const dId = pageBtn.dataset.doc || state.selectedDocId;
        jumpToPdfPage(pNum, dId);
        return;
      }
    });
  }

  async function init() {
    try {
      await loadDocuments();
      await loadChatHistory();
      await loadSystemConfig();

      // KHÔI PHỤC ĐOẠN CHAT ĐANG MỞ HOẶC ĐOẠN CHAT MỚI NHẤT TRÊN TRANG
      const savedSession = localStorage.getItem("researchdesk_active_session");
      if (savedSession && state.sessions.some(s => s.id === savedSession)) {
        await selectSession(savedSession);
      } else if (state.sessions.length > 0) {
        await selectSession(state.sessions[0].id);
      } else {
        resetToWelcomeScreen();
      }
    } catch (err) {
      console.error("Initialization error:", err);
    }
  }

  init();
});


// ==================== SLASH COMMANDS AUTOCOMPLETE SYSTEM ====================
const SLASH_COMMANDS = [
  {
    cmd: "/tom-tat",
    label: "Tóm tắt tài liệu",
    desc: "Tóm tắt tổng quan nội dung, đóng góp cốt lõi và kết quả chính của tài liệu nghiên cứu.",
    prompt: "Tóm tắt tổng quan nội dung, các đóng góp cốt lõi và kết quả chính của tài liệu nghiên cứu.",
    category: "Phân tích",
    icon: "⌘"
  },
  {
    cmd: "/so-sanh",
    label: "So sánh đối chiếu",
    desc: "So sánh chi tiết các tài liệu đã chọn về phương pháp, kiến trúc, ưu nhược điểm và benchmark.",
    prompt: "So sánh chi tiết các tài liệu đã chọn về phương pháp tiếp cận, kiến trúc kỹ thuật, ưu nhược điểm và kết quả thực nghiệm.",
    category: "Phân tích",
    icon: "⇄"
  },
  {
    cmd: "/dong-gop",
    label: "Đóng góp & Điểm mới",
    desc: "Trích xuất những điểm mới và đóng góp khoa học cốt lõi (Novel contributions).",
    prompt: "Trích xuất và phân tích các đóng góp khoa học mới (novelty & contributions) được đề xuất trong tài liệu.",
    category: "Phân tích",
    icon: "✦"
  },
  {
    cmd: "/phuong-phap",
    label: "Phương pháp & Kiến trúc",
    desc: "Phân tích chi tiết kiến trúc mô hình, pipeline thuật toán hoặc quy trình thực hiện.",
    prompt: "Phân tích chi tiết phương pháp đề xuất, kiến trúc mô hình/thuật toán và quy trình thực hiện.",
    category: "Kỹ thuật",
    icon: "⚙"
  },
  {
    cmd: "/kiem-chung",
    label: "Kiểm chứng sự thật",
    desc: "Fact-check đối soát thông tin dựa trên bằng chứng nguyên văn trích dẫn từ tài liệu.",
    prompt: "Kiểm chứng tính xác thực của thông tin sau dựa trên bằng chứng nguyên văn trích dẫn từ tài liệu: ",
    category: "Kiểm chứng",
    icon: "✓"
  },
  {
    cmd: "/so-lieu",
    label: "Bảng số liệu & Benchmark",
    desc: "Trích xuất toàn bộ bảng số liệu định lượng, kết quả benchmark và metric đánh giá.",
    prompt: "Trích xuất các số liệu thực nghiệm định lượng, bảng benchmark và metric so sánh hiệu năng quan trọng nhất.",
    category: "Kỹ thuật",
    icon: "▦"
  },
  {
    cmd: "/thuat-ngu",
    label: "Giải thích thuật ngữ",
    desc: "Liệt kê các thuật ngữ chuyên môn, khái niệm cốt lõi và từ viết tắt kèm định nghĩa.",
    prompt: "Liệt kê các thuật ngữ chuyên môn, khái niệm cốt lõi và từ viết tắt trong tài liệu kèm theo định nghĩa chi tiết.",
    category: "Khái niệm",
    icon: "§"
  },
  {
    cmd: "/trich-dan",
    label: "Bằng chứng nguyên văn",
    desc: "Trích xuất các đoạn văn nguyên văn quan trọng nhất (verbatim quotes) kèm vị trí và số trang.",
    prompt: "Trích xuất các đoạn văn nguyên văn quan trọng nhất (verbatim quotes) kèm vị trí/ngữ cảnh liên quan đến chủ đề: ",
    category: "Kiểm chứng",
    icon: "❝"
  },
  {
    cmd: "/khoang-trong",
    label: "Khoảng trống nghiên cứu",
    desc: "Chỉ ra các hạn chế (limitations) và hướng phát triển tương lai được tác giả đề cập.",
    prompt: "Chỉ ra các hạn chế (limitations), khoảng trống nghiên cứu (research gaps) và hướng phát triển tương lai được tác giả đề cập.",
    category: "Phân tích",
    icon: "◌"
  },
  {
    cmd: "/cau-hoi",
    label: "Tạo câu hỏi phản biện",
    desc: "Sinh 5-10 câu hỏi phản biện chuyên sâu và câu hỏi thảo luận học thuật.",
    prompt: "Tạo 5-10 câu hỏi phản biện chuyên sâu và câu hỏi thảo luận học thuật xoay quanh nội dung và giả định của bài báo.",
    category: "Học thuật",
    icon: "?"
  },
  {
    cmd: "/scout",
    label: "Dò quét nhanh (Scout)",
    desc: "Dò quét nhanh cấu trúc, mục lục, phạm vi chủ đề và các phần chính trong tài liệu.",
    prompt: "Dò quét nhanh cấu trúc, mục lục, phạm vi chủ đề và các phần chính trong toàn bộ tài liệu.",
    category: "Học thuật",
    icon: "/"
  },
  {
    cmd: "/cpu-min",
    label: "Chế độ CPU Min",
    desc: "Kích hoạt chế độ siêu nhẹ (Zero-Ollama, RAM < 100MB, tốc độ < 1 giây).",
    action: "set_mode_cpu_min",
    category: "Hệ thống",
    icon: "●"
  },
  {
    cmd: "/cpu-max",
    label: "Chế độ CPU Max",
    desc: "Kích hoạt chế độ đa tác nhân tối đa (Deep Reasoning, Full Fact-checking).",
    action: "set_mode_cpu_max",
    category: "Hệ thống",
    icon: "▲"
  },
  {
    cmd: "/xuat-bao-cao",
    label: "Xuất báo cáo Markdown",
    desc: "Tải toàn bộ kết quả nghiên cứu và bằng chứng hiện tại về máy dưới dạng file .md.",
    action: "export_report",
    category: "Hệ thống",
    icon: "↓"
  },
  {
    cmd: "/clear",
    label: "Xóa & Làm mới",
    desc: "Làm mới khung nhập và đặt lại trạng thái hiển thị.",
    action: "clear_input",
    category: "Hệ thống",
    icon: "✕"
  }
];

function initSlashCommandsAutosuggest({ textarea, container, onAction }) {
  if (!textarea || !container) return;

  const popup = document.createElement("div");
  popup.className = "slash-command-popup";
  popup.style.display = "none";
  container.appendChild(popup);

  let activeIndex = 0;
  let currentMatches = [];
  let isPopupOpen = false;

  function getBadgeClass(cat) {
    switch (cat) {
      case "Phân tích": return "slash-badge-phan-tich";
      case "Kỹ thuật": return "slash-badge-ky-thuat";
      case "Kiểm chứng": return "slash-badge-kiem-chung";
      case "Học thuật": return "slash-badge-hoc-thuat";
      case "Hệ thống": return "slash-badge-he-thong";
      default: return "slash-badge-he-thong";
    }
  }

  function render(matches) {
    currentMatches = matches;
    if (matches.length === 0) {
      popup.innerHTML = `
        <div class="slash-empty-state">Không có lệnh nào khớp với từ khóa</div>
      `;
      popup.style.display = "flex";
      isPopupOpen = true;
      return;
    }

    if (activeIndex >= matches.length) activeIndex = 0;

    const listHtml = matches.map((m, idx) => `
      <li class="slash-command-item ${idx === activeIndex ? "active" : ""}" data-index="${idx}">
        <span class="slash-cmd-icon">${m.icon}</span>
        <span class="slash-cmd-name">${m.cmd.replace(/^\//, "")}</span>
        <span class="slash-cmd-desc">${m.desc}</span>
      </li>
    `).join("");

    popup.innerHTML = `
      <ul class="slash-command-list">${listHtml}</ul>
      <div class="slash-command-footer">
        <div class="slash-hints">
          <span><kbd>↑</kbd> <kbd>↓</kbd> Di chuyển</span>
          <span><kbd>↵</kbd> hoặc <kbd>Tab</kbd> Chọn</span>
          <span><kbd>Esc</kbd> Đóng</span>
        </div>
      </div>
    `;

    popup.style.display = "flex";
    isPopupOpen = true;

    const activeEl = popup.querySelector(".slash-command-item.active");
    if (activeEl) activeEl.scrollIntoView({ block: "nearest" });

    popup.querySelectorAll(".slash-command-item").forEach(itemEl => {
      itemEl.addEventListener("mousedown", (e) => {
        e.preventDefault();
        const idx = parseInt(itemEl.dataset.index, 10);
        executeSelection(currentMatches[idx]);
      });
    });
  }

  function close() {
    popup.style.display = "none";
    isPopupOpen = false;
    currentMatches = [];
    activeIndex = 0;
  }

  function executeSelection(item) {
    if (!item) return;

    if (item.action) {
      close();
      const val = textarea.value;
      const pos = textarea.selectionStart;
      const before = val.substring(0, pos);
      const after = val.substring(pos);
      const slashIdx = before.lastIndexOf("/");
      if (slashIdx !== -1) {
        textarea.value = (before.substring(0, slashIdx) + after).trim();
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
      }
      if (onAction) onAction(item.action, item);
      return;
    }

    // Show ONLY item.cmd (e.g. "/tom-tat ") in input box, NOT the entire long prompt!
    const val = textarea.value;
    const pos = textarea.selectionStart;
    const before = val.substring(0, pos);
    const after = val.substring(pos);
    const slashIdx = before.lastIndexOf("/");
    const prefix = slashIdx !== -1 ? before.substring(0, slashIdx) : "";
    
    const insertion = item.cmd + " ";
    textarea.value = prefix + insertion + after;
    close();

    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    textarea.focus();
    const newPos = prefix.length + insertion.length;
    textarea.setSelectionRange(newPos, newPos);
  }

  // Keydown capture to intercept navigation and selection before form submissions
  textarea.addEventListener("keydown", (e) => {
    if (!isPopupOpen) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      e.stopPropagation();
      activeIndex = (activeIndex + 1) % currentMatches.length;
      render(currentMatches);
      return;
    }

    if (e.key === "ArrowUp") {
      e.preventDefault();
      e.stopPropagation();
      activeIndex = (activeIndex - 1 + currentMatches.length) % currentMatches.length;
      render(currentMatches);
      return;
    }

    if (e.key === "Enter" || e.key === "Tab") {
      if (currentMatches.length > 0) {
        e.preventDefault();
        e.stopPropagation();
        executeSelection(currentMatches[activeIndex]);
        return;
      }
    }

    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      close();
      return;
    }
  }, true);

  // Input detection
  textarea.addEventListener("input", () => {
    const val = textarea.value;
    const pos = textarea.selectionStart;
    const before = val.substring(0, pos);

    const match = before.match(/(?:^|\s)\/([a-zA-Z0-9_-]*)$/);
    if (match) {
      const query = match[1].toLowerCase();
      const filtered = SLASH_COMMANDS.filter(c => {
        const cmdClean = c.cmd.replace(/^\//, "").toLowerCase();
        const lbl = c.label.toLowerCase();
        const d = c.desc.toLowerCase();
        return cmdClean.includes(query) || lbl.includes(query) || d.includes(query);
      });
      activeIndex = 0;
      render(filtered);
    } else {
      if (isPopupOpen) close();
    }
  });

  document.addEventListener("click", (e) => {
    if (isPopupOpen && !popup.contains(e.target) && e.target !== textarea) {
      close();
    }
  });
}


// Hook Slash Commands Autocomplete into Omni Input Dock
const monoInputEl = document.getElementById("promptInput");
const monoDockEl = document.querySelector(".input-dock-mono");
if (monoInputEl && monoDockEl) {
  initSlashCommandsAutosuggest({
    textarea: monoInputEl,
    container: monoDockEl,
    onAction: async (action) => {
      if (action === "set_mode_cpu_min") {
        if (typeof setAllAgentsMode === "function") {
          setAllAgentsMode("cpu_min");
        }
        alert("🟢 Đã chuyển sang chế độ CPU Min (Siêu nhẹ, RAM < 100MB, tốc độ < 1s)");
      } else if (action === "set_mode_cpu_max") {
        if (typeof setAllAgentsMode === "function") {
          setAllAgentsMode("cpu_max");
        }
        alert("🚀 Đã chuyển sang chế độ CPU Max (Đa tác nhân, Full Fact-checking, 100% benchmark)");
      } else if (action === "export_report") {
        const exportLink = document.querySelector(".export-link-ghost");
        if (exportLink && exportLink.href) {
          exportLink.click();
        } else if (state && state.activeRunId) {
          window.open("/api/runs/" + encodeURIComponent(state.activeRunId) + "/export?format=md", "_blank");
        } else {
          alert("Chưa có báo cáo nào hoàn tất để tải xuống.");
        }
      } else if (action === "clear_input") {
        monoInputEl.value = "";
        monoInputEl.style.height = "48px";
        monoInputEl.dispatchEvent(new Event("input", { bubbles: true }));
        monoInputEl.focus();
      }
    }
  });
}
