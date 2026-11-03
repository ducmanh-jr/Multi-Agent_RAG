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
    kaggle_fn: "Kaggle FN",
    kaggle_max: "Kaggle Max",
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
    selectedDocId: null,
    checkedDocIds: new Set(),
    docSelectMode: false,
    activeRunId: null,
    currentEvidence: [],
    autoScout: false,
    isKaggleOnline: false,
    isGeminiOnline: true,
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

  // ==================== 1. LOAD CHAT HISTORY (NỬA TRÊN - CHỈ HIỂN THỊ TIÊU ĐỀ) ====================
  async function loadChatHistory() {
    try {
      const res = await fetch("/api/runs");
      const data = await res.json();
      state.runs = data.runs || [];
      if (historyCountBadge) historyCountBadge.textContent = state.runs.length;
      renderChatHistory();
    } catch (err) {
      console.error("Failed to load chat history:", err);
      if (chatHistoryList) {
        chatHistoryList.innerHTML = `<div class="doc-empty-state">Lỗi nạp lịch sử</div>`;
      }
    }
  }

  function renderChatHistory() {
    if (!chatHistoryList) return;
    if (!state.runs.length) {
      chatHistoryList.innerHTML = `<div class="doc-empty-state">Chưa có lịch sử nghiên cứu</div>`;
      return;
    }

    chatHistoryList.innerHTML = state.runs.map(run => {
      const isSelected = state.activeRunId === run.id;
      const question = run.question || "Cuộc trò chuyện";
      return `
        <div class="history-item ${isSelected ? 'active' : ''}" data-id="${run.id}" title="${escapeHtml(question)}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="history-item-icon">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          <span class="history-title">${escapeHtml(question)}</span>
          <div class="more-dots-wrap">
            <button class="more-dots-btn history-more-btn" data-id="${run.id}" title="Tùy chọn">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <circle cx="12" cy="5" r="2.2"></circle>
                <circle cx="12" cy="12" r="2.2"></circle>
                <circle cx="12" cy="19" r="2.2"></circle>
              </svg>
            </button>
            <div class="more-dots-menu">
              <button class="more-dots-menu-item history-del-action" data-del="${run.id}">
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
        selectRun(id);
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

    // Delete chat session action inside 3-dots menu
    chatHistoryList.querySelectorAll(".history-del-action").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const id = btn.dataset.del;
        await deleteRun(id);
      });
    });
  }

  async function deleteRun(runId) {
    clearActivePolling();
    try {
      await fetch(`/api/runs/${runId}`, { method: "DELETE" });
      const wasActive = state.activeRunId === runId;
      state.runs = state.runs.filter(r => r.id !== runId);
      if (historyCountBadge) historyCountBadge.textContent = state.runs.length;

      if (wasActive) {
        if (state.runs.length > 0) {
          await selectRun(state.runs[0].id);
        } else {
          localStorage.removeItem("researchdesk_active_run");
          resetToWelcomeScreen();
        }
      }
      renderChatHistory();
    } catch (err) {
      console.error("Failed to delete run:", err);
    }
  }

  async function clearAllHistory() {
    if (!state.runs.length) return;
    if (!confirm(`Bạn có chắc chắn muốn xóa toàn bộ ${state.runs.length} lịch sử nghiên cứu?`)) return;
    clearActivePolling();
    try {
      await fetch("/api/runs", { method: "DELETE" });
      state.runs = [];
      if (historyCountBadge) historyCountBadge.textContent = "0";
      localStorage.removeItem("researchdesk_active_run");
      resetToWelcomeScreen();
      renderChatHistory();
    } catch (err) {
      console.error("Failed to clear all history:", err);
    }
  }

  if (clearAllHistoryBtn) {
    clearAllHistoryBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearAllHistory();
    });
  }

  async function selectRun(runId) {
    clearActivePolling();
    state.activeRunId = runId;
    localStorage.setItem("researchdesk_active_run", runId);

    // Update active state in sidebar
    if (chatHistoryList) {
      chatHistoryList.querySelectorAll(".history-item").forEach(el => {
        el.classList.toggle("active", el.dataset.id === runId);
      });
    }

    try {
      const res = await fetch(`/api/runs/${runId}`);
      if (!res.ok) throw new Error("Run not found");
      const data = await res.json();

      if (welcomeHero) welcomeHero.style.display = "none";
      messageFeed.innerHTML = "";

      // 1. Render User Question
      appendUserMessage(data.question || "Nội dung câu hỏi");

      // 2. Render Agent Answer Card (or resume live polling if still running)
      const messageCard = createAgentSkeleton();
      messageFeed.appendChild(messageCard);
      if (data.status === "processing" || data.status === "running") {
        activeMessageCard = messageCard;
        setButtonState("running");
        updateThinkingTraces(messageCard, data);
        const startTime = performance.now();
        clearActiveTimer();
        activeTimerInterval = setInterval(() => {
          const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
          const timerEls = messageCard.querySelectorAll(".live-timer, .live-body-timer");
          timerEls.forEach(el => {
            el.textContent = `(${elapsed}s)`;
          });
        }, 100);
        pollRunStatus(runId, messageCard, data.question, startTime, activeTimerInterval);
      } else {
        setButtonState(promptInput && promptInput.value.trim().length > 0 ? "ready" : "idle");
        renderFinalAnswer(messageCard, data, data.question);
      }

      // 3. Sync associated Document
      const docId = data.document_id || (data.document_ids && data.document_ids[0]);
      if (docId) {
        selectDocument(docId, false);
      }
    } catch (err) {
      console.error("Failed to restore run details:", err);
      resetToWelcomeScreen();
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
    state.currentEvidence = [];
    state.selectedDocId = null;
    localStorage.removeItem("researchdesk_active_run");
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
        </div>
      `;
    }).join("");

    if (!state.selectedDocId && activeDocName && !state.autoScout) {
      activeDocName.textContent = "Chưa chọn tài liệu";
    }

    docList.querySelectorAll(".doc-item").forEach(item => {
      const docId = item.dataset.id;
      const chk = item.querySelector("input[type='checkbox']");

      // Clicking the checkbox toggles multi-select (can tick 1 or many documents)
      if (chk) {
        chk.addEventListener("click", (e) => {
          e.stopPropagation();
          toggleDocCheckbox(docId, chk.checked);
        });
      }

      // Clicking the row toggles selection for that document and opens PDF in right tab
      item.addEventListener("click", () => {
        const nextChecked = !state.checkedDocIds.has(docId);
        toggleDocCheckbox(docId, nextChecked);
        if (nextChecked) {
          viewDocumentInSidecar(docId);
        }
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
        if (updatePdf && pdfFrame) {
          pdfFrame.src = `/api/documents/${encodeURIComponent(doc.id)}/file#page=1`;
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
      if (updatePdf && pdfFrame) {
        pdfFrame.src = `/api/documents/${encodeURIComponent(doc.id)}/file#page=1`;
      }
    }
    updateDocDeleteUI();
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

  if (deleteDocsBtn) {
    deleteDocsBtn.addEventListener("click", (e) => {
      e.stopPropagation();
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
      if (val.startsWith("kaggle") && !state.isKaggleOnline) {
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

    // KIỂM TRA ĐIỀU KIỆN BẮT BUỘC: Chưa chọn tài liệu VÀ Chưa bật Auto-Scout
    if (!state.selectedDocId && !state.autoScout) {
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
    clearActiveTimer();
    const timerInterval = setInterval(() => {
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
      const timerEls = messageCard.querySelectorAll(".live-timer, .live-body-timer");
      timerEls.forEach(el => {
        el.textContent = `(${elapsed}s)`;
      });
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
          body: JSON.stringify({ query: text, max_results: 3 }),
          signal: activeAbortController.signal,
        });

        clearActiveTimer();
        const finalElapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        if (!scoutRes.ok) throw new Error("HTTP " + scoutRes.status);
        const scoutData = await scoutRes.json();

        // Tự động làm mới Kho Tài Liệu ở Sidebar bên trái
        await loadDocuments();

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
        document_ids: state.selectedDocId ? [state.selectedDocId] : [],
        max_iterations: 1,
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
      localStorage.setItem("researchdesk_active_run", data.id);
      
      // Ensure running state stays active while polling
      setButtonState("running");

      // Refresh history list so the new question immediately appears
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

        updateThinkingTraces(messageCard, data);

        if (data.status === "completed") {
          if (timerInterval) clearInterval(timerInterval);
          clearActiveTimer();
          clearActivePolling();
          activeMessageCard = null;
          const finalElapsed = startTime ? ((performance.now() - startTime) / 1000).toFixed(1) : null;
          renderFinalAnswer(messageCard, data, query, finalElapsed);
          await loadChatHistory();
          updateGeminiUsageBadge(); // Refresh quota
          setButtonState(promptInput.value.trim().length > 0 ? "ready" : "idle");
        } else if (data.status === "failed" || data.status === "cancelled") {
          if (timerInterval) clearInterval(timerInterval);
          clearActiveTimer();
          clearActivePolling();
          activeMessageCard = null;
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
    }, 1000);
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
              <span class="thinking-dot pulsing"></span>
              <span class="thinking-label">Thinking</span>
              <span class="thinking-dots-anim"><span>.</span><span>.</span><span>.</span></span>
              <span class="thinking-timer live-timer">(0.0s)</span>
              <span class="thinking-task-sep">•</span>
              <span class="thinking-task-text live-task-text">${escapeHtml(initialTask)}</span>
            </div>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </div>
          <div class="thinking-body" id="liveTraces">
            <div class="thinking-live-row">
              <span class="thinking-line-bullet">•</span>
              <span class="thinking-label">Thinking</span>
              <span class="thinking-dots-anim"><span>.</span><span>.</span><span>.</span></span>
              <span class="thinking-timer live-body-timer">(0.0s)</span>
            </div>
            <div class="thinking-traces-content" id="tracesContent">
              • Đang phân rã câu hỏi theo các góc nhìn nghiên cứu...<br>
              • Đang trích xuất cấu trúc văn bản và khối ngữ cảnh...
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

  function updateThinkingTraces(messageCard, data) {
    const traceBox = messageCard.querySelector("#tracesContent");
    if (!traceBox) return;

    const traces = data.reasoning_trace || [];
    if (traces.length > 0) {
      traceBox.innerHTML = traces.map(t => {
        return `• ${escapeHtml(t.action || 'Xử lý')}`;
      }).join("<br>");

      const lastTrace = traces[traces.length - 1];
      const taskText = messageCard.querySelector(".live-task-text");
      if (taskText && lastTrace && lastTrace.action) {
        taskText.textContent = lastTrace.action;
      }
    }
  }

  function updateAgentSkeletonError(messageCard, errorMsg) {
    const container = messageCard.querySelector(".agent-response-container");
    container.innerHTML = `
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
      const titleDiv = acc.querySelector(".thinking-title");
      if (titleDiv) {
        const papersCount = data.total_discovered || (data.papers && data.papers.length) || 0;
        const timeStr = finalElapsed ? `trong ${finalElapsed}s` : "";
        titleDiv.innerHTML = `
          <span class="thinking-dot"></span>
          <span>Agent 0 hoàn thành trinh sát ${timeStr} (${papersCount} bài báo tìm thấy)</span>
        `;
      }
    }

    const papers = data.papers || [];

    if (data.decision === "reject" || data.decision === "clarify" || data.decision === "no_relevant_papers" || papers.length === 0) {
      const bannerTitle = data.decision === "clarify" 
        ? "Cần làm rõ chủ đề nghiên cứu" 
        : (data.decision === "reject" ? "Chủ đề nằm ngoài danh mục STEM của ArXiv" : "Không tìm thấy bài báo phù hợp");
      const bannerMsg = data.message || "Hệ thống không tìm thấy tài liệu phù hợp trên ArXiv cho chủ đề này.";
      
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
      const pdfUrl = p.pdf_url || "#";
      const abstract = (p.abstract || "").trim();
      const docId = p.doc_id || p.id;

      papersHtml += `
        <div class="scout-paper-card" data-doc-id="${escapeHtml(docId)}">
          <div class="scout-card-header">
            <span class="scout-card-badge">arXiv: ${escapeHtml(arxivId)} • ${year}</span>
            <span class="scout-card-source">Đã lưu vào kho</span>
          </div>
          <h4 class="scout-card-title">${escapeHtml(title)}</h4>
          <div class="scout-card-authors">Tác giả: ${escapeHtml(authors)}</div>
          <p class="scout-card-abstract">${escapeHtml(abstract.slice(0, 220))}...</p>
          <div class="scout-card-actions">
            <a href="${escapeHtml(pdfUrl)}" target="_blank" class="scout-action-btn ghost" title="Xem PDF gốc">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
              <span>Xem PDF</span>
            </a>
            <button class="scout-action-btn primary deep-dive-btn" data-doc-id="${escapeHtml(docId)}" data-title="${escapeHtml(title)}">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              <span>Đọc &amp; Phân tích chuyên sâu bài này</span>
            </button>
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
        const titleDiv = acc.querySelector(".thinking-title");
        if (titleDiv) {
          const timeStr = finalElapsed ? `trong ${finalElapsed}s` : "";
          titleDiv.innerHTML = `
            <span class="thinking-dot"></span>
            <span>Đã đối soát trích dẫn (5 bước suy luận ${timeStr})</span>
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

          <!-- Collapsible Academic Report & Evidence Ledger -->
          <div class="academic-report-collapsible" style="margin-top:12px;">
            <button class="academic-collapse-toggle" id="academicToggleBtn" style="display:flex; justify-content:space-between; width:100%; padding:8px 12px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; cursor:pointer; font-size:12.5px;">
              <div class="collapse-toggle-left" style="display:flex; align-items:center; gap:6px;">
                <svg class="toggle-arrow" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
                <span>Xem Báo cáo Học thuật Chi tiết &amp; Đối soát Bằng chứng</span>
              </div>
              <div class="collapse-toggle-right" style="display:flex; align-items:center; gap:8px;">
                <span class="evidence-tag" style="background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:4px; font-size:11px;">${evidence ? evidence.length : 0} trích dẫn</span>
                <a href="/api/runs/${encodeURIComponent((data && data.id) || state.activeRunId || '')}/export?format=md" target="_blank" class="export-link-ghost" style="color:#2563eb; text-decoration:none;" title="Tải file Markdown">Xuất .md</a>
              </div>
            </button>
            <div class="academic-collapse-content" id="academicContent" style="display:none; padding:12px; border:1px solid #e2e8f0; border-top:none; border-radius:0 0 6px 6px;">
              <div class="academic-content-inner report-markdown-view">
                ${renderMarkdown(report, evidence)}
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
        toggleBtn.addEventListener("click", (e) => {
          if (e.target.closest(".export-link-ghost")) return;
          const isOpen = contentDiv.style.display !== "none";
          contentDiv.style.display = isOpen ? "none" : "block";
          if (arrowSvg) {
            arrowSvg.style.transform = isOpen ? "rotate(0deg)" : "rotate(90deg)";
          }
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
        const pageNum = ev.page || ev.page_no || 1;
        const doc = state.documents.find(d => d.id === state.selectedDocId) || state.documents[0];
        if (doc && pdfFrame) {
          pdfFrame.src = `/api/documents/${encodeURIComponent(doc.id)}/file#page=${encodeURIComponent(pageNum)}`;
          sidecarTabs.forEach(t => t.classList.toggle("active", t.dataset.tab === "pdfViewerTab"));
          const evTab = document.getElementById("evidenceTab");
          const pdfTab = document.getElementById("pdfViewerTab");
          if (evTab) evTab.classList.remove("active");
          if (pdfTab) pdfTab.classList.add("active");
        }
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

  function viewDocumentInSidecar(docId) {
    const doc = state.documents.find(d => d.id === docId);
    if (!doc) return;
    state.selectedDocId = docId;
    if (pdfFrame) {
      pdfFrame.src = `/api/documents/${encodeURIComponent(doc.id)}/file#page=1`;
    }
    const sidecarDocTitle = document.getElementById("sidecarDocTitle");
    if (sidecarDocTitle) {
      sidecarDocTitle.textContent = doc.name;
      sidecarDocTitle.title = doc.name;
    }
    const openExtBtn = document.getElementById("openExternalPdfBtn");
    if (openExtBtn) {
      openExtBtn.href = `/api/documents/${encodeURIComponent(doc.id)}/file`;
      openExtBtn.style.display = "inline-flex";
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
    if (sidebar) sidebar.classList.toggle("collapsed");
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
      localStorage.removeItem("researchdesk_active_run");
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
  function updateModelOptionsAvailability(isKaggleOnline, isGeminiOnline = true) {
    state.isKaggleOnline = Boolean(isKaggleOnline);
    state.isGeminiOnline = Boolean(isGeminiOnline);

    agentModelSelects.forEach(select => {
      const options = select.options;

      for (let i = 0; i < options.length; i++) {
        const opt = options[i];
        if (opt.value.startsWith("kaggle")) {
          opt.disabled = false;
          if (!isKaggleOnline) {
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
        } else if (opt.value.startsWith("cpu") || opt.value === "hybrid") {
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
      if (quickActivePill) quickActivePill.textContent = "Hybrid";
      if (activeModelLabel) activeModelLabel.textContent = "Hybrid";
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
      if (isOnline) {
        ollamaToggleBtn.className = "split-action-btn online";
        ollamaToggleBtn.innerHTML = '<span class="action-icon">🟢</span><span class="action-text">Bật</span>';
        ollamaToggleBtn.title = "Ollama CPU đang chạy (Online). Nhấp để kiểm tra lại.";
      } else {
        ollamaToggleBtn.className = "split-action-btn offline";
        ollamaToggleBtn.innerHTML = '<span class="action-icon">⚡</span><span class="action-text">Bật</span>';
        ollamaToggleBtn.title = "Ollama CPU chưa bật. Nhấp để bật ngay.";
      }
      return isOnline;
    } catch (e) {
      if (ollamaToggleBtn) {
        ollamaToggleBtn.className = "split-action-btn offline";
        ollamaToggleBtn.innerHTML = '<span class="action-icon">⚡</span><span class="action-text">Bật</span>';
        ollamaToggleBtn.title = "Chưa kết nối server CPU. Nhấp để bật.";
      }
      return false;
    }
  }

  if (ollamaToggleBtn) {
    ollamaToggleBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      e.preventDefault();

      ollamaToggleBtn.className = "split-action-btn loading";
      ollamaToggleBtn.innerHTML = '<span class="action-icon">⏳</span><span class="action-text">Đang bật...</span>';
      ollamaToggleBtn.disabled = true;

      try {
        const res = await fetch("/api/config/local-cpu/start", { method: "POST" });
        const data = await res.json();

        if (data.is_online) {
          ollamaToggleBtn.className = "split-action-btn online";
          ollamaToggleBtn.innerHTML = '<span class="action-icon">🟢</span><span class="action-text">Bật</span>';
          ollamaToggleBtn.title = "Ollama CPU đang chạy (Online)";
          alert("✅ " + (data.message || "Đã khởi động Ollama CPU thành công!"));
        } else {
          ollamaToggleBtn.className = "split-action-btn offline";
          ollamaToggleBtn.innerHTML = '<span class="action-icon">⚡</span><span class="action-text">Bật</span>';
          alert("⚠️ " + (data.message || "Chưa thể khởi động Ollama tự động. Vui lòng mở ứng dụng Ollama hoặc chạy lệnh 'ollama serve' trong terminal."));
        }
      } catch (err) {
        ollamaToggleBtn.className = "split-action-btn offline";
        ollamaToggleBtn.innerHTML = '<span class="action-icon">⚡</span><span class="action-text">Bật</span>';
        alert("❌ Lỗi khi yêu cầu bật Ollama: " + err.message);
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
      if (isOnline && kaggleStatusBadge) {
        kaggleStatusBadge.textContent = "Online";
        kaggleStatusBadge.className = "kaggle-status-badge online";
      } else if (kaggleStatusBadge) {
        kaggleStatusBadge.textContent = "Offline";
        kaggleStatusBadge.className = "kaggle-status-badge";
      }
      updateModelOptionsAvailability(isOnline, true);
      return isOnline;
    } catch (e) {
      updateModelOptionsAvailability(false, true);
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
          await setAllAgentsMode("kaggle_max");
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
            if (targetMode === "model" || targetMode === "kaggle") targetMode = "kaggle_max";
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

  quickSelectBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      setAllAgentsMode(mode);
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
      const pageInfo = (ev && ev.page) ? ` • Trang ${ev.page}` : "";
      return `<button class="inline-citation-badge" data-ev="${evIdx}" title="Xem trích dẫn đối soát [${num}] trong tài liệu">${escapeHtml(shortName)}${pageInfo}</button>`;
    });

    // 7. Structured List Parsing
    const lines = html.split(/\r?\n/);
    const processedLines = [];
    let inList = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const ulMatch = line.match(/^(\s*)[-\*•]\s+(.*)$/);
      const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/);

      if (ulMatch) {
        if (!inList) {
          processedLines.push('<ul class="chat-bullet-list">');
          inList = true;
        }
        const isSub = ulMatch[1].length >= 2;
        const subClass = isSub ? ' class="chat-sub-bullet"' : '';
        processedLines.push(`<li${subClass}>${ulMatch[2]}</li>`);
      } else if (olMatch) {
        if (!inList) {
          processedLines.push('<ol class="chat-number-list">');
          inList = true;
        }
        processedLines.push(`<li>${olMatch[3]}</li>`);
      } else {
        if (inList) {
          processedLines.push('</ul>');
          inList = false;
        }
        if (line.trim().length > 0) {
          if (line.startsWith('&gt; ')) {
            processedLines.push(`<blockquote class="chat-blockquote">${line.slice(5)}</blockquote>`);
          } else if (!line.startsWith('<h1') && !line.startsWith('<h2') && !line.startsWith('<h3') && !line.startsWith('<pre')) {
            processedLines.push(`<p class="chat-paragraph">${line}</p>`);
          } else {
            processedLines.push(line);
          }
        }
      }
    }
    if (inList) {
      processedLines.push('</ul>');
    }

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
  async function init() {
    try {
      await loadDocuments();
      await loadChatHistory();
      await loadSystemConfig();

      // RESTORE SESSION ON PAGE RELOAD:
      const savedRunId = localStorage.getItem("researchdesk_active_run");
      if (savedRunId && state.runs.some(r => r.id === savedRunId)) {
        await selectRun(savedRunId);
      } else {
        // Start with a clean welcome screen, no forced document checking
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
    icon: "📝"
  },
  {
    cmd: "/so-sanh",
    label: "So sánh đối chiếu",
    desc: "So sánh chi tiết các tài liệu đã chọn về phương pháp, kiến trúc, ưu nhược điểm và benchmark.",
    prompt: "So sánh chi tiết các tài liệu đã chọn về phương pháp tiếp cận, kiến trúc kỹ thuật, ưu nhược điểm và kết quả thực nghiệm.",
    category: "Phân tích",
    icon: "⚖️"
  },
  {
    cmd: "/dong-gop",
    label: "Đóng góp & Điểm mới",
    desc: "Trích xuất những điểm mới và đóng góp khoa học cốt lõi (Novel contributions).",
    prompt: "Trích xuất và phân tích các đóng góp khoa học mới (novelty & contributions) được đề xuất trong tài liệu.",
    category: "Phân tích",
    icon: "💡"
  },
  {
    cmd: "/phuong-phap",
    label: "Phương pháp & Kiến trúc",
    desc: "Phân tích chi tiết kiến trúc mô hình, pipeline thuật toán hoặc quy trình thực hiện.",
    prompt: "Phân tích chi tiết phương pháp đề xuất, kiến trúc mô hình/thuật toán và quy trình thực hiện.",
    category: "Kỹ thuật",
    icon: "⚙️"
  },
  {
    cmd: "/kiem-chung",
    label: "Kiểm chứng sự thật",
    desc: "Fact-check đối soát thông tin dựa trên bằng chứng nguyên văn trích dẫn từ tài liệu.",
    prompt: "Kiểm chứng tính xác thực của thông tin sau dựa trên bằng chứng nguyên văn trích dẫn từ tài liệu: ",
    category: "Kiểm chứng",
    icon: "🛡️"
  },
  {
    cmd: "/so-lieu",
    label: "Bảng số liệu & Benchmark",
    desc: "Trích xuất toàn bộ bảng số liệu định lượng, kết quả benchmark và metric đánh giá.",
    prompt: "Trích xuất các số liệu thực nghiệm định lượng, bảng benchmark và metric so sánh hiệu năng quan trọng nhất.",
    category: "Kỹ thuật",
    icon: "📊"
  },
  {
    cmd: "/thuat-ngu",
    label: "Giải thích thuật ngữ",
    desc: "Liệt kê các thuật ngữ chuyên môn, khái niệm cốt lõi và từ viết tắt kèm định nghĩa.",
    prompt: "Liệt kê các thuật ngữ chuyên môn, khái niệm cốt lõi và từ viết tắt trong tài liệu kèm theo định nghĩa chi tiết.",
    category: "Khái niệm",
    icon: "📖"
  },
  {
    cmd: "/trich-dan",
    label: "Bằng chứng nguyên văn",
    desc: "Trích xuất các đoạn văn nguyên văn quan trọng nhất (verbatim quotes) kèm vị trí và số trang.",
    prompt: "Trích xuất các đoạn văn nguyên văn quan trọng nhất (verbatim quotes) kèm vị trí/ngữ cảnh liên quan đến chủ đề: ",
    category: "Kiểm chứng",
    icon: "🔍"
  },
  {
    cmd: "/khoang-trong",
    label: "Khoảng trống nghiên cứu",
    desc: "Chỉ ra các hạn chế (limitations) và hướng phát triển tương lai được tác giả đề cập.",
    prompt: "Chỉ ra các hạn chế (limitations), khoảng trống nghiên cứu (research gaps) và hướng phát triển tương lai được tác giả đề cập.",
    category: "Phân tích",
    icon: "🔬"
  },
  {
    cmd: "/cau-hoi",
    label: "Tạo câu hỏi phản biện",
    desc: "Sinh 5-10 câu hỏi phản biện chuyên sâu và câu hỏi thảo luận học thuật.",
    prompt: "Tạo 5-10 câu hỏi phản biện chuyên sâu và câu hỏi thảo luận học thuật xoay quanh nội dung và giả định của bài báo.",
    category: "Học thuật",
    icon: "❓"
  },
  {
    cmd: "/scout",
    label: "Dò quét nhanh (Scout)",
    desc: "Dò quét nhanh cấu trúc, mục lục, phạm vi chủ đề và các phần chính trong tài liệu.",
    prompt: "Dò quét nhanh cấu trúc, mục lục, phạm vi chủ đề và các phần chính trong toàn bộ tài liệu.",
    category: "Học thuật",
    icon: "⚡"
  },
  {
    cmd: "/cpu-min",
    label: "Chế độ CPU Min",
    desc: "Kích hoạt chế độ siêu nhẹ (Zero-Ollama, RAM < 100MB, tốc độ < 1 giây).",
    action: "set_mode_cpu_min",
    category: "Hệ thống",
    icon: "🟢"
  },
  {
    cmd: "/cpu-max",
    label: "Chế độ CPU Max",
    desc: "Kích hoạt chế độ đa tác nhân tối đa (Deep Reasoning, Full Fact-checking).",
    action: "set_mode_cpu_max",
    category: "Hệ thống",
    icon: "🚀"
  },
  {
    cmd: "/xuat-bao-cao",
    label: "Xuất báo cáo Markdown",
    desc: "Tải toàn bộ kết quả nghiên cứu và bằng chứng hiện tại về máy dưới dạng file .md.",
    action: "export_report",
    category: "Hệ thống",
    icon: "📥"
  },
  {
    cmd: "/clear",
    label: "Xóa & Làm mới",
    desc: "Làm mới khung nhập và đặt lại trạng thái hiển thị.",
    action: "clear_input",
    category: "Hệ thống",
    icon: "🗑️"
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
        <div class="slash-command-header">
          <span class="slash-title">⚡ Lệnh nhanh (Slash Commands)</span>
          <span>Không tìm thấy</span>
        </div>
        <div class="slash-empty-state">Không có lệnh nào khớp với từ khóa</div>
      `;
      popup.style.display = "flex";
      isPopupOpen = true;
      return;
    }

    if (activeIndex >= matches.length) activeIndex = 0;

    const listHtml = matches.map((m, idx) => `
      <li class="slash-command-item ${idx === activeIndex ? "active" : ""}" data-index="${idx}">
        <div class="slash-cmd-icon-box">${m.icon}</div>
        <div class="slash-cmd-body">
          <div class="slash-cmd-title-row">
            <span class="slash-cmd-name">${m.cmd}</span>
            <span class="slash-cmd-label">${m.label}</span>
            <span class="slash-cmd-badge ${getBadgeClass(m.category)}">${m.category}</span>
          </div>
          <div class="slash-cmd-desc">${m.desc}</div>
        </div>
      </li>
    `).join("");

    popup.innerHTML = `
      <div class="slash-command-header">
        <span class="slash-title">⚡ Lệnh nhanh (Slash Commands)</span>
        <span>${matches.length} gợi ý</span>
      </div>
      <ul class="slash-command-list">${listHtml}</ul>
      <div class="slash-command-footer">
        <div class="slash-hints">
          <span><kbd>↑</kbd> <kbd>↓</kbd> Di chuyển</span>
          <span><kbd>↵</kbd> hoặc <kbd>Tab</kbd> Chọn</span>
          <span><kbd>Esc</kbd> Đóng</span>
        </div>
        <span>Antigravity RAG</span>
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

    if (item.prompt) {
      const val = textarea.value;
      const pos = textarea.selectionStart;
      const before = val.substring(0, pos);
      const after = val.substring(pos);
      const slashIdx = before.lastIndexOf("/");
      const prefix = slashIdx !== -1 ? before.substring(0, slashIdx) : "";
      
      textarea.value = prefix + item.prompt + after;
      close();

      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.focus();
      const newPos = prefix.length + item.prompt.length;
      textarea.setSelectionRange(newPos, newPos);
    }
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
