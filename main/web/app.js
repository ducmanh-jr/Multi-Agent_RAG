import { api, uploadDocument } from "./api.js";
import { $, $$, el, icon, dateLabel, statuses, toast } from "./dom.js";
import { renderEvents, renderResult, selectTab, copyReport } from "./report.js";
import { renderBenchmark, renderRoutes } from "./lab.js";
import { recordStatusLabel } from "./run_status.js";

const state = {
  documents: [],
  selected: new Set(),
  scopeMode: "selected",
  config: null,
  runId: null,
  working: false,
  uploading: false,
  pollVersion: 0,
};

async function initialize() {
  try {
    const auth = await api("/auth/status");
    if (auth.required && !auth.authenticated) {
      showLogin();
      return;
    }
    const [config, documents] = await Promise.all([
      api("/config"),
      api("/documents"),
    ]);
    state.config = config;
    state.documents = documents.documents;
    renderConfig();
    $("#workflow-profile").value = config.workflow_profile || "balanced";
    renderDocuments();
    $("#connection-label").textContent = "Backend đã kết nối";
    initPipelineStudio();
    checkPipelineStatus();
    const saved = localStorage.getItem("research-active-run");
    if (saved) {
      try {
        await openRun(saved);
      } catch {
        localStorage.removeItem("research-active-run");
      }
    }
  } catch (error) {
    $("#connection-label").textContent = "Chưa kết nối backend";
    $("#provider-pill").textContent = "Mất kết nối";
    $("#provider-pill").classList.add("disconnected");
    toast(error.message || "Không kết nối được backend.", true);
  }
}

function renderConfig() {
  const config = state.config;
  const pill = $("#provider-pill");
  pill.replaceChildren(
    el("span", "status-dot"),
    document.createTextNode(
      config.demo ? "Demo · không cần key" : config.model,
    ),
  );
  pill.classList.toggle("demo", config.demo);
  pill.classList.remove("disconnected");
  $("#mode-notice").hidden = !config.demo;
  $("#logout-button").hidden = !config.auth_required;
  const summary = $("#settings-summary");
  summary.replaceChildren();
  const values = [
    ["Provider", config.provider],
    ["Model", config.model],
    ["Embedding", config.embedding_model || "Chưa bật · dùng BM25"],
    ["Reranker", config.reranker_model || "Chưa bật"],
  ];
  for (const [label, value] of values) {
    const group = el("dl");
    group.append(el("dt", "", label), el("dd", "", value));
    summary.append(group);
  }
  renderRoutes(config.agent_routes || []);
}

function renderDocuments() {
  state.selected = new Set(
    [...state.selected].filter((id) =>
      state.documents.some((d) => d.id === id),
    ),
  );
  $("#nav-doc-count").textContent = String(state.documents.length);
  $("#selected-doc-count").textContent =
    `${state.scopeMode === "all" ? state.documents.length : state.selected.size}/${state.documents.length}`;
  $("#library-total").textContent = String(state.documents.length);
  $("#library-chunks").textContent = state.documents
    .reduce((sum, doc) => sum + doc.chunk_count, 0)
    .toLocaleString("vi-VN");
  $("#getting-started-title").textContent = state.documents.length
    ? "Tài liệu đã sẵn sàng. Bắt đầu khám phá thôi."
    : "Bắt đầu từ tài liệu bạn tin cậy";
  $("#getting-started-note").textContent = state.documents.length
    ? "Chọn nguồn, nhập câu hỏi hoặc dùng một gợi ý ở bên cạnh."
    : "Tải tài liệu của bạn hoặc khám phá bằng bộ ghi chú mẫu.";
  const sidebar = $("#sidebar-documents");
  sidebar.replaceChildren();
  if (!state.documents.length)
    sidebar.append(
      el(
        "p",
        "sidebar-empty",
        "Chưa có tài liệu.\nTải tệp hoặc thử bộ ghi chú mẫu.",
      ),
    );
  for (const document of state.documents) {
    const row = el("label", "sidebar-document");
    const input = el("input");
    input.type = "checkbox";
    input.checked = state.selected.has(document.id);
    input.disabled = state.working;
    input.setAttribute("aria-label", `Chọn ${document.filename}`);
    input.addEventListener("change", () => {
      input.checked
        ? state.selected.add(document.id)
        : state.selected.delete(document.id);
      state.scopeMode = "selected";
      renderDocuments();
    });
    const text = el("div");
    text.append(
      el("strong", "", document.filename),
      el(
        "small",
        "",
        `${(document.kind || document.doc_type || 'TXT').toUpperCase()} · ${(document.chunk_count || 1)} đoạn`,
      ),
    );
    row.append(input, text);
    sidebar.append(row);
  }
  renderLibrary();
  renderScopePicker();
  renderScope();
}

function renderLibrary() {
  const body = $("#document-table-body");
  body.replaceChildren();
  const query = $("#library-search").value.trim().toLocaleLowerCase("vi");
  const documents = state.documents.filter((doc) =>
    `${doc.title} ${doc.filename}`.toLocaleLowerCase("vi").includes(query),
  );
  if (!documents.length) {
    const row = el("tr");
    const cell = el(
      "td",
      "table-empty",
      query
        ? "Không tìm thấy tài liệu phù hợp. Thử một từ khóa khác."
        : "Kho tri thức của bạn đang chờ. Tải tài liệu hoặc nạp bộ ghi chú mẫu để bắt đầu.",
    );
    cell.colSpan = 5;
    row.append(cell);
    body.append(row);
  }
  for (const document of documents) {
    const row = el("tr");
    const first = el("td");
    first.append(
      el("strong", "", document.title),
      el("small", "", document.filename),
    );
    const kind = el("td");
    kind.append(el("span", "doc-type", (document.kind || document.doc_type || "TXT").toUpperCase()));
    row.append(
      first,
      kind,
      el("td", "", document.page_count || "—"),
      el("td", "", document.chunk_count),
    );
    const actions = el("td");
    const container = el("div", "document-actions");
    const open = el("a", "icon-button");
    open.href = `/api/documents/${document.id}/file`;
    open.target = "_blank";
    open.rel = "noopener";
    open.setAttribute("aria-label", `Mở ${document.filename}`);
    open.append(icon("external"));
    const remove = el("button", "icon-button");
    remove.setAttribute("aria-label", `Xóa ${document.filename}`);
    remove.append(icon("trash"));
    remove.addEventListener("click", () => deleteDocument(document));
    const use = el("button", "text-button", "Hỏi tài liệu này");
    use.disabled = state.working;
    use.addEventListener("click", () => {
      state.selected = new Set([document.id]);
      state.scopeMode = "selected";
      renderDocuments();
      switchView("research");
      $("#question").focus();
    });
    container.append(use, open, remove);
    actions.append(container);
    row.append(actions);
    body.append(row);
  }
}

function renderScope() {
  const selectedCount =
    state.scopeMode === "all" ? state.documents.length : state.selected.size;
  $("#library-selected").textContent = String(selectedCount);
  $("#selected-doc-count").textContent =
    `${selectedCount}/${state.documents.length}`;
  $("#scope-mode").value = state.scopeMode;
  $("#scope-label").textContent =
    state.scopeMode === "all"
      ? `Toàn bộ ${state.documents.length} tài liệu`
      : `${state.selected.size} tài liệu được chọn`;
  const names = state.documents.filter(
    (d) => state.scopeMode === "all" || state.selected.has(d.id),
  );
  $("#scope-filenames").textContent = names.length
    ? names.map((d) => d.filename).join(" · ")
    : "Chưa chọn tài liệu";
  updateResearchSteps();
}

function updateResearchSteps() {
  const hasSources =
    state.scopeMode === "all"
      ? state.documents.length > 0
      : state.selected.size > 0;
  const hasQuestion = $("#question").value.trim().length >= 8;
  const hasReport = !$("#results-section").hidden;
  const steps = [$("#step-sources"), $("#step-question"), $("#step-report")];
  steps.forEach((step, index) => {
    const complete =
      index === 0
        ? hasSources
        : index === 1
          ? hasSources && hasQuestion
          : hasReport;
    const current =
      !hasReport &&
      (index === 0
        ? !hasSources
        : index === 1
          ? hasSources && !hasQuestion
          : hasSources && hasQuestion);
    step.classList.toggle("complete", complete);
    step.classList.toggle("current", current);
    if (current) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");
  });
}

async function refreshDocuments() {
  state.documents = (await api("/documents")).documents;
  renderDocuments();
}

function setWorking(value) {
  state.working = value;
  $("#run-button").disabled = value;
  $("#cancel-button").hidden = !value;
  $("#question").disabled = value;
  $("#suggestion-area").hidden = value;
  $$(".prompt-chip").forEach((button) => {
    button.disabled = value;
  });
  $("#activity-cancel").hidden = !value;
  $("#scope-mode").disabled = value;
  $("#choose-sources").disabled = value;
  $("#requirements").disabled = value;
  renderDocuments();
  $$("#advanced-panel select, #advanced-panel input").forEach((select) => {
    select.disabled = value;
  });
}

async function startResearch() {
  if (state.working) return;
  const question = $("#question").value.trim();
  if (question.length < 8) {
    toast("Câu hỏi cần ít nhất 8 ký tự.", true);
    $("#question").focus();
    return;
  }
  if (!state.documents.length) {
    toast("Tải tài liệu hoặc nạp bộ mẫu trước khi nghiên cứu.", true);
    return;
  }
  if (state.uploading) {
    toast("Hãy chờ tài liệu được lập chỉ mục.");
    return;
  }
  if (state.scopeMode === "selected" && !state.selected.size) {
    $("#scope-dialog").showModal();
    toast("Chọn ít nhất một tài liệu.", true);
    return;
  }
  const requirements = $("#requirements")
    .value.split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (requirements.length > 12 || requirements.some((s) => s.length > 500)) {
    toast("Tối đa 12 yêu cầu, mỗi yêu cầu tối đa 500 ký tự.", true);
    return;
  }
  setWorking(true);
  $("#results-section").hidden = true;
  $("#empty-state").hidden = true;
  renderEvents([], "queued");
  try {
    const record = await api("/runs", {
      method: "POST",
      body: {
        question,
        requirements: $("#requirements")
          .value.split(/\r?\n/)
          .map((s) => s.trim())
          .filter(Boolean),
        document_ids: state.scopeMode === "all" ? [] : [...state.selected],
        scope_mode: state.scopeMode,
        language: $("#language").value,
        style: $("#report-style").value,
        retrieval_mode: $("#retrieval-mode").value,
        top_k: Number($("#top-k").value),
        max_rounds: Number($("#max-rounds").value),
        profile: $("#workflow-profile").value,
        use_cache: $("#use-cache").checked,
      },
    });
    state.runId = record.id;
    switchView("activity");
    localStorage.setItem("research-active-run", record.id);
    await pollRun(record.id);
  } catch (error) {
    toast(error.message, true);
    setWorking(false);
  }
}

async function pollRun(runId) {
  const version = ++state.pollVersion;
  let failures = 0;
  setWorking(true);
  while (version === state.pollVersion) {
    let record;
    try {
      record = await api(`/runs/${runId}`);
      failures = 0;
    } catch (error) {
      failures++;
      if (failures >= 3 || error.status === 404 || error.status === 401) {
        setWorking(false);
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 1300));
      continue;
    }
    if (version !== state.pollVersion) return;
    activityRecord(record);
    renderEvents(record.events, record.status, record.result);
    if (["completed", "failed", "cancelled"].includes(record.status)) {
      setWorking(false);
      if (record.result) {
        renderResult(record.result);
        switchView("research");
        setTimeout(() => {
          $("#results-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 150);
        toast("Nghiên cứu hoàn tất! Đã mở báo cáo kết quả.");
      } else {
        toast(
          record.error || statuses[record.status],
          record.status === "failed",
        );
        $("#empty-state").hidden = false;
      }
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 600));
  }
}

async function openRun(runId) {
  if (state.working && state.runId !== runId) {
    toast("Hãy hoàn thành hoặc dừng phiên hiện tại trước.");
    return;
  }
  const record = await api(`/runs/${runId}`);
  switchView("research");
  state.runId = runId;
  $("#question").value = record.request.question;
  $("#requirements").value = (record.request.requirements || []).join("\n");
  $("#question-count").textContent = `${record.request.question.length} / 2000`;
  $("#language").value = record.request.language;
  $("#report-style").value = record.request.style;
  $("#retrieval-mode").value = record.request.retrieval_mode;
  $("#top-k").value = String(record.request.top_k);
  $("#max-rounds").value = String(record.request.max_rounds);
  $("#workflow-profile").value =
    record.request.profile || state.config.workflow_profile || "academic";
  $("#use-cache").checked = record.request.use_cache !== false;
  state.selected = new Set(record.request.document_ids);
  state.scopeMode = state.selected.size ? "selected" : "all";
  activityRecord(record);
  renderDocuments();
  localStorage.setItem("research-active-run", runId);
  if (record.result) {
    renderEvents(record.events, record.status, record.result);
    renderResult(record.result);
    setWorking(false);
  } else if (["queued", "running"].includes(record.status))
    await pollRun(runId);
  else {
    renderEvents(record.events, record.status, record.result);
    $("#results-section").hidden = true;
    $("#empty-state").hidden = false;
    toast(record.error || statuses[record.status], record.status === "failed");
  }
}

async function cancelResearch() {
  if (!state.runId || !state.working) return;
  $("#cancel-button").disabled = true;
  try {
    await api(`/runs/${state.runId}/cancel`, { method: "POST" });
  } catch (error) {
    toast(error.message, true);
  } finally {
    $("#cancel-button").disabled = false;
  }
}

let currentUpload = null;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const unit = bytes < 1024 * 1024 ? "KB" : "MB";
  const size = unit === "KB" ? bytes / 1024 : bytes / (1024 * 1024);
  return `${new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 }).format(size)} ${unit}`;
}

function formatDuration(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

function formatEmbeddingRate(rate) {
  const amount = rate < 1 ? rate * 60 : rate;
  const unit = rate < 1 ? "đoạn/phút" : "đoạn/giây";
  return `${new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 }).format(amount)} ${unit}`;
}

function newUploadId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = [...bytes]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function renderUploadProgress(view) {
  if (!view) return;
  const panel = $("#upload-status");
  panel.hidden = false;
  panel.dataset.phase = view.phase;
  $("#upload-title").textContent =
    view.phase === "uploading"
      ? "Đang tải tài liệu"
      : view.phase === "processing"
        ? "Đang xử lý tài liệu"
        : view.phase === "error"
          ? "Tải tài liệu chưa hoàn tất"
          : "Đã hoàn tất";
  $("#upload-count").textContent = `${view.index}/${view.total}`;
  $("#upload-filename").textContent = view.file?.name || "";
  $("#upload-size").textContent = view.file
    ? formatBytes(view.file.size)
    : `${view.total} tệp · ${formatBytes(view.batchBytes)}`;
  const server = view.server || {};
  const facts = [];
  if (server.pages) facts.push(`${server.pages} trang PDF`);
  if (server.text_chars)
    facts.push(`${server.text_chars.toLocaleString("vi-VN")} ký tự`);
  if (server.chunks) facts.push(`${server.chunks} đoạn`);
  $("#upload-input-details").textContent = facts.length
    ? facts.join(" · ")
    : view.phase === "uploading"
      ? "Đang nhận tài liệu"
      : view.phase === "processing"
        ? "Đang phân tích đầu vào"
        : "Đã xử lý danh sách tệp";

  const progress = $("#upload-progress");
  progress.parentElement.hidden =
    view.phase === "complete" || view.phase === "error";
  if (view.phase === "uploading") {
    progress.value = view.percent;
    progress.setAttribute("aria-label", "Tiến độ truyền tài liệu");
    $("#upload-percent").textContent = `${view.percent}%`;
    panel.dataset.progressMode = "determinate";
  } else if (
    view.phase === "processing" &&
    server.embedding_percent !== null &&
    server.embedding_percent !== undefined
  ) {
    progress.value = server.embedding_percent;
    progress.setAttribute(
      "aria-label",
      "Tiến độ tạo vector cho các đoạn văn bản",
    );
    $("#upload-percent").textContent = `${server.embedding_percent}%`;
    panel.dataset.progressMode = "determinate";
  } else if (view.phase === "processing") {
    progress.removeAttribute("value");
    $("#upload-percent").textContent = "";
    panel.dataset.progressMode = "indeterminate";
  } else {
    $("#upload-percent").textContent = "";
    panel.dataset.progressMode = "none";
  }
  const stageText = {
    reading: "Đang đọc tệp…",
    waiting: "Đang chờ lượt xử lý tài liệu…",
    extracting: "Đang đọc tài liệu và trích xuất văn bản…",
    chunking: "Đang chia văn bản thành các đoạn…",
    embedding: server.model_loading
      ? "Đang nạp mô hình embedding lần đầu. Thời gian có thể lâu hơn…"
      : server.chunks
        ? `Đang tạo vector ${server.embedded || 0}/${server.chunks} đoạn…`
        : "Đang tạo vector cho tài liệu…",
    saving: "Đang lưu tài liệu và chỉ mục…",
    completed: "Đã lập chỉ mục. Đang nhận kết quả…",
  };
  $("#upload-message").textContent =
    view.message ||
    (view.phase === "uploading"
      ? "Đang truyền tệp lên máy chủ…"
      : stageText[server.stage] || "Đã truyền xong. Đang phân tích tài liệu…");
  $("#upload-transferred").textContent =
    `${formatBytes(view.bytes)} / ${formatBytes(view.file?.size ?? view.batchBytes)}`;
  $("#upload-work").textContent = server.chunks
    ? server.stage === "embedding"
      ? `${server.embedded || 0}/${server.chunks} đoạn`
      : `${server.chunks} đoạn đã tạo`
    : server.pages
      ? `${server.pages} trang đã đọc`
      : view.work || "Chưa xác định";
  const elapsed = Math.max(
    ((view.finishedAt || Date.now()) - view.startedAt) / 1000,
    server.elapsed_seconds || 0,
  );
  $("#upload-elapsed").textContent = formatDuration(elapsed);
  $("#upload-speed").textContent =
    view.phase === "uploading" && view.bytes > 0 && elapsed > 0.5
      ? `Tốc độ truyền: ${formatBytes(view.bytes / elapsed)}/giây`
      : view.phase === "processing" && server.embedding_rate
        ? `Tốc độ lập chỉ mục: ${formatEmbeddingRate(server.embedding_rate)}`
        : "";
  let remaining = "Đang tính";
  if (view.phase === "uploading" && view.bytes > 0 && elapsed > 0.5) {
    const seconds = ((view.file.size - view.bytes) * elapsed) / view.bytes;
    remaining = seconds < 1 ? "< 1 giây" : `~${formatDuration(seconds)}`;
  } else if (
    view.phase === "processing" &&
    server.remaining_seconds !== null &&
    server.remaining_seconds !== undefined
  ) {
    remaining =
      server.remaining_seconds < 1
        ? "< 1 giây"
        : `~${formatDuration(server.remaining_seconds)}`;
  } else if (view.phase === "complete" || view.phase === "error") {
    remaining = "—";
  }
  $("#upload-remaining").textContent = remaining;
  const step =
    view.phase === "uploading"
      ? 0
      : view.phase === "error" && !server.stage
        ? 0
        : ["reading", "waiting", "extracting"].includes(server.stage)
          ? 1
          : server.stage === "chunking"
            ? 2
            : 3;
  ["transfer", "extract", "chunk", "index"].forEach((name, index) => {
    const item = $(`#upload-step-${name}`);
    item.dataset.state =
      view.phase === "complete"
        ? "done"
        : view.phase === "error" && index === step
          ? "error"
          : index < step
            ? "done"
            : index === step
              ? "active"
              : "pending";
  });
  $("#upload-close").hidden =
    view.phase === "uploading" || view.phase === "processing";
}

function monitorUpload(uploadId, view) {
  let stopped = false;
  let timer;
  async function poll() {
    try {
      const progress = await api(`/uploads/${uploadId}`);
      if (!stopped && currentUpload === view) {
        view.server = progress;
        renderUploadProgress(view);
      }
    } catch (error) {
      if (error.status === 401) stopped = true;
      // The first poll can arrive before the upload handler registers its ID.
    }
    if (!stopped) timer = setTimeout(poll, 600);
  }
  poll();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}

function setUploadBusy(value) {
  state.uploading = value;
  for (const id of [
    "sidebar-upload",
    "library-upload",
    "dropzone",
    "seed-button",
    "library-seed",
  ]) {
    $(`#${id}`).disabled = value;
  }
}

async function uploadFiles(files) {
  if (!files.length) return;
  if (state.uploading) {
    toast("Đang tải tài liệu. Hãy chờ thao tác hiện tại hoàn thành.");
    return;
  }
  setUploadBusy(true);
  const accepted = ["pdf", "docx", "txt", "md"];
  let added = 0;
  let duplicates = 0;
  let failed = 0;
  let lastError = "";
  let sentBytes = 0;
  const batchBytes = files.reduce((sum, file) => sum + file.size, 0);
  const batchStartedAt = Date.now();
  const ticker = setInterval(() => renderUploadProgress(currentUpload), 1000);
  $("#connection-label").textContent = "Đang tải tài liệu…";
  try {
    for (const [offset, file] of files.entries()) {
      const view = {
        file,
        index: offset + 1,
        total: files.length,
        batchBytes,
        startedAt: Date.now(),
        bytes: 0,
        percent: 0,
        phase: "uploading",
        server: null,
      };
      currentUpload = view;
      renderUploadProgress(view);
      $("#connection-label").textContent = "Đang tải tài liệu…";
      if (!accepted.includes(file.name.split(".").at(-1).toLowerCase())) {
        failed++;
        lastError = `Không hỗ trợ ${file.name}.`;
        view.phase = "error";
        view.message = lastError;
        view.finishedAt = Date.now();
        renderUploadProgress(view);
        toast(lastError, true);
        continue;
      }
      if (file.size > (state.config?.max_upload_mb || 12) * 1024 * 1024) {
        failed++;
        lastError = `${file.name} vượt giới hạn dung lượng.`;
        view.phase = "error";
        view.message = lastError;
        view.finishedAt = Date.now();
        renderUploadProgress(view);
        toast(lastError, true);
        continue;
      }
      const uploadId = newUploadId();
      let stopMonitor = () => {};
      try {
        const result = await uploadDocument(file, {
          uploadId,
          onProgress: ({ percent, loaded }) => {
            view.percent = percent;
            view.bytes = loaded;
            renderUploadProgress(view);
          },
          onTransferred: () => {
            if (view.phase !== "uploading") return;
            view.bytes = file.size;
            view.phase = "processing";
            renderUploadProgress(view);
            $("#connection-label").textContent = "Đang lập chỉ mục tài liệu…";
            stopMonitor = monitorUpload(uploadId, view);
          },
        });
        view.bytes = file.size;
        try {
          view.server = await api(`/uploads/${uploadId}`);
        } catch {
          view.server = {
            stage: "completed",
            chunks: result.document.chunk_count,
            pages: result.document.page_count,
          };
        }
        view.phase = "complete";
        view.finishedAt = Date.now();
        view.message = result.duplicate
          ? "Tài liệu đã có trong kho."
          : `Đã lập chỉ mục ${result.document.chunk_count} đoạn.`;
        renderUploadProgress(view);
        if (result.duplicate) duplicates++;
        else added++;
        if (result.document.warnings?.length)
          toast(result.document.warnings[0], true);
        if (!state.working) {
          state.selected.add(result.document.id);
          state.scopeMode = "selected";
        }
        try {
          await refreshDocuments();
        } catch (error) {
          toast(
            `Tài liệu đã lưu nhưng chưa làm mới được danh sách: ${error.message}`,
            true,
          );
        }
      } catch (error) {
        failed++;
        lastError = `${file.name}: ${error.message}`;
        view.phase = "error";
        view.message = lastError;
        view.finishedAt = Date.now();
        renderUploadProgress(view);
        toast(lastError, true);
      } finally {
        sentBytes += view.bytes;
        stopMonitor();
      }
    }
  } finally {
    clearInterval(ticker);
    const parts = [];
    if (added) parts.push(`Đã thêm ${added} tài liệu`);
    if (duplicates) parts.push(`${duplicates} tài liệu đã có`);
    if (failed) parts.push(`${failed} tệp lỗi`);
    currentUpload = {
      phase: failed && !added && !duplicates ? "error" : "complete",
      index: files.length,
      total: files.length,
      file: null,
      batchBytes,
      bytes: sentBytes,
      startedAt: batchStartedAt,
      finishedAt: Date.now(),
      work: `${added}/${files.length} tài liệu đã thêm`,
      message:
        failed === 1 && files.length === 1
          ? lastError
          : `${parts.join(" · ")}.`,
    };
    renderUploadProgress(currentUpload);
    setUploadBusy(false);
    $("#connection-label").textContent = "Backend đã kết nối";
    $("#file-input").value = "";
  }
}

async function seedDocuments() {
  if (state.uploading) return;
  setUploadBusy(true);
  $("#connection-label").textContent = "Đang nạp tài liệu mẫu…";
  try {
    const seeded = await api("/documents/demo", { method: "POST" });
    if (!state.working) {
      state.selected = new Set(seeded.documents.map((d) => d.id));
      state.scopeMode = "selected";
    }
    await refreshDocuments();
    toast("Đã nạp ghi chú mẫu. Đây là tài liệu demo do hệ thống biên soạn.");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setUploadBusy(false);
    $("#connection-label").textContent = "Backend đã kết nối";
  }
}

async function deleteDocument(document) {
  if (
    !window.confirm(
      `Xóa "${document.filename}" khỏi kho? Các báo cáo đã lưu giữ bản chụp bằng chứng.`,
    )
  )
    return;
  try {
    await api(`/documents/${document.id}`, { method: "DELETE" });
    await refreshDocuments();
    toast("Đã xóa tài liệu.");
  } catch (error) {
    toast(error.message, true);
  }
}

function switchView(name) {
  $$(".view").forEach((view) => {
    view.hidden = view.id !== `view-${name}`;
  });
  $$(".main-nav .nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === name);
    if (item.dataset.view === name) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  $("#breadcrumb-view").textContent = {
    research: "Nghiên cứu",
    activity: "Hoạt động agent",
    library: "Kho tài liệu",
    history: "Lịch sử phiên",
    experiments: "Thực nghiệm",
    settings: "Cài đặt",
    pipeline: "Quy trình",
  }[name] || "Cài đặt";
  if (name === "pipeline") {
    checkPipelineStatus();
  }
  const menuWasOpen = $("#sidebar").classList.contains("open");
  setSidebarOpen(false);
  if (menuWasOpen) $("#main-content").focus({ preventScroll: true });
}

async function showHistory() {
  const root = $("#history-list");
  root.replaceChildren(el("p", "loading-label", "Đang tải lịch sử…"));
  try {
    const records = (await api("/runs")).runs;
    root.replaceChildren();
    if (!records.length)
      root.append(
        el(
          "p",
          "loading-label",
          "Chưa có phiên nghiên cứu. Bắt đầu từ tài liệu trong kho.",
        ),
      );
    for (const record of records) {
      const item = el("button", "history-item");
      const content = el("div");
      content.append(
        el("strong", "", record.question),
        el(
          "small",
          "",
          `${dateLabel(record.created_at)} · ${record.id.slice(0, 16)}…`,
        ),
      );
      item.append(
        icon("file"),
        content,
        el("span", `history-state ${record.status}`, recordStatusLabel(record)),
      );
      item.addEventListener("click", () =>
        openRun(record.id).catch((error) => toast(error.message, true)),
      );
      root.append(item);
    }
  } catch (error) {
    root.replaceChildren(el("p", "loading-label", error.message));
  }
}

function showLogin() {
  if (!$("#login-dialog").open) $("#login-dialog").showModal();
}

$$(".main-nav .nav-item").forEach((button) =>
  button.addEventListener("click", () => {
    switchView(button.dataset.view);
    if (button.dataset.view === "history") showHistory();
    if (button.dataset.view === "library")
      refreshDocuments().catch((error) => toast(error.message, true));
    if (button.dataset.view === "settings")
      renderSettingsView();
    if (button.dataset.view === "pipeline")
      checkPipelineStatus();
  }),
);
["sidebar-upload", "library-upload", "dropzone"].forEach((id) =>
  $(`#${id}`).addEventListener("click", () => $("#file-input").click()),
);
$("#file-input").addEventListener("change", (event) =>
  uploadFiles([...event.target.files]),
);
$("#upload-close").addEventListener("click", () => {
  $("#upload-status").hidden = true;
});
["seed-button", "library-seed"].forEach((id) =>
  $(`#${id}`).addEventListener("click", seedDocuments),
);
$("#reindex-button").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const result = await api("/documents/reindex", { method: "POST" });
    toast(`Đã reindex ${result.chunks_updated} đoạn.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
});
$("#dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("#dropzone").classList.add("dragging");
});
$("#dropzone").addEventListener("dragleave", () =>
  $("#dropzone").classList.remove("dragging"),
);
$("#dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("#dropzone").classList.remove("dragging");
  uploadFiles([...event.dataTransfer.files]);
});
$("#question").addEventListener("input", (event) => {
  $("#question-count").textContent = `${event.target.value.length} / 2000`;
  updateResearchSteps();
});
$("#question").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    startResearch();
  }
});
$$(".prompt-chip").forEach((button) =>
  button.addEventListener("click", () => {
    $("#question").value = button.dataset.question;
    $("#question").dispatchEvent(new Event("input"));
    $("#question").focus();
  }),
);
$("#run-button").addEventListener("click", startResearch);
$("#cancel-button").addEventListener("click", cancelResearch);
$("#advanced-toggle").addEventListener("click", () => {
  const panel = $("#advanced-panel");
  panel.hidden = !panel.hidden;
  $("#advanced-toggle").setAttribute("aria-expanded", String(!panel.hidden));
});
["settings-button", "notice-settings"].forEach((id) =>
  $(`#${id}`).addEventListener("click", async () => {
    $("#settings-dialog").showModal();
    try {
      const info = await api("/diagnostics");
      $("#diagnostics").textContent =
        `${info.documents} tài liệu · ${info.chunks} đoạn · ${info.indexed_chunks} đoạn có vector. ${info.reindex_needed ? "Cần reindex cho model hiện tại." : "Chỉ mục sẵn sàng."} Ngân sách: ${info.budget.calls} lượt gọi/phiên, ${info.budget.requests_per_minute} request/phút.`;
    } catch (error) {
      $("#diagnostics").textContent = error.message;
    }
  }),
);
$("#clear-cache").addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  try {
    const result = await api("/cache/clear", { method: "POST" });
    toast(`Đã xóa ${result.deleted} phản hồi đã lưu.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    event.currentTarget.disabled = false;
  }
});
$("#import-benchmark").addEventListener("click", () =>
  $("#benchmark-input").click(),
);
$("#benchmark-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 8 * 1024 * 1024)
      throw new Error("File benchmark vượt quá 8 MB.");
    renderBenchmark(JSON.parse(await file.text()));
    toast("Đã nhập kết quả benchmark.");
  } catch (error) {
    toast(error.message || "JSON không hợp lệ.", true);
  } finally {
    event.target.value = "";
  }
});
$$(".close-dialog").forEach((button) =>
  button.addEventListener("click", () => button.closest("dialog").close()),
);
$$("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog && dialog.id !== "login-dialog") {
      const bounds = dialog.getBoundingClientRect();
      if (
        event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom
      )
        dialog.close();
    }
  }),
);
const mobileLayout = window.matchMedia?.("(max-width: 800px)");
function setSidebarOpen(open) {
  $("#sidebar").classList.toggle("open", open);
  $("#sidebar-backdrop").hidden = !open;
  $("#sidebar").inert = Boolean(mobileLayout?.matches && !open);
  $(".content-shell").inert = Boolean(mobileLayout?.matches && open);
  document.body.classList.toggle(
    "menu-open",
    Boolean(mobileLayout?.matches && open),
  );
  $("#mobile-menu").setAttribute("aria-expanded", String(open));
  $("#mobile-menu").setAttribute("aria-label", open ? "Đóng menu" : "Mở menu");
  if (open) $(".main-nav .active").focus();
}
mobileLayout?.addEventListener("change", () => setSidebarOpen(false));
setSidebarOpen(false);
$("#mobile-menu").addEventListener("click", () =>
  setSidebarOpen(!$("#sidebar").classList.contains("open")),
);
$("#sidebar-backdrop").addEventListener("click", () => {
  setSidebarOpen(false);
  $("#mobile-menu").focus();
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Tab" &&
    mobileLayout?.matches &&
    $("#sidebar").classList.contains("open")
  ) {
    const focusable = $$(
      "#sidebar a, #sidebar button:not(:disabled), #sidebar input:not(:disabled):not([hidden])",
    );
    const first = focusable[0];
    const last = focusable.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  if (event.key === "Escape" && $("#sidebar").classList.contains("open")) {
    setSidebarOpen(false);
    $("#mobile-menu").focus();
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (!document.querySelector("dialog[open]")) openLibrarySearch();
  }
});
function openLibrarySearch() {
  switchView("library");
  $("#library-search").focus();
}
$("#workspace-search").addEventListener("click", openLibrarySearch);
$("#library-search").addEventListener("input", renderLibrary);
window.addEventListener("research-result-rendered", updateResearchSteps);
$$(".tab").forEach((button) =>
  button.addEventListener("click", () => selectTab(button.dataset.tab)),
);
$("#copy-button").addEventListener("click", copyReport);
window.addEventListener("authentication-required", showLogin);
$("#login-dialog").addEventListener("cancel", (event) =>
  event.preventDefault(),
);
$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/auth/login", {
      method: "POST",
      body: { token: $("#access-token").value },
    });
    $("#access-token").value = "";
    $("#login-error").textContent = "";
    $("#login-dialog").close();
    await initialize();
setupUXEnhancements();
setupSettingsListeners();
setupRemoteWorkerListeners();
renderSettingsView();
  } catch (error) {
    $("#login-error").textContent = error.message;
  }
});
$("#logout-button").addEventListener("click", async () => {
  try {
    await api("/auth/logout", { method: "POST" });
    localStorage.removeItem("research-active-run");
    window.location.reload();
  } catch (error) {
    toast(error.message, true);
  }
});
initialize();
setupUXEnhancements();
setupSettingsListeners();
setupRemoteWorkerListeners();
renderSettingsView();

$("#architecture-button").addEventListener("click", async () => {
  const root = $("#architecture-content");
  root.replaceChildren(el("p", "loading-label", "Đang tải luồng agent…"));
  $("#architecture-dialog").showModal();
  try {
    const data = await api("/agents");
    root.replaceChildren();
    data.agents.forEach((agent, index) => {
      const card = el("article", "architecture-card");
      card.append(
        el("small", "", `0${index + 1} · ${agent.role}`),
        el("h3", "", agent.purpose),
      );
      const list = el("ol");
      agent.stages.forEach((stage) => list.append(el("li", "", stage)));
      card.append(list, el("p", "", agent.guard));
      root.append(card);
    });
  } catch (error) {
    root.replaceChildren(el("p", "", error.message));
  }
});

function renderScopePicker() {
  const root = $("#scope-documents");
  const filter = $("#scope-search").value.toLocaleLowerCase("vi");
  root.replaceChildren();
  for (const doc of state.documents.filter((d) =>
    d.filename.toLocaleLowerCase("vi").includes(filter),
  )) {
    const row = el("label", "scope-document");
    const box = el("input");
    box.type = "checkbox";
    box.checked = state.selected.has(doc.id);
    box.disabled = state.working;
    box.setAttribute("aria-label", `Chọn nguồn ${doc.filename}`);
    box.addEventListener("change", () => {
      box.checked ? state.selected.add(doc.id) : state.selected.delete(doc.id);
      state.scopeMode = "selected";
      renderDocuments();
    });
    const text = el("div");
    text.append(
      el("strong", "", doc.filename),
      el(
        "small",
        "",
        `${doc.page_count || "—"} trang · ${doc.chunk_count} đoạn`,
      ),
    );
    row.append(box, text);
    root.append(row);
  }
  if (!root.children.length)
    root.append(el("p", "", "Không có tài liệu phù hợp."));
}
function activityRecord(record) {
  $("#activity-run").textContent =
    `${record.request.question} · ${recordStatusLabel(record)}`;
  $("#export-trace").hidden = false;
  $("#export-trace").href = `/api/runs/${record.id}/trace`;
  const documents = record.scope_documents || [];
  $("#activity-sources").textContent = documents.length
    ? documents.map((doc) => doc.filename).join(" · ")
    : "Phạm vi được lưu trong nhật ký phiên.";
}
$("#scope-mode").addEventListener("change", (e) => {
  state.scopeMode = e.target.value;
  renderScope();
});
$("#choose-sources").addEventListener("click", () =>
  $("#scope-dialog").showModal(),
);
$("#scope-search").addEventListener("input", renderScopePicker);
$("#scope-clear").addEventListener("click", () => {
  if (!state.working) {
    state.selected.clear();
    state.scopeMode = "selected";
    renderDocuments();
  }
});
$("#back-research").addEventListener("click", () => {
  switchView("research");
  if (!$("#results-section").hidden)
    $("#results-section").scrollIntoView?.({ block: "start" });
});
$("#activity-cancel").addEventListener("click", cancelResearch);
$("#trace-filter").addEventListener("change", (e) => {
  $$(".trace-item").forEach(
    (row) =>
      (row.hidden =
        e.target.value !== "all" && row.dataset.agent !== e.target.value),
  );
});
$("#rebuild-button").addEventListener("click", async (e) => {
  if (
    !window.confirm(
      "Chia lại toàn bộ tài liệu bằng cấu hình hiện tại? Báo cáo cũ được giữ lại.",
    )
  )
    return;
  const button = e.currentTarget;
  button.disabled = true;
  try {
    const result = await api("/documents/rebuild", { method: "POST" });
    await refreshDocuments();
    toast(`Đã xây dựng lại ${result.chunks_updated} đoạn.`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
});


// Quick Prompt Chips & Mode Switch
function setupUXEnhancements() {
  $$(".quick-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = $("#question");
      if (q) {
        q.value = chip.dataset.prompt;
        const count = $("#question-count");
        if (count) count.textContent = `${q.value.length} / 2000`;
        q.focus();
      }
    });
  });

  const btnGemini = $("#btn-mode-gemini");
  const btnKaggle = $("#btn-mode-kaggle");

  function updateModeButtons(mode) {
    btnGemini?.classList.toggle("active", mode === "local_api");
    btnKaggle?.classList.toggle("active", mode === "kaggle_gpu");
  }

  btnGemini?.addEventListener("click", async () => {
    try {
      const cfg = await api("/config/mode", { method: "POST", body: { mode: "local_api" } });
      state.config = cfg;
      renderConfig();
      updateModeButtons("local_api");
      toast("Đã chuyển sang chế độ ⚡ Gemini API (1-2s)");
    } catch (err) {
      toast(err.message, true);
    }
  });

  btnKaggle?.addEventListener("click", async () => {
    try {
      const cfg = await api("/config/mode", { method: "POST", body: { mode: "kaggle_gpu" } });
      state.config = cfg;
      renderConfig();
      updateModeButtons("kaggle_gpu");
      toast("Đã chuyển sang chế độ 🚀 Kaggle Dual T4 GPU");
    } catch (err) {
      toast(err.message, true);
    }
  });

  if (state.config?.execution_mode) {
    updateModeButtons(state.config.execution_mode);
  }
}


function renderSettingsView() {
  const cfg = state.config;
  if (!cfg) return;

  const agentModes = cfg.agent_modes || {};
  const modelCount = Object.values(agentModes).filter((m) => m === "model").length;
  const cpuCount = Object.values(agentModes).filter((m) => m === "cpu").length;
  const apiCount = Object.values(agentModes).filter((m) => m === "api").length;
  const isAllModel = modelCount === 5;
  const isAllCpu = cpuCount === 5;
  const isAllApi = apiCount === 5;
  const activeModeSummary = isAllModel ? "Kaggle Dual T4 GPU" : (isAllCpu ? "Local CPU (Máy tính)" : (isAllApi ? "Google Gemini Cloud API" : `Hybrid (${apiCount} API, ${cpuCount} CPU, ${modelCount} GPU)`));

  const modelName = $("#settings-model-name");
  if (modelName) modelName.textContent = activeModeSummary;

  const stProvider = $("#st-provider");
  if (stProvider) stProvider.textContent = isAllApi ? "Google Gemini API" : (isAllModel ? "Kaggle Dual T4 (Qwen2.5 + DocLayNet)" : "Tích hợp Hybrid Đa nền tảng");

  const stModel = $("#st-model");
  if (stModel) stModel.textContent = cfg.default_model || "gemini-3.1-flash-lite";

  const stFallback = $("#st-fallback");
  if (stFallback) stFallback.textContent = "Qwen2.5-7B LoRA / gemini-3-flash-preview";

  const remoteInput = $("#input-remote-url");
  if (remoteInput && !remoteInput.value) {
    remoteInput.value = cfg.remote_worker || "";
  }
  const cpuUrlInput = $("#input-local-cpu-url");
  if (cpuUrlInput && cfg.local_cpu_url) {
    cpuUrlInput.value = cfg.local_cpu_url;
  }
  const cpuModelInput = $("#input-local-cpu-model");
  if (cpuModelInput && cfg.local_cpu_model) {
    cpuModelInput.value = cfg.local_cpu_model;
  }
  checkLocalCpuStatus();
  checkTunnelStatus();

  // Render Per-Agent Modes
  const agentDescriptions = {
    retrieval: {
      api: "⚡ Gemini API (Decomposition)",
      cpu: "💻 CPU Router (BM25 + Regex/0.5B)",
      model: "🚀 Qwen-7B Router LoRA (Kaggle T4)",
    },
    reader: {
      api: "⚡ PyMuPDF & Gemini Guardrail",
      cpu: "💻 PyMuPDF CPU Layout Engine",
      model: "🚀 LayoutLMv3 DocLayNet (Segmentation)",
    },
    summarizer: {
      api: "⚡ Gemini API (Generator & Critic)",
      cpu: "💻 Local Qwen-1.5B (Ollama CPU)",
      model: "🚀 Generator-Critic LoRA (Dual T4)",
    },
    factchecker: {
      api: "⚡ Gemini NLI & Numeric Verifier",
      cpu: "💻 Local Rule & Numeric Verifier (CPU)",
      model: "🚀 Qwen-3B SciFact (GPU 1)",
    },
    writer: {
      api: "⚡ Gemini API (Academic Synthesis)",
      cpu: "💻 Local Qwen-3B (Ollama CPU)",
      model: "🚀 Qwen2.5-7B Instruct (GPU 0)",
    },
  };

  Object.entries(agentModes).forEach(([agent, curMode]) => {
    const row = document.querySelector(`.agent-mode-row[data-agent="${agent}"]`);
    if (row) {
      row.querySelectorAll(".btn-agent-mode").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === curMode);
      });
      const descEl = document.getElementById(`desc-${agent}`);
      if (descEl && agentDescriptions[agent]) {
        descEl.textContent = `Đang dùng: ${agentDescriptions[agent][curMode] || curMode.toUpperCase()}`;
      }
    }
  });
}

function setupSettingsListeners() {
  const radios = document.querySelectorAll('input[name="exec-mode"]');
  radios.forEach((radio) => {
    radio.addEventListener("change", async (e) => {
      const newMode = e.target.value;
      try {
        const updated = await api("/config/mode", { method: "POST", body: { mode: newMode } });
        state.config = updated;
        renderConfig();
        renderSettingsView();
        const modeLabel = newMode === "local_api" ? "⚡ Google Gemini API" : "🚀 Kaggle Dual T4 GPU";
        toast(`Đã chuyển thành công sang ${modeLabel}`);
      } catch (err) {
        toast(err.message, true);
      }
    });
  });

  $("#btn-settings-clear-cache")?.addEventListener("click", async () => {
    try {
      await api("/cache/clear", { method: "POST" });
      toast("Đã xóa sạch bộ nhớ đệm LLM Cache!");
    } catch (err) {
      toast(err.message, true);
    }
  });

  $("#btn-settings-rebuild-fts")?.addEventListener("click", async () => {
    try {
      const res = await api("/documents/rebuild", { method: "POST" });
      toast(`Đã tái tạo chỉ mục C-Speed FTS5 (${res.chunks_updated || 8} đoạn)!`);
    } catch (err) {
      toast(err.message, true);
    }
  });

  // Per-Agent Mode Toggle Buttons
  document.querySelectorAll(".btn-agent-mode").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const agent = btn.dataset.agent;
      const mode = btn.dataset.mode;
      try {
        const res = await api("/config/agents", {
          method: "POST",
          body: { agent, mode },
        });
        state.config = res.config;
        renderSettingsView();
        const modeTitle = mode === "api" ? "⚡ Cloud API" : "🚀 Model / LoRA";
        toast(`Đã chuyển ${agent.toUpperCase()} sang ${modeTitle} (Tự động ghi vào .env)!`);
      } catch (err) {
        toast(err.message, true);
      }
    });
  });

  // Presets
  const applyPreset = async (presetModes, label) => {
    try {
      const res = await api("/config/agents", {
        method: "POST",
        body: { agent_modes: presetModes },
      });
      state.config = res.config;
      renderSettingsView();
      toast(`Đã áp dụng cấu hình: ${label} (Đã lưu .env)!`);
    } catch (err) {
      toast(err.message, true);
    }
  };

  $("#preset-all-cpu")?.addEventListener("click", () => {
    applyPreset({
      retrieval: "cpu",
      reader: "cpu",
      summarizer: "cpu",
      factchecker: "cpu",
      writer: "cpu",
    }, "💻 Toàn bộ CPU (Máy tính cá nhân)");
  });

  $("#preset-all-api")?.addEventListener("click", () => {
    applyPreset({
      retrieval: "api",
      reader: "api",
      summarizer: "api",
      factchecker: "api",
      writer: "api",
    }, "⚡ Toàn bộ API (Gemini Cloud)");
  });

  $("#preset-all-model")?.addEventListener("click", () => {
    applyPreset({
      retrieval: "model",
      reader: "model",
      summarizer: "model",
      factchecker: "model",
      writer: "model",
    }, "🚀 Toàn bộ Model (Kaggle Dual T4)");
  });

  $("#preset-hybrid")?.addEventListener("click", () => {
    applyPreset({
      retrieval: "model",
      reader: "api",
      summarizer: "model",
      factchecker: "model",
      writer: "api",
    }, "⚖️ Hybrid Tối ưu (Router & Generator Model, Reader & Writer API)");
  });
}



async function checkLocalCpuStatus() {
  const badge = $("#local-cpu-status");
  if (!badge) return;
  badge.textContent = "Đang kiểm tra CPU...";
  badge.className = "tunnel-status-badge";
  try {
    const res = await api("/config/local-cpu/test");
    if (res.is_online) {
      badge.textContent = `🟢 ${res.details}`;
      badge.className = "tunnel-status-badge online";
    } else {
      badge.textContent = `⚪ Chưa bật (${res.details})`;
      badge.className = "tunnel-status-badge offline";
    }
  } catch (err) {
    badge.textContent = "⚪ Offline";
    badge.className = "tunnel-status-badge offline";
  }
}

async function checkTunnelStatus() {
  const badge = $("#tunnel-status");
  if (!badge) return;
  badge.textContent = "Đang kiểm tra kết nối...";
  badge.className = "tunnel-status-badge";
  try {
    const res = await api("/config/remote-url/test");
    if (res.is_online) {
      badge.textContent = `🟢 ${res.details}`;
      badge.className = "tunnel-status-badge online";
    } else {
      badge.textContent = `🔴 Offline - ${res.details}`;
      badge.className = "tunnel-status-badge offline";
    }
  } catch (err) {
    badge.textContent = "🔴 Không phản hồi";
    badge.className = "tunnel-status-badge offline";
  }
}

function setupRemoteWorkerListeners() {
  $("#btn-save-local-cpu")?.addEventListener("click", async () => {
    const urlVal = $("#input-local-cpu-url")?.value?.trim() || "http://localhost:11434";
    const modelVal = $("#input-local-cpu-model")?.value?.trim() || "qwen2.5:1.5b";
    const btn = $("#btn-save-local-cpu");
    btn.disabled = true;
    btn.textContent = "Đang lưu...";
    try {
      const res = await api("/config/local-cpu", {
        method: "POST",
        body: { local_cpu_url: urlVal, local_cpu_model: modelVal },
      });
      state.config = res.config;
      renderConfig();
      toast(res.is_online ? `Đã lưu cấu hình CPU và kết nối thành công (${res.details})` : `Đã lưu cấu hình CPU vào .env (${res.details})`);
      await checkLocalCpuStatus();
    } catch (err) {
      toast(err.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "Lưu cấu hình CPU";
    }
  });

  $("#btn-test-local-cpu")?.addEventListener("click", async () => {
    const btn = $("#btn-test-local-cpu");
    btn.disabled = true;
    btn.textContent = "Đang test...";
    try {
      await checkLocalCpuStatus();
      toast("Kiểm tra kết nối CPU hoàn tất.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Kiểm tra CPU";
    }
  });

  $("#btn-save-remote-url")?.addEventListener("click", async () => {
    const val = $("#input-remote-url")?.value?.trim();
    if (!val) {
      toast("Vui lòng nhập đường dẫn URL Cloudflare Tunnel hợp lệ.", true);
      return;
    }
    const btn = $("#btn-save-remote-url");
    btn.disabled = true;
    btn.textContent = "Đang lưu...";
    try {
      const res = await api("/config/remote-url", {
        method: "POST",
        body: { remote_url: val },
      });
      state.config = res.config;
      renderConfig();
      toast(res.is_online ? "Đã lưu vào .env và kết nối Kaggle GPU thành công!" : `Đã lưu vào .env (${res.details})`);
      await checkTunnelStatus();
    } catch (err) {
      toast(err.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "Lưu & Tự động ghi vào .env";
    }
  });

  $("#btn-test-remote-url")?.addEventListener("click", async () => {
    const btn = $("#btn-test-remote-url");
    btn.disabled = true;
    btn.textContent = "Đang test...";
    try {
      await checkTunnelStatus();
      toast("Kiểm tra kết nối Kaggle hoàn tất.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Kiểm tra kết nối";
    }
  });
}


/* ==================== PIPELINE / AGENT 1 STUDIO LOGIC ==================== */

let pipeSelectedMode = "api";
let pipeModesStatus = {
  cpu: { available: true, is_online: true },
  api: { available: false, is_online: false },
  model: { available: false, is_online: false },
};

function escapePipeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML;
}

async function checkPipelineStatus() {
  const cpuPill = $("#pipe-status-cpu");
  const apiPill = $("#pipe-status-api");
  const modelPill = $("#pipe-status-model");
  if (!cpuPill) return;

  try {
    const data = await api("/pipeline/agent1/status");
    if (data && data.modes) {
      pipeModesStatus = data.modes;

      // 1. CPU
      cpuPill.textContent = "🟢 Sẵn sàng 100%";
      cpuPill.className = "pipe-status-pill status-ready";

      // 2. API
      if (data.modes.api && data.modes.api.available) {
        apiPill.textContent = "🟢 Đã cấu hình Key";
        apiPill.className = "pipe-status-pill status-ready";
      } else {
        apiPill.textContent = "🔴 Chưa cấu hình Key";
        apiPill.className = "pipe-status-pill status-offline";
      }

      // 3. Model
      if (data.modes.model && data.modes.model.is_online) {
        modelPill.textContent = "🟢 Worker Online";
        modelPill.className = "pipe-status-pill status-ready";
      } else {
        modelPill.textContent = "🔴 Chưa kích hoạt";
        modelPill.className = "pipe-status-pill status-offline";
      }

      updatePipeWarningBanner();
    }
  } catch (err) {
    console.warn("Failed to check pipeline status:", err);
  }
}

function updatePipeWarningBanner() {
  const banner = $("#pipe-mode-warning-banner");
  const title = $("#pipe-warning-title");
  const desc = $("#pipe-warning-desc");
  if (!banner) return;

  const status = pipeModesStatus[pipeSelectedMode];
  if (!status || !status.is_online) {
    banner.hidden = false;
    if (pipeSelectedMode === "model") {
      title.textContent = "Chế độ Kaggle GPU chưa được kích hoạt!";
      desc.textContent = "Worker GPU đang offline hoặc chưa cấu hình URL Cloudflare Tunnel trong Cài đặt. Hãy khởi động Notebook trên Kaggle hoặc chuyển sang Local CPU / Cloud API.";
    } else if (pipeSelectedMode === "api") {
      title.textContent = "Chế độ Cloud API chưa được kích hoạt!";
      desc.textContent = "Chưa tìm thấy GEMINI_API_KEY trong file .env. Hãy cấu hình API Key hoặc chuyển sang chế độ Local CPU.";
    } else {
      banner.hidden = true;
    }
  } else {
    banner.hidden = true;
  }
}

function initPipelineStudio() {
  initAgentStepper();

  // Mode selection buttons (Cloud API, Kaggle GPU, and Baseline Local CPU)
  $$(".pipe-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".pipe-mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      pipeSelectedMode = btn.dataset.mode;
      updatePipeWarningBanner();
    });
  });

  // Toggle button for Baseline Local CPU
  const toggleBaselineBtn = $("#pipe-toggle-baseline-btn");
  const baselineContainer = $("#pipe-baseline-container");
  const baselineIcon = $("#pipe-baseline-toggle-icon");
  if (toggleBaselineBtn && baselineContainer) {
    toggleBaselineBtn.addEventListener("click", () => {
      const isHidden = baselineContainer.hasAttribute("hidden");
      if (isHidden) {
        baselineContainer.removeAttribute("hidden");
        toggleBaselineBtn.setAttribute("aria-expanded", "true");
        if (baselineIcon) baselineIcon.textContent = "▲";
      } else {
        baselineContainer.setAttribute("hidden", "");
        toggleBaselineBtn.setAttribute("aria-expanded", "false");
        if (baselineIcon) baselineIcon.textContent = "▼";
      }
    });
  }

  // Guide Dialog (Tìm hiểu cách hoạt động của Agent 1)
  const guideBtn = $("#pipe-btn-learn-agent1");
  const guideDialog = $("#pipe-agent1-guide-dialog");
  const guideCloseBtn = $("#pipe-guide-close-btn");
  const guideConfirmBtn = $("#pipe-guide-confirm-btn");

  if (guideBtn && guideDialog) {
    guideBtn.addEventListener("click", () => {
      guideDialog.showModal();
    });
    guideCloseBtn?.addEventListener("click", () => {
      guideDialog.close();
    });
    guideConfirmBtn?.addEventListener("click", () => {
      guideDialog.close();
    });
    guideDialog.addEventListener("click", (e) => {
      if (e.target === guideDialog) {
        guideDialog.close();
      }
    });
  }

  // Prompt chips
  $$(".pipe-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.dataset.query;
      if (q && $("#pipe-query-input")) {
        $("#pipe-query-input").value = q;
        $("#pipe-query-input").focus();
      }
    });
  });

  // Refresh status button
  $("#pipe-btn-refresh-status")?.addEventListener("click", () => {
    checkPipelineStatus();
    toast("Đã kiểm tra lại kết nối các chế độ.");
  });

  // Go to settings button
  $("#pipe-btn-goto-settings")?.addEventListener("click", () => {
    switchView("settings");
  });

  // Execute Agent 1 button
  $("#btn-pipe-execute")?.addEventListener("click", async () => {
    const queryInput = $("#pipe-query-input");
    const query = queryInput ? queryInput.value.trim() : "";
    if (!query) {
      alert("Vui lòng nhập câu hỏi kiểm thử cho Agent 1.");
      return;
    }

    // Pre-flight check before execution
    const modeStatus = pipeModesStatus[pipeSelectedMode];
    if (pipeSelectedMode !== "cpu" && modeStatus && !modeStatus.is_online) {
      updatePipeWarningBanner();
      alert(`⚠️ Chế độ ${pipeSelectedMode === "model" ? "Kaggle GPU" : "Cloud API"} chưa được kích hoạt! Vui lòng chọn chế độ khác hoặc bật kết nối trong tab Cài đặt.`);
      return;
    }

    const mockGap = $("#pipe-mock-gap")?.checked || false;

    // Show loading
    $("#pipe-empty-state").hidden = true;
    $("#pipe-output-container").hidden = true;
    $("#pipe-loading-state").hidden = false;
    const btn = $("#btn-pipe-execute");
    if (btn) btn.disabled = true;

    try {
      const res = await api("/pipeline/agent1/run", {
        method: "POST",
        body: {
          query: query,
          mode: pipeSelectedMode,
          mock_evidence_gap: mockGap,
          retry_count: mockGap ? 1 : 0,
        },
      });

      $("#pipe-loading-state").hidden = true;

      if (!res.success) {
        $("#pipe-mode-warning-banner").hidden = false;
        $("#pipe-warning-title").textContent = "Không thể thực thi";
        $("#pipe-warning-desc").textContent = res.message || "Đã xảy ra lỗi khi chạy Agent 1.";
        return;
      }

      // Populate Metrics
      const tool = (res.router_decision?.tool_selected || "bm25").toUpperCase();
      $("#res-tool-selected").textContent = tool;
      $("#res-latency").textContent = `${res.latency_ms} ms`;
      $("#res-mode-used").textContent = {
        cpu: "Local CPU (FTS5)",
        api: "Cloud API (Gemini)",
        model: "Kaggle Dual T4",
      }[res.mode_used] || res.mode_used;
      $("#res-total-retrieved").textContent = `${res.total_retrieved} chunks`;

      // Populate Router & Critique
      $("#res-router-thought").textContent = res.router_decision?.thought || "Router xác định công cụ phù hợp với cấu trúc câu hỏi.";
      const isRelevant = res.router_decision?.critique_is_relevant;
      const critiqueBadge = $("#res-critique-badge");
      if (critiqueBadge) {
        critiqueBadge.textContent = isRelevant ? "✓ Đạt yêu cầu" : "⚠️ Cần bổ sung nguồn";
        critiqueBadge.className = `pipe-critique-pill ${isRelevant ? "passed" : "failed"}`;
      }
      $("#res-critique-reason").textContent = res.router_decision?.critique_reason || "Ngữ cảnh trích xuất khớp với từ khóa truy vấn.";

      // Populate STORM Perspectives
      const perspCount = $("#res-perspectives-count");
      if (perspCount) perspCount.textContent = `${(res.perspectives || []).length} góc nhìn`;
      const perspContainer = $("#res-perspectives-chips");
      if (perspContainer) {
        perspContainer.innerHTML = "";
        (res.perspectives || []).forEach((p) => {
          const badge = document.createElement("span");
          badge.className = "pipe-perspective-badge";
          badge.textContent = p;
          perspContainer.appendChild(badge);
        });
      }

      // Populate Sub-queries
      const subqList = $("#res-subqueries-list");
      if (subqList) {
        subqList.innerHTML = "";
        (res.sub_queries || []).forEach((sq) => {
          const li = document.createElement("li");
          li.textContent = sq;
          subqList.appendChild(li);
        });
      }

      // Populate Retrieved Documents with Click-to-View Modal
      const docsCount = $("#res-docs-count");
      if (docsCount) docsCount.textContent = `${(res.retrieved_docs || []).length} kết quả`;
      const docsContainer = $("#res-docs-list");
      if (docsContainer) {
        docsContainer.innerHTML = "";
        (res.retrieved_docs || []).forEach((doc, idx) => {
          const item = document.createElement("div");
          const isFallback = Boolean(doc.is_fallback);
          item.className = `pipe-doc-item ${isFallback ? "is-fallback" : ""}`;
          item.title = "Nhấn để xem toàn bộ nội dung tài liệu này";

          const fallbackTag = isFallback ? '<span class="pipe-doc-fallback-tag">⚠️ Truy vấn dự phòng</span>' : '<span class="pipe-doc-fallback-tag" style="background:#e0f2fe;color:#0369a1;">📄 Bài báo ArXiv</span>';

          item.innerHTML = `
            <div class="pipe-doc-header">
              <div class="pipe-doc-title-row">
                <span class="pipe-rank-badge">#${idx + 1}</span>
                <span class="pipe-doc-id">${escapePipeHtml(doc.doc_id || doc.id)}</span>
                ${fallbackTag}
                <small>(Trang ${doc.page_no || 1})</small>
              </div>
              <span class="pipe-rrf-score">RRF: ${doc.rrf_score || 0}</span>
            </div>
            <p class="pipe-doc-snippet">${escapePipeHtml((doc.text || "").slice(0, 240))}${(doc.text || "").length > 240 ? "..." : ""}</p>
            <div class="pipe-click-hint">
              <span>🔍 Nhấn để xem toàn văn & đối soát bằng chứng ➔</span>
            </div>
          `;

          item.addEventListener("click", () => {
            openPipeDocModal(doc, idx + 1);
          });

          docsContainer.appendChild(item);
        });
      }
      lastAgent1RetrievedDocs = res.retrieved_docs || [];

      // Populate Section 4: All Agent 1 Outputs Summary & JSON Payload
      const strategyText = res.retrieval_strategy || `${res.router_decision?.tool_selected || "dense"}_hybrid_rrf`;
      $("#out-strategy-summary").textContent = strategyText;
      $("#out-docs-summary").textContent = `${res.total_retrieved || 0} tài liệu đã trích xuất & ranker RRF`;
      $("#out-subqueries-summary").textContent = `${(res.sub_queries || []).length} truy vấn con`;
      $("#out-perspectives-summary").textContent = `${(res.perspectives || []).length} góc nhìn học thuật`;
      $("#out-decision-summary").textContent = `Tool: ${(res.router_decision?.tool_selected || "").toUpperCase()} | Phê bình: ${res.router_decision?.critique_is_relevant ? "Đạt" : "Cần bổ sung"}`;
      $("#out-seen-summary").textContent = `Tổng tích lũy: ${res.total_seen || res.total_retrieved || 0} chunks (Tránh mất dấu bằng chứng)`;
      $("#out-early-stop-summary").textContent = `${res.no_new_evidence ? "True (Dừng sớm)" : "False (Bằng chứng mới được ghi nhận)"}`;

      // Populate Raw JSON State Block
      const rawJsonPre = $("#res-agent1-raw-json");
      const statePayload = res.agent1_state_payload || {
        sub_queries: res.sub_queries,
        perspectives: res.perspectives,
        retrieved_docs: res.retrieved_docs,
        retrieval_strategy: strategyText,
        router_decision: res.router_decision,
        no_new_evidence: res.no_new_evidence,
      };
      if (rawJsonPre) {
        rawJsonPre.textContent = JSON.stringify(statePayload, null, 2);
      }

      // Wire copy JSON State button
      const copyPayloadBtn = $("#btn-copy-agent1-payload");
      if (copyPayloadBtn) {
        copyPayloadBtn.onclick = async () => {
          try {
            await navigator.clipboard.writeText(JSON.stringify(statePayload, null, 2));
            toast("Đã sao chép toàn bộ gói State Payload của Agent 1!");
          } catch {
            toast("Không thể sao chép dữ liệu.");
          }
        };
      }

      // Show output
      $("#pipe-output-container").hidden = false;

    } catch (err) {
      $("#pipe-loading-state").hidden = true;
      alert("Lỗi kết nối khi gọi Agent 1: " + err.message);
    } finally {
      if (btn) btn.disabled = false;
    }
  });
}


function openPipeDocModal(doc, rank) {
  const dialog = $("#pipe-doc-dialog");
  if (!dialog) return;

  $("#modal-doc-rank").textContent = `#${rank}`;
  $("#pipe-doc-dialog-title").textContent = doc.title || doc.doc_id || "Đoạn trích tài liệu";
  $("#modal-doc-meta").textContent = `${doc.doc_id || "N/A"} · Trang ${doc.page_no || 1} · Điểm RRF: ${doc.rrf_score || 0}`;
  $("#modal-context-hash").textContent = doc.context_hash || "N/A";
  $("#modal-full-text").textContent = doc.text || "(Trống)";

  const fallbackAlert = $("#modal-fallback-alert");
  if (fallbackAlert) {
    fallbackAlert.hidden = !Boolean(doc.is_fallback);
  }

  // Close handlers
  const closeBtn1 = $("#btn-close-doc-dialog");
  const closeBtn2 = $("#btn-close-doc-dialog-2");
  const copyBtn = $("#btn-copy-doc-text");

  if (closeBtn1) closeBtn1.onclick = () => dialog.close();
  if (closeBtn2) closeBtn2.onclick = () => dialog.close();
  if (copyBtn) {
    copyBtn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(doc.text || "");
        toast("Đã sao chép toàn bộ nội dung tài liệu!");
      } catch {
        toast("Không thể tự động sao chép.");
      }
    };
  }

  dialog.showModal();
}


/* ==================== PIPELINE AGENT 2 STUDIO & STEPPER LOGIC ==================== */

let lastAgent1RetrievedDocs = [];

function initAgentStepper() {
  // Navigation between Agent 1 and Agent 2 panels
  const btnStep1 = $("#pipe-nav-agent1");
  const btnStep2 = $("#pipe-nav-agent2");
  const panel1 = $("#pipe-agent1-panel");
  const panel2 = $("#pipe-agent2-panel");

  function switchToStep(step) {
    if (step === "agent1") {
      btnStep1?.classList.add("active");
      btnStep2?.classList.remove("active");
      if (panel1) panel1.hidden = false;
      if (panel2) panel2.hidden = true;
    } else {
      btnStep1?.classList.remove("active");
      btnStep2?.classList.add("active");
      if (panel1) panel1.hidden = true;
      if (panel2) panel2.hidden = false;

      // Update count of docs received from Agent 1
      const countLabel = $("#agent2-input-docs-count");
      if (countLabel) {
        countLabel.textContent = `${lastAgent1RetrievedDocs.length} tài liệu đã trích xuất`;
      }
    }
  }

  btnStep1?.addEventListener("click", () => switchToStep("agent1"));
  btnStep2?.addEventListener("click", () => switchToStep("agent2"));

  // Handoff button from Agent 1 results to Agent 2
  $("#btn-handoff-to-agent2")?.addEventListener("click", () => {
    switchToStep("agent2");
    // Automatically select the "from agent1" radio option
    const radioA1 = $("#agent2-src-from-agent1");
    if (radioA1) radioA1.checked = true;
    toast("Đã chuyển dữ liệu sang Agent 2 (Reader). Nhấn 'Chạy Thử Nghiệm Agent 2' để bóc tách!");
  });

  // Engine selection buttons for Agent 2
  let agent2SelectedEngine = "pymupdf";
  $$("#pipe-agent2-panel .pipe-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("#pipe-agent2-panel .pipe-mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      agent2SelectedEngine = btn.dataset.engine || "pymupdf";
    });
  });

  // Execute Agent 2 button
  $("#btn-agent2-execute")?.addEventListener("click", async () => {
    const isFromAgent1 = $("#agent2-src-from-agent1")?.checked;
    const docsToSend = isFromAgent1 ? lastAgent1RetrievedDocs : [];
    const pdfPath = isFromAgent1 ? null : "sample.pdf";

    // Show loading
    $("#agent2-empty-state").hidden = true;
    $("#agent2-output-container").hidden = true;
    $("#agent2-loading-state").hidden = false;
    const btn = $("#btn-agent2-execute");
    if (btn) btn.disabled = true;

    try {
      const res = await api("/pipeline/agent2/run", {
        method: "POST",
        body: {
          retrieved_docs: docsToSend,
          pdf_path: pdfPath,
          mode: agent2SelectedEngine === "doclaynet" ? "model" : "cpu",
        },
      });

      $("#agent2-loading-state").hidden = true;

      if (!res.success) {
        alert("Lỗi khi chạy Agent 2: " + (res.message || "Không xác định"));
        return;
      }

      // Metrics
      $("#res-agent2-total-units").textContent = `${res.total_units || 0} units`;
      $("#res-agent2-latency").textContent = `${res.latency_ms || 0} ms`;
      $("#res-agent2-sections").textContent = `${res.sections_count || 0} sections`;
      $("#res-agent2-source-name").textContent = res.pdf_path_used ? `File: ${res.pdf_path_used}` : "Agent 1 Chunks";

      // Sections List
      const secContainer = $("#res-agent2-sections-list");
      if (secContainer) {
        secContainer.innerHTML = "";
        const sections = res.document_schema?.sections || [];
        if (!sections.length) {
          secContainer.innerHTML = '<p class="pipe-empty-note">Không có section cấu trúc đặc biệt (Dùng chunk phẳng).</p>';
        } else {
          sections.slice(0, 8).forEach((sec) => {
            const item = document.createElement("div");
            item.className = "pipe-section-badge-item";
            item.innerHTML = `
              <span>${escapePipeHtml(sec.title || sec.unit_id || "Section")}</span>
              <span class="pipe-section-tag">${sec.category || "Section"}</span>
            `;
            secContainer.appendChild(item);
          });
        }
      }

      // SourceUnits List
      const unitsContainer = $("#res-agent2-units-list");
      if (unitsContainer) {
        unitsContainer.innerHTML = "";
        const units = res.source_units || [];
        units.forEach((u, idx) => {
          const item = document.createElement("div");
          item.className = "pipe-doc-item";
          item.title = "Nhấn để xem toàn bộ SourceUnit";
          item.innerHTML = `
            <div class="pipe-doc-header">
              <div class="pipe-doc-title-row">
                <span class="pipe-rank-badge">Unit #${idx + 1}</span>
                <span class="pipe-doc-id">${escapePipeHtml(u.id || `unit_${idx}`)}</span>
                <small>(Trang ${u.page_no || 1} · Offset: ${u.character_offset || 0})</small>
              </div>
              <span class="pipe-rrf-score">Hash: ${(u.context_hash || "").slice(0, 8)}...</span>
            </div>
            <p class="pipe-doc-snippet">${escapePipeHtml((u.text || "").slice(0, 200))}${(u.text || "").length > 200 ? "..." : ""}</p>
          `;

          item.addEventListener("click", () => {
            openPipeDocModal({
              title: `SourceUnit: ${u.id}`,
              doc_id: u.doc_id || u.id,
              page_no: u.page_no || 1,
              rrf_score: `Offset: ${u.character_offset || 0}`,
              context_hash: u.context_hash || "N/A",
              text: u.text || "",
            }, idx + 1);
          });

          unitsContainer.appendChild(item);
        });
      }

      // Raw JSON Payload for Agent 3
      const rawJsonPre = $("#res-agent2-raw-json");
      if (rawJsonPre) {
        rawJsonPre.textContent = JSON.stringify(res.agent2_state_payload || {}, null, 2);
      }

      // Copy payload button
      const copyPayloadBtn = $("#btn-copy-agent2-payload");
      if (copyPayloadBtn) {
        copyPayloadBtn.onclick = async () => {
          try {
            await navigator.clipboard.writeText(JSON.stringify(res.agent2_state_payload || {}, null, 2));
            toast("Đã sao chép gói State chuyển giao cho Agent 3!");
          } catch {
            toast("Không thể sao chép dữ liệu.");
          }
        };
      }

      // Show container
      $("#agent2-output-container").hidden = false;

    } catch (err) {
      $("#agent2-loading-state").hidden = true;
      alert("Lỗi kết nối khi gọi Agent 2: " + err.message);
    } finally {
      if (btn) btn.disabled = false;
    }
  });
}
