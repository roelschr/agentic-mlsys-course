import {escapeHTML as h, inline, markdown} from "/markdown.js";

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
let course;
let overview;
let chapter;
let weekData;
let stage = "read";
let navigating = false;
let progressQueue = Promise.resolve();
let progressBlocked = false;
let noteText = "";
let savedNote = "";
let noteRevision = "";
let noteConflict = false;
let noteTimeout;
let noteFlight = null;
let browserDraft = null;
let resourceInReader = null;
let toastTimeout;

async function api(path, body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? {} : {
      method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
    });
  } catch {
    throw new Error("The local server is unavailable. Keep this tab open, restart the server, and try saving again.");
  }
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function showError(error) {
  $("#error-banner > span").textContent = error.message || String(error);
  $("#error-banner").hidden = false;
}

function run(action) {
  Promise.resolve().then(action).catch(showError);
}

function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => { $("#toast").hidden = true; }, 4200);
}

const pad = value => String(value).padStart(2, "0");
const gatePassed = progress => progress.evidence.status === "passed" && progress.explained;
const progressLabel = progress => gatePassed(progress) ? "Gate recorded complete" : progress.evidence.status === "passed" ? "Explanation still to record" : progress.evidence.status === "failed" ? "Verification needs work" : progress.attempted ? "Implementation attempted" : progress.readings.length ? "Reading in progress" : "Ready to begin";
const phase = week => week < 5 ? "THE FOUNDATIONS" : week < 9 ? "THE SYSTEMS" : "THE CAPSTONE";
const timeLabel = minutes => minutes >= 60 ? `${Math.floor(minutes / 60)}h${minutes % 60 ? ` ${minutes % 60}m` : ""}` : `${minutes}m`;

function route() {
  const params = new URLSearchParams(location.hash.slice(1));
  const requested = Number(params.get("week"));
  return {
    week: Number.isInteger(requested) && requested >= 1 && requested <= 10 ? requested : overview.current_week,
    stage: ["read", "build", "review"].includes(params.get("stage")) ? params.get("stage") : "read",
  };
}

function updateURL(replace = false) {
  const url = `#week=${chapter.number}&stage=${stage}`;
  if (location.hash !== url) history[replace ? "replaceState" : "pushState"](null, "", url);
}

function nextResource() {
  const active = chapter.resources.find(item => item.id === weekData.progress.active_resource && !weekData.progress.readings.includes(item.id));
  return active || chapter.resources.find(item => !weekData.progress.readings.includes(item.id));
}

function renderNavigation() {
  let previousPhase;
  $("#week-nav").innerHTML = course.weeks.map(week => {
    const group = phase(week.number);
    const groupLabel = group !== previousPhase ? `<div class="journey-phase">${week.number === 1 ? "I" : week.number === 5 ? "II" : "III"}. ${group}</div>` : "";
    previousPhase = group;
    const progress = overview.weeks[String(week.number)];
    return `${groupLabel}<button class="week-link ${week.number === chapter.number ? "current" : ""}" data-action="week" data-week="${week.number}" ${week.number === chapter.number ? 'aria-current="step"' : ""}><span>${pad(week.number)}</span><span>${h(week.title)}</span><span class="week-dot" aria-label="${gatePassed(progress) ? "Gate recorded complete" : ""}">${gatePassed(progress) ? "✓" : week.number === chapter.number ? "←" : ""}</span></button>`;
  }).join("");
  const complete = Object.values(overview.weeks).filter(gatePassed).length;
  $("#gate-count").textContent = `${complete} / 10 gates recorded complete`;
  $("#curriculum-grid").innerHTML = course.weeks.map(week => `<button class="curriculum-card" data-action="week" data-week="${week.number}"><span>${pad(week.number)}</span><div><strong>${h(week.title)}</strong><small>${h(progressLabel(overview.weeks[String(week.number)]))} · 11h allocation</small></div></button>`).join("");
}

function commandRows() {
  return chapter.commands.map((command, i) => `<div class="command-row"><code>${h(command)}</code><button data-action="copy-command" data-command="${i}">Copy command</button></div>`).join("");
}

function renderChapter() {
  const split = chapter.title.split(" & ");
  $("#week-title").innerHTML = split.length > 1 ? `${h(split.shift())}<br><em>& ${h(split.join(" & "))}</em>` : h(chapter.title);
  document.title = `Week ${pad(chapter.number)} · ${chapter.title} · MLSys`;
  $("#chapter-number").textContent = `CHAPTER ${pad(chapter.number)} / 10`;
  $("#chapter-phase").textContent = phase(chapter.number);
  $("#art-number").textContent = `${pad(chapter.number)}—A`;
  $("#note-path").textContent = `notes/week${pad(chapter.number)}.md`;
  $(".note-file-label").textContent = `week${pad(chapter.number)}.md`;
  $("#source-revision").textContent = `SYLLABUS ${course.revision}`;
  $("#panel-read").innerHTML = `<div class="list-heading"><span>THREE HOURS. JUST THE ASSIGNED SECTIONS.</span><span id="reading-count"></span></div><div class="resource-grid">${chapter.resources.map((resource, i) => `<article class="paper-card" data-card="${resource.id}"><div class="paper-top"><span class="paper-number">${pad(i + 1)} / ${resource.pdf ? "PAPER" : resource.url ? "ARTICLE" : "REVISIT"}<span class="paper-next" data-next="${resource.id}"></span></span><label class="check-circle"><input type="checkbox" data-reading="${resource.id}" aria-label="Mark ${h(resource.title)} reading complete"><span></span></label></div><h3>${h(resource.title)}</h3><div class="paper-detail">${inline(resource.detail)}</div><div class="paper-assignment"><span>${resource.pdf ? "Paper + your notes" : "Source + your notes"}</span><span>${resource.minutes} min</span></div><button data-action="resource" data-resource="${resource.id}">${resource.pdf ? "Read side by side" : "Open assignment"}<span aria-hidden="true">↗</span></button></article>`).join("")}</div><p class="reading-footnote">Read the assigned sections, not every page. Reading checkboxes don’t log time or mark verification gates complete.</p><details class="foldout"><summary>The concepts to keep in mind</summary><div class="prose">${markdown(chapter.concepts)}</div></details>`;
  $("#panel-build").innerHTML = `<h3 class="implementation-heading">${chapter.number >= 9 ? "Bring the pieces together." : "Small components. Deep understanding."}</h3><div class="prose">${markdown(chapter.implementation)}</div><div class="source-actions"><a class="button secondary" href="/source/${h(chapter.exercise)}" target="_blank" rel="noopener noreferrer">Exercise contract ↗</a><a class="button secondary" href="/source/${h(chapter.tests)}" target="_blank" rel="noopener noreferrer">Test contracts ↗</a></div>${chapter.number >= 9 ? `<details class="foldout"><summary>The shared capstone scope</summary><div class="prose">${markdown(course.capstone)}</div></details>` : ""}<details class="foldout"><summary>Revisit the concepts and shapes</summary><div class="prose">${markdown(chapter.concepts)}</div></details><label class="completion-check"><input type="checkbox" data-progress="attempted"><span>I’ve attempted this week’s implementation.<small>This records an attempt; verification is recorded separately.</small></span></label><h3 class="implementation-heading">Verify in your own terminal.</h3>${commandRows()}<div class="prose">${markdown(chapter.verification)}</div>`;
  $("#panel-review").innerHTML = `<details class="foldout"><summary>How to use the two-hour review</summary><div class="prose">${markdown(course.review_format)}</div></details><h3 class="implementation-heading">Close the references. Explain it aloud.</h3><div class="prose">${markdown(chapter.review)}</div><div class="checkpoint prose"><h3>This week’s checkpoint</h3>${markdown(chapter.checkpoint)}</div>${commandRows()}<form class="evidence-form" id="evidence-form"><h3>Leave evidence, not just a checkmark.</h3><p>Manual record of a run in your terminal. This site does not run tests or infer that they passed. Include the environment, outcomes, numerical discrepancies, and memory observations where relevant.</p><label>Observed outcome<select name="status"><option value="not_run">Not yet verified</option><option value="failed">Needs work / tests failed</option><option value="passed">Required tests passed</option></select></label><label>Exact command<input name="command" maxlength="2000" required></label><label>Observed results<textarea name="summary" maxlength="20000" placeholder="Command output, environment, error bounds, byte ledger, first violated invariant…"></textarea></label><p class="form-error" id="evidence-error" role="alert" hidden></p><div class="evidence-footer"><button class="button primary" type="submit">Record outcome</button><small id="evidence-timestamp"></small></div></form><label class="completion-check"><input type="checkbox" data-progress="explained"><span>I can explain this checkpoint independently.<small>A gate is recorded complete only with passing test evidence and this explanation.</small></span></label>`;
  const evidence = weekData.progress.evidence;
  const form = $("#evidence-form");
  form.elements.status.value = evidence.status;
  form.elements.command.value = evidence.command || chapter.commands.join(" && ");
  form.elements.summary.value = evidence.summary;
  $("#course-contract").innerHTML = markdown(course.pacing);
  renderIndicators();
  setStage(stage, false);
}

function renderIndicators() {
  const progress = weekData.progress;
  overview.weeks[String(chapter.number)] = progress;
  renderNavigation();
  const next = nextResource();
  const count = chapter.resources.filter(item => progress.readings.includes(item.id)).length;
  $("#reading-count").textContent = `${count} / ${chapter.resources.length} read`;
  $$('[data-reading]').forEach(input => { input.checked = progress.readings.includes(input.dataset.reading); });
  $$('[data-progress]').forEach(input => { input.checked = progress[input.dataset.progress]; });
  $$('[data-card]').forEach(card => card.classList.toggle("next", card.dataset.card === next?.id));
  $$('[data-next]').forEach(label => { label.textContent = label.dataset.next === next?.id ? "UP NEXT" : ""; });
  if (resourceInReader) $("#reader-complete").checked = progress.readings.includes(resourceInReader.id);
  $("#continue-button").innerHTML = next ? `Open the reading room <span aria-hidden="true">↗</span>` : `Continue to ${progress.attempted ? "reflection" : "implementation"} <span aria-hidden="true">↗</span>`;
  $("#next-assignment").textContent = next ? `UP NEXT · ${next.title} · ${next.minutes} min allocation` : "READINGS COMPLETE · Keep building, verifying, and explaining.";
  $("#week-status").textContent = progressLabel(progress);
  $("#week-status").classList.toggle("complete", gatePassed(progress));
  $("#budget-items").innerHTML = Object.entries(chapter.budget).map(([key, budget], i) => `<div class="budget-item"><div><span class="ordinal">${pad(i + 1)}</span><strong>${["Read", "Implement", "Reflect"][i]}</strong><span class="allocation">${budget / 60}h</span></div><progress max="${budget}" value="${Math.min(progress.minutes[key], budget)}" aria-label="${key}: ${progress.minutes[key]} minutes logged, ${budget} allocated"></progress><small>${timeLabel(progress.minutes[key])} logged / ${budget / 60}h allocated</small></div>`).join("");
  $("#budget-limit").hidden = Object.values(progress.minutes).reduce((a, b) => a + b, 0) < 660;
  $("#evidence-timestamp").textContent = progress.evidence.recorded_at ? `Recorded ${new Date(progress.evidence.recorded_at).toLocaleString()}` : "No recorded run yet";
}

function updateProgress(change) {
  const week = chapter.number;
  const pending = progressQueue.then(async () => {
    if (progressBlocked) throw new Error("Reload progress before making another change; your current file has been preserved.");
    const progress = structuredClone(weekData.progress);
    change(progress);
    try {
      const result = await api(`/api/weeks/${week}/progress`, {revision: weekData.progress_revision, progress});
      weekData = {...weekData, ...result};
      renderIndicators();
    } catch (error) {
      if (error.status === 409) progressBlocked = true;
      renderIndicators();
      throw error;
    }
  });
  progressQueue = pending.catch(() => {});
  return pending;
}

function noteStatus(text, unsaved = false) {
  $$('[data-note-status]').forEach(el => { el.textContent = text; el.classList.toggle("unsaved", unsaved); });
}

const draftKey = () => `mlsys-field-guide:week${chapter.number}:draft`;
function keepDraft(text) {
  try {
    if (text === savedNote) localStorage.removeItem(draftKey());
    else localStorage.setItem(draftKey(), text);
  } catch { /* The on-disk save is primary; localStorage is only crash recovery. */ }
}

function editNote(text, source = null) {
  noteText = text;
  $$('[data-note]').forEach(input => { if (input !== source) input.value = text; });
  keepDraft(text);
  noteStatus(text === savedNote ? "Saved to your local Markdown file" : noteConflict ? "Conflict · draft kept here; load disk version to resolve" : "Unsaved changes · saving shortly…", text !== savedNote);
  clearTimeout(noteTimeout);
  if (!noteConflict) noteTimeout = setTimeout(() => { saveNote().catch(() => {}); }, 650);
}

async function saveNote() {
  clearTimeout(noteTimeout);
  if (noteFlight) return noteFlight;
  if (noteText === savedNote) return;
  if (noteConflict) throw new Error("Your note changed on disk. Export this draft or load the disk version to resolve the conflict.");
  noteFlight = (async () => {
    try {
      while (noteText !== savedNote) {
        const snapshot = noteText;
        noteStatus("Saving to your local Markdown file…", true);
        const result = await api(`/api/weeks/${chapter.number}/note`, {revision: noteRevision, text: snapshot});
        noteRevision = result.revision;
        savedNote = snapshot;
        keepDraft(noteText);
      }
      noteStatus("Saved to your local Markdown file");
    } catch (error) {
      noteConflict = error.status === 409;
      $$('[data-note-conflict]').forEach(el => { el.hidden = !noteConflict; });
      noteStatus(noteConflict ? "File changed · your draft has been kept" : "Not saved · server unavailable or file could not be written", true);
      showError(error);
      throw error;
    } finally {
      noteFlight = null;
    }
  })();
  return noteFlight;
}

function loadNote(data) {
  noteText = savedNote = data.text;
  noteRevision = data.revision;
  noteConflict = false;
  $$('[data-note]').forEach(input => { input.value = noteText; });
  $$('[data-note-conflict]').forEach(el => { el.hidden = true; });
  browserDraft = null;
  try { browserDraft = localStorage.getItem(draftKey()); } catch { /* Optional recovery only. */ }
  $("#draft-notice").hidden = browserDraft === null || browserDraft === noteText;
  noteStatus("Saved locally · Markdown autosave is on");
}

async function navigate(week, requestedStage = "read", historyMode = "push") {
  if (navigating) return;
  navigating = true;
  $("main").setAttribute("aria-busy", "true");
  $("main").inert = true;
  try {
    await saveNote();
    await progressQueue;
    const data = await api(`/api/weeks/${week}`);
    await api("/api/current", {week});
    $("#curriculum-dialog").close();
    $("#reader-dialog").close();
    chapter = course.weeks[week - 1];
    weekData = data;
    overview.current_week = week;
    stage = requestedStage;
    progressBlocked = false;
    resourceInReader = null;
    loadNote(data.note);
    renderChapter();
    updateURL(historyMode !== "push");
    $("main").hidden = false;
    $("#loading").hidden = true;
    window.scrollTo({top: 0, behavior: "instant"});
  } catch (error) {
    if (chapter) updateURL(true);
    throw error;
  } finally {
    navigating = false;
    $("main").removeAttribute("aria-busy");
    $("main").inert = false;
  }
}

function setStage(value, history = true) {
  stage = value;
  $$('[data-tab]').forEach(tab => {
    const selected = tab.dataset.tab === stage;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  });
  if (history) updateURL();
}

function scrollToElement(selector) {
  $(selector).scrollIntoView({behavior: "smooth", block: "start"});
}

async function openResource(id) {
  const resource = chapter.resources.find(item => item.id === id);
  if (!resource) return;
  await updateProgress(progress => { progress.active_resource = id; });
  resourceInReader = resource;
  $("#reader-title").textContent = resource.title;
  $("#reader-kicker").textContent = `WEEK ${pad(chapter.number)} · ${resource.minutes} MIN READING ALLOCATION`;
  $("#reader-assignment").innerHTML = markdown(resource.assignment);
  const original = $("#reader-original");
  original.hidden = !resource.url;
  if (resource.url) original.href = resource.url;
  else original.removeAttribute("href");
  const frame = $("#paper-frame");
  frame.hidden = !resource.pdf;
  if (resource.pdf) frame.src = `${resource.pdf}#view=FitH`;
  else frame.removeAttribute("src");
  $("#reader-fallback").hidden = Boolean(resource.pdf);
  $(".pdf-help").hidden = !resource.pdf;
  $("#reader-revisit").innerHTML = !resource.url ? `<p>Use the earlier chapters’ readings and your own notes for this assignment.</p><a href="/source/SYLLABUS.md" target="_blank" rel="noopener noreferrer">Open source syllabus ↗</a>` : "";
  $("#reader-complete").checked = weekData.progress.readings.includes(id);
  $("#reader-dialog").showModal();
}

function readingChanged(id, checked) {
  return updateProgress(progress => {
    progress.readings = checked ? [...new Set([...progress.readings, id])] : progress.readings.filter(value => value !== id);
  });
}

function exportNote() {
  const url = URL.createObjectURL(new Blob([noteText], {type: "text/markdown;charset=utf-8"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = `week${pad(chapter.number)}.md`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function copyCommand(index, button) {
  try {
    await navigator.clipboard.writeText(chapter.commands[index]);
    toast("Command copied. Run it from the repository root in your terminal.");
  } catch {
    const selection = getSelection();
    const range = document.createRange();
    range.selectNodeContents(button.parentElement.querySelector("code"));
    selection.removeAllRanges();
    selection.addRange(range);
    toast("The command is selected. Press Ctrl+C to copy.");
  }
}

function showTimeLog() {
  for (const [key, value] of Object.entries(weekData.progress.minutes)) $("#time-form").elements[key].value = value;
  $("#time-error").textContent = "";
  $("#time-dialog").showModal();
}

// An optional countdown, independent of logged time and paused on tab hiding.
let seconds = 1500;
let deadline = 0;
let timer = null;
function drawTimer() {
  const label = `${pad(Math.floor(seconds / 60))}:${pad(seconds % 60)}`;
  $("#timer-clock").textContent = $("#timer-mini").textContent = label;
  $("#timer-toggle").textContent = timer ? "Pause" : seconds < 1500 && seconds > 0 ? "Resume" : "Start";
}
function pauseTimer() {
  if (timer) seconds = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
  clearInterval(timer);
  timer = null;
  drawTimer();
}
function toggleTimer() {
  if (timer) { pauseTimer(); return; }
  if (seconds <= 0) seconds = 1500;
  deadline = Date.now() + seconds * 1000;
  timer = setInterval(() => {
    seconds = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
    if (!seconds) pauseTimer();
    drawTimer();
  }, 250);
  drawTimer();
}
document.addEventListener("visibilitychange", () => { if (document.hidden) pauseTimer(); });
$("#reader-dialog").addEventListener("close", () => {
  pauseTimer();
  $("#paper-frame").removeAttribute("src");
  resourceInReader = null;
  saveNote().catch(() => {});
});

document.addEventListener("click", event => {
  const close = event.target.closest("[data-close]");
  if (close) document.getElementById(close.dataset.close).close();
  const tab = event.target.closest("[data-tab]");
  if (tab && chapter && !navigating) setStage(tab.dataset.tab);
  const button = event.target.closest("[data-action]");
  if (!button) return;
  run(async () => {
    const action = button.dataset.action;
    if (action === "dismiss-error") { $("#error-banner").hidden = true; return; }
    if (navigating) return;
    if (!chapter) {
      if (action === "reload-progress") location.reload();
      return;
    }
    switch (action) {
      case "week": await navigate(Number(button.dataset.week)); break;
      case "study": window.scrollTo({top: 0, behavior: "smooth"}); break;
      case "curriculum": $("#curriculum-dialog").showModal(); break;
      case "reading-room": setStage("read"); scrollToElement("#itinerary"); break;
      case "notebook": scrollToElement("#notebook"); $("#main-note").focus({preventScroll: true}); break;
      case "course-contract": $("#contract-dialog").showModal(); break;
      case "resource": await openResource(button.dataset.resource); break;
      case "continue": {
        const next = nextResource();
        if (next) await openResource(next.id);
        else { setStage(weekData.progress.attempted ? "review" : "build"); scrollToElement("#itinerary"); }
        break;
      }
      case "close-reader": $("#reader-dialog").close(); break;
      case "copy-command": await copyCommand(Number(button.dataset.command), button); break;
      case "save-note": await saveNote(); toast("Note saved locally."); break;
      case "export-note": exportNote(); break;
      case "insert-template": editNote(`${noteText}${noteText ? "\n\n" : ""}## Prediction\n\n\n## Test evidence\n\n- Command: \n- Environment / revision: \n- Outcomes / first violated invariant: \n- Logical bytes / observed allocation: \n\n## Explanation\n\n\n## Next experiment / blocker\n\n`); scrollToElement("#notebook"); break;
      case "restore-draft": editNote(browserDraft); $("#draft-notice").hidden = true; break;
      case "discard-draft": keepDraft(savedNote); browserDraft = null; $("#draft-notice").hidden = true; break;
      case "reload-note": {
        clearTimeout(noteTimeout);
        if (noteFlight) await noteFlight.catch(() => {});
        const data = await api(`/api/weeks/${chapter.number}`);
        keepDraft(noteText);
        loadNote(data.note);
        $("#error-banner").hidden = true;
        break;
      }
      case "reload-progress": {
        await progressQueue;
        const data = await api(`/api/weeks/${chapter.number}`);
        weekData = {...weekData, progress: data.progress, progress_revision: data.progress_revision};
        progressBlocked = false;
        renderChapter();
        $("#error-banner").hidden = true;
        toast("Progress reloaded. Your note draft is unchanged.");
        break;
      }
      case "log-time": showTimeLog(); break;
      case "timer-toggle": toggleTimer(); break;
      case "timer-reset": pauseTimer(); seconds = 1500; drawTimer(); break;
    }
  });
});

document.addEventListener("change", event => {
  if (navigating) return;
  const input = event.target;
  if (input.matches("[data-reading]")) run(() => readingChanged(input.dataset.reading, input.checked));
  if (input.matches("[data-progress]")) {
    const checked = input.checked;
    run(() => updateProgress(progress => { progress[input.dataset.progress] = checked; }));
  }
  if (input.id === "reader-complete" && resourceInReader) {
    const id = resourceInReader.id;
    const checked = input.checked;
    run(() => readingChanged(id, checked));
  }
});

$$('[data-note]').forEach(input => input.addEventListener("input", () => editNote(input.value, input)));

document.addEventListener("submit", event => {
  if (navigating) { event.preventDefault(); return; }
  if (event.target.id === "time-form") {
    event.preventDefault();
    const form = event.target;
    const minutes = Object.fromEntries(new FormData(form).entries());
    for (const key of Object.keys(minutes)) minutes[key] = Number(minutes[key]);
    const submit = form.querySelector("[type=submit]");
    submit.disabled = true;
    updateProgress(progress => { progress.minutes = minutes; }).then(() => {
      $("#time-dialog").close();
      toast("Time log saved. You can adjust these totals whenever needed.");
    }).catch(error => { $("#time-error").textContent = error.message; }).finally(() => { submit.disabled = false; });
  }
  if (event.target.id === "evidence-form") {
    event.preventDefault();
    const form = event.target;
    const data = Object.fromEntries(new FormData(form).entries());
    const submit = form.querySelector("[type=submit]");
    submit.disabled = true;
    $("#evidence-error").hidden = true;
    updateProgress(progress => { progress.evidence = {...data, recorded_at: progress.evidence.recorded_at}; }).then(() => {
      toast("Your reported outcome has been saved with a timestamp.");
    }).catch(error => {
      $("#evidence-error").textContent = error.message;
      $("#evidence-error").hidden = false;
    }).finally(() => { submit.disabled = false; });
  }
});

const tabs = $$('[data-tab]');
tabs.forEach((tab, index) => tab.addEventListener("keydown", event => {
  let next;
  if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
  if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = tabs.length - 1;
  if (next === undefined) return;
  event.preventDefault();
  setStage(tabs[next].dataset.tab);
  tabs[next].focus();
}));

window.addEventListener("beforeunload", event => {
  if (noteText !== savedNote) { event.preventDefault(); event.returnValue = ""; }
});
window.addEventListener("popstate", () => run(async () => {
  if (!overview) return;
  const requested = route();
  if (requested.week === chapter.number) setStage(requested.stage, false);
  else await navigate(requested.week, requested.stage, "replace");
}));

run(async () => {
  [course, overview] = await Promise.all([api("/api/course"), api("/api/state")]);
  const requested = route();
  await navigate(requested.week, requested.stage, "replace");
});
