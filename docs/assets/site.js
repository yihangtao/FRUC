"use strict";

// One master clock controls the supplied streams. Hidden stages are paused to
// avoid decoding every output stream, and seek to the current clock when shown.
const master = document.querySelector("#master-video");
const videos = [...document.querySelectorAll(".sync-video")];
const timeline = document.querySelector("#timeline");
const playButton = document.querySelector("#play-toggle");
const status = document.querySelector("#video-status");
let playing = false;
let duration = 13.9;
let frameRequest;

function activeVideos() {
  return videos.filter(video => video === master || !video.closest("[hidden]"));
}
function seekAll(time) {
  for (const video of activeVideos()) {
    if (video.readyState > 0) video.currentTime = Math.min(time, Math.max(0, video.duration - 0.05));
  }
  updateTime(time);
}
function updateTime(time) {
  timeline.value = String(time);
  document.querySelector("#time-display").textContent = `${time.toFixed(1)} / ${duration.toFixed(1)}s`;
}
function tick() {
  if (!playing) return;
  const time = master.currentTime;
  updateTime(time);
  for (const video of activeVideos()) {
    if (video !== master && video.readyState >= 2 && Math.abs(video.currentTime - time) > 0.18) {
      video.currentTime = time;
    }
  }
  frameRequest = requestAnimationFrame(tick);
}
async function setPlaying(next) {
  playing = next;
  cancelAnimationFrame(frameRequest);
  playButton.textContent = playing ? "Ⅱ Pause" : "▶ Play";
  playButton.setAttribute("aria-label", playing ? "Pause synchronized videos" : "Play synchronized videos");
  if (playing) {
    if (master.currentTime >= duration - 0.1) seekAll(0);
    const results = await Promise.allSettled(activeVideos().map(video => video.play()));
    const masterResult = results[activeVideos().indexOf(master)];
    if (masterResult?.status === "rejected") {
      playing = false;
      playButton.textContent = "▶ Play";
      playButton.setAttribute("aria-label", "Play synchronized videos");
      status.textContent = "Playback could not start. Please try again.";
      videos.forEach(video => video.pause());
      return;
    }
    tick();
  } else videos.forEach(video => video.pause());
}
master.addEventListener("loadedmetadata", () => {
  duration = master.duration;
  timeline.max = String(duration);
  updateTime(master.currentTime);
});
videos.forEach(video => {
  video.addEventListener("loadedmetadata", () => {
    if (video !== master) video.currentTime = master.currentTime;
  });
  video.addEventListener("error", () => {
    status.classList.remove("sr-only");
    status.textContent = "A video could not load. Check the local media files and serve this page over HTTP.";
  });
});
master.addEventListener("ended", () => { setPlaying(false); seekAll(0); });
playButton.addEventListener("click", () => setPlaying(!playing));
timeline.addEventListener("input", () => seekAll(Number(timeline.value)));
document.querySelector("#speed").addEventListener("change", event => {
  videos.forEach(video => { video.playbackRate = Number(event.target.value); });
});

const tabs = [...document.querySelectorAll(".stage-tab")];
const stackedTabs = matchMedia("(max-width: 480px)");
function updateTabOrientation() {
  document.querySelector('[role="tablist"]').setAttribute(
    "aria-orientation", stackedTabs.matches ? "vertical" : "horizontal"
  );
}
stackedTabs.addEventListener("change", updateTabOrientation);
updateTabOrientation();
function selectTab(tab) {
  tabs.forEach(item => {
    const selected = item === tab;
    item.classList.toggle("active", selected);
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
  });
  videos.forEach(video => {
    if (video !== master && video.closest("[hidden]")) video.pause();
  });
  seekAll(master.currentTime);
  if (playing) activeVideos().forEach(video => { video.play().catch(() => {}); });
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", event => {
    let next;
    if (["ArrowDown", "ArrowRight"].includes(event.key)) next = (index + 1) % tabs.length;
    if (["ArrowUp", "ArrowLeft"].includes(event.key)) next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); selectTab(tabs[next]); tabs[next].focus(); }
  });
});

const hero = document.querySelector("#hero-video");
const heroButton = document.querySelector("#hero-toggle");
function updateHeroButton() {
  heroButton.textContent = hero.paused ? "▶" : "Ⅱ";
  heroButton.setAttribute("aria-label", hero.paused ? "Play hero video" : "Pause hero video");
}
hero.addEventListener("play", updateHeroButton);
hero.addEventListener("pause", updateHeroButton);
heroButton.addEventListener("click", () => {
  if (hero.paused) hero.play().catch(() => { updateHeroButton(); });
  else hero.pause();
});
if (matchMedia("(prefers-reduced-motion: reduce)").matches) { hero.autoplay = false; hero.pause(); }
updateHeroButton();

const dialog = document.querySelector("#figure-dialog");
document.querySelectorAll("[data-figure]").forEach(button => {
  button.addEventListener("click", () => {
    const image = document.querySelector("#dialog-image");
    image.src = button.dataset.figure;
    image.alt = button.querySelector("img").alt;
    document.querySelector("#dialog-caption").textContent = button.dataset.caption;
    dialog.showModal();
  });
});
document.querySelector("#close-figure").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });

document.querySelector("#copy-citation").addEventListener("click", async () => {
  const code = document.querySelector("#bibtex");
  const copyStatus = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(code.textContent);
    copyStatus.textContent = "Citation copied.";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(code);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = "Citation selected. Press Ctrl+C or Command+C to copy.";
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) { setPlaying(false); hero.pause(); }
});
