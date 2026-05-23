import { FilesetResolver, PoseLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";
import {
  PickleballMotionClassifier,
  type MotionFeatures,
  type PosePoint,
  type PoseFrame,
} from "./motionClassifier";

const statusEl = document.getElementById("status") as HTMLSpanElement;
const motionEl = document.getElementById("motion") as HTMLSpanElement;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const stopBtn = document.getElementById("stopBtn") as HTMLButtonElement;
const recordStateEl = document.getElementById("recordState") as HTMLDivElement;
const resultEl = document.getElementById("result") as HTMLDivElement;

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;

const classifier = new PickleballMotionClassifier();
let poseLandmarker: PoseLandmarker | null = null;
let drawingUtils: DrawingUtils;

let webcamStream: MediaStream | null = null;
let isRecording = false;
let rafId = 0;

// Buffer frames during recording for post-analysis
let frameBuffer: PoseFrame[] = [];

function setStatus(text: string) {
  statusEl.textContent = text;
}

function setMotion(text: string) {
  motionEl.textContent = text;
}

function setResult(main: string, secondary: string[] = []) {
  const extra = secondary.length
    ? `<small>Motion phụ: ${secondary.join(", ")}</small>`
    : "<small>Motion phụ: (không có)</small>";
  resultEl.innerHTML = `Kết quả: <strong>${main}</strong>${extra}`;
}

function resetUIForIdle() {
  recordStateEl.classList.add("hidden");
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

async function initPoseLandmarker() {
  setStatus("Loading MediaPipe model...");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  drawingUtils = new DrawingUtils(ctx);
  setStatus("Ready");
}

function resizeCanvasToVideo() {
  const w = video.videoWidth || 1280;
  const h = video.videoHeight || 720;
  canvas.width = w;
  canvas.height = h;
}

function toPosePoint(lm: { x: number; y: number; z: number; visibility?: number }): PosePoint {
  return {
    x: lm.x,
    y: lm.y,
    z: lm.z,
    visibility: lm.visibility ?? 1,
  };
}

function collectFeatures(frame: PoseFrame): MotionFeatures {
  return classifier.extractFeatures(frame);
}

function drawPose(landmarks: PosePoint[]) {
  drawingUtils.drawConnectors(landmarks as any, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#56b6ff",
    lineWidth: 2,
  });
  drawingUtils.drawLandmarks(landmarks as any, {
    color: "#7dffcf",
    lineWidth: 1,
    radius: 2,
  });
}

async function ensureWebcamStream() {
  if (webcamStream) return webcamStream;
  webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  return webcamStream;
}

/** Capture 1 frame from webcam → extract landmarks → buffer for later analysis */
function captureFrame(now: number) {
  if (!poseLandmarker) return;

  resizeCanvasToVideo();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const result = poseLandmarker.detectForVideo(video, now);

  if (result.landmarks && result.landmarks.length > 0) {
    const lms = result.landmarks[0].map(toPosePoint);
    const frame: PoseFrame = { t: now, landmarks: lms };
    frameBuffer.push(frame);

    // Draw skeleton for live preview
    drawPose(lms);
  }
}

/** Recording loop: capture frames at ~15fps while recording */
function recordingLoop() {
  if (!isRecording) return;

  const now = performance.now();
  captureFrame(now);
  rafId = requestAnimationFrame(recordingLoop);
}

async function startRecording() {
  try {
    if (!poseLandmarker) await initPoseLandmarker();

    const stream = await ensureWebcamStream();
    video.srcObject = stream;
    await video.play();

    // Reset buffers
    frameBuffer = [];
    isRecording = true;
    setMotion("Recording");
    setResult("Đang ghi...", []);
    recordStateEl.classList.remove("hidden");

    startBtn.disabled = true;
    stopBtn.disabled = false;

    setStatus("Recording...");
    rafId = requestAnimationFrame(recordingLoop);
  } catch (err) {
    console.error(err);
    setStatus("Cannot start recording");
    resetUIForIdle();
  }
}

function stopRecording() {
  if (!isRecording) return;

  isRecording = false;
  stopBtn.disabled = true;
  recordStateEl.classList.add("hidden");

  // Stop the capture loop
  cancelAnimationFrame(rafId);

  setStatus("Analyzing...");

  // Run analysis on the buffered frames
  setTimeout(() => {
    analyzeFrames();
    setStatus("Ready");
    resetUIForIdle();
  }, 50);
}

function analyzeFrames() {
  if (!frameBuffer.length) {
    setMotion("Không có dữ liệu");
    setResult("Không phát hiện động tác", []);
    return;
  }

  // Classify every collected frame
  const labels: string[] = [];
  for (const frame of frameBuffer) {
    const features = collectFeatures(frame);
    const motion = classifier.classify(features);
    if (motion.label !== "Unknown" && motion.label !== "No pose") {
      labels.push(motion.label);
    }
  }

  if (!labels.length) {
    setMotion("Không rõ");
    setResult("Không phát hiện rõ động tác", []);
    return;
  }

  const { topLabel, secondary } = summarizeLabels(labels);
  setMotion(topLabel);
  setResult(topLabel, secondary);
  console.log(`Result: ${topLabel}`, `${frameBuffer.length} frames, ${labels.length} labeled`);
}

function summarizeLabels(labels: string[]) {
  const counts = new Map<string, number>();
  for (const label of labels) {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const topLabel = sorted[0][0];
  const secondary = sorted.slice(1, 4).map(([name, c]) => `${name} (${c})`);

  return { topLabel, secondary };
}

startBtn.addEventListener("click", () => {
  startRecording();
});

stopBtn.addEventListener("click", () => {
  stopRecording();
});

initPoseLandmarker().catch((e) => {
  console.error(e);
  setStatus("Init error");
});
