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

let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let webcamStream: MediaStream | null = null;
let isRecording = false;

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
  resultEl.innerHTML = `Kết quả: ${main}${extra}`;
}

function resetUIForIdle() {
  recordStateEl.classList.add("hidden");
  startBtn.disabled = false;
  stopBtn.disabled = true;
  video.classList.remove("hidden");
  canvas.classList.remove("hidden");
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

function resizeCanvasToVideo(target: HTMLVideoElement) {
  const w = target.videoWidth || 1280;
  const h = target.videoHeight || 720;
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

// Keep drawing logic from previous implementation for debug/analysis view.
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

function chooseSupportedMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const t of candidates) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "video/webm";
}

async function startRecording() {
  try {
    if (!poseLandmarker) await initPoseLandmarker();

    const stream = await ensureWebcamStream();
    video.srcObject = stream;
    await video.play();

    recordedChunks = [];
    const mimeType = chooseSupportedMimeType();
    mediaRecorder = new MediaRecorder(stream, { mimeType });

    mediaRecorder.ondataavailable = (evt: BlobEvent) => {
      if (evt.data && evt.data.size > 0) recordedChunks.push(evt.data);
    };

    mediaRecorder.onstop = async () => {
      setStatus("Analyzing recorded video...");
      const blob = new Blob(recordedChunks, { type: mimeType });
      await analyzeRecordedBlob(blob);
      setStatus("Ready for next recording");
      resetUIForIdle();
    };

    isRecording = true;
    setMotion("Recording");
    setResult("Đang ghi...", []);

    // Hide preview while recording, show only recording indicator.
    video.classList.add("hidden");
    canvas.classList.add("hidden");
    recordStateEl.classList.remove("hidden");

    startBtn.disabled = true;
    stopBtn.disabled = false;

    mediaRecorder.start(120);
    setStatus("Recording...");
  } catch (err) {
    console.error(err);
    setStatus("Cannot start recording");
    resetUIForIdle();
  }
}

function stopRecording() {
  if (!mediaRecorder || !isRecording) return;
  isRecording = false;
  stopBtn.disabled = true;
  recordStateEl.classList.add("hidden");
  setStatus("Stopping recorder...");

  if (mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

async function analyzeRecordedBlob(blob: Blob) {
  if (!poseLandmarker) return;

  const tempVideo = document.createElement("video");
  tempVideo.muted = true;
  tempVideo.playsInline = true;
  tempVideo.preload = "auto";
  const blobUrl = URL.createObjectURL(blob);
  tempVideo.src = blobUrl;

  await new Promise<void>((resolve, reject) => {
    tempVideo.onloadedmetadata = () => resolve();
    tempVideo.onerror = () => reject(new Error("Cannot read recorded blob"));
  });

  await tempVideo.play();
  tempVideo.pause();

  resizeCanvasToVideo(tempVideo);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const labels: string[] = [];

  // Analyze frame-by-frame by seeking in small steps.
  const fps = 15;
  const step = 1 / fps;
  const total = tempVideo.duration;

  for (let t = 0; t < total; t += step) {
    await seekTo(tempVideo, t);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tempVideo, 0, 0, canvas.width, canvas.height);

    const tsMs = t * 1000;
    const result = poseLandmarker.detectForVideo(tempVideo, tsMs);

    if (result.landmarks && result.landmarks.length > 0) {
      const lms = result.landmarks[0].map(toPosePoint);
      const frame: PoseFrame = { t: tsMs, landmarks: lms };
      const features = collectFeatures(frame);
      const motion = classifier.classify(features);

      if (motion.label !== "Unknown" && motion.label !== "No pose") {
        labels.push(motion.label);
      }

      // Keep drawing for analysis preview after recording.
      drawPose(lms);
    }
  }

  tempVideo.pause();
  URL.revokeObjectURL(blobUrl);

  const { topLabel, secondary } = summarizeLabels(labels);
  setMotion(topLabel);
  setResult(topLabel, secondary);
}

function summarizeLabels(labels: string[]) {
  if (!labels.length) {
    return { topLabel: "Không phát hiện rõ động tác", secondary: [] as string[] };
  }

  const counts = new Map<string, number>();
  for (const label of labels) {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const topLabel = sorted[0][0];
  const secondary = sorted.slice(1, 4).map(([name, c]) => `${name} (${c})`);

  return { topLabel, secondary };
}

function seekTo(videoEl: HTMLVideoElement, time: number) {
  return new Promise<void>((resolve) => {
    const onSeeked = () => {
      videoEl.removeEventListener("seeked", onSeeked);
      resolve();
    };
    videoEl.addEventListener("seeked", onSeeked, { once: true });
    videoEl.currentTime = Math.min(time, Math.max(0, videoEl.duration - 0.001));
  });
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
