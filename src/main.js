import { FilesetResolver, PoseLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";
import { PickleballMotionClassifier, } from "./motionClassifier";
const statusEl = document.getElementById("status");
const motionEl = document.getElementById("motion");
const webcamBtn = document.getElementById("webcamBtn");
const videoFileInput = document.getElementById("videoFile");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const classifier = new PickleballMotionClassifier();
let poseLandmarker = null;
let drawingUtils;
let usingWebcam = false;
let running = false;
function setStatus(text) {
    statusEl.textContent = text;
}
function setMotion(text) {
    motionEl.textContent = text;
}
async function initPoseLandmarker() {
    setStatus("Loading MediaPipe model...");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
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
function toPosePoint(lm) {
    return {
        x: lm.x,
        y: lm.y,
        z: lm.z,
        visibility: lm.visibility ?? 1,
    };
}
function collectFeatures(frame) {
    return classifier.extractFeatures(frame);
}
function drawPose(landmarks) {
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
        color: "#56b6ff",
        lineWidth: 2,
    });
    drawingUtils.drawLandmarks(landmarks, {
        color: "#7dffcf",
        lineWidth: 1,
        radius: 2,
    });
}
async function detectLoop() {
    if (!poseLandmarker || !running)
        return;
    if (video.readyState >= 2) {
        resizeCanvasToVideo();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const now = performance.now();
        const result = poseLandmarker.detectForVideo(video, now);
        if (result.landmarks && result.landmarks.length > 0) {
            const lms = result.landmarks[0].map(toPosePoint);
            const frame = { t: now, landmarks: lms };
            const features = collectFeatures(frame);
            const motion = classifier.classify(features);
            setMotion(motion.label);
            if (motion.isNew && motion.label !== "Unknown") {
                console.log(`${motion.label} detected`, motion.debug);
            }
            drawPose(lms);
        }
        else {
            setMotion("No pose");
        }
    }
    requestAnimationFrame(detectLoop);
}
async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    usingWebcam = true;
    running = true;
    setStatus("Running (webcam)");
    detectLoop();
}
async function startVideoFile(file) {
    const url = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = url;
    await video.play();
    usingWebcam = false;
    running = true;
    setStatus(`Running (file: ${file.name})`);
    detectLoop();
}
webcamBtn.addEventListener("click", async () => {
    try {
        if (!poseLandmarker)
            await initPoseLandmarker();
        await startWebcam();
    }
    catch (err) {
        console.error(err);
        setStatus("Cannot start webcam");
    }
});
videoFileInput.addEventListener("change", async (e) => {
    try {
        if (!poseLandmarker)
            await initPoseLandmarker();
        const input = e.target;
        const file = input.files?.[0];
        if (!file)
            return;
        await startVideoFile(file);
    }
    catch (err) {
        console.error(err);
        setStatus("Cannot load video file");
    }
});
// Lazy init; user can still click to force start
initPoseLandmarker().catch((e) => {
    console.error(e);
    setStatus("Init error");
});
