export type PosePoint = {
  x: number;
  y: number;
  z: number;
  visibility: number;
};

export type PoseFrame = {
  t: number;
  landmarks: PosePoint[]; // 33 points from MediaPipe Pose
};

export type MotionFeatures = {
  t: number;
  wristAboveShoulder: boolean;
  wristBelowHip: boolean;
  elbowAngleDeg: number;
  shoulderAngleDeg: number;
  trunkLeaningForward: boolean;
  wristSpeed: number;
  wristVerticalVel: number;
  wristHorizontalVel: number;
  handCrossBody: boolean;
  contactZoneHigh: boolean;
  contactZoneMid: boolean;
};

export type MotionResult = {
  label: string;
  isNew: boolean;
  debug: Record<string, number | boolean | string>;
};

const IDX = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  NOSE: 0,
};

function getPoint(lms: PosePoint[], i: number): PosePoint {
  return lms[i] ?? { x: 0, y: 0, z: 0, visibility: 0 };
}

function mid(a: PosePoint, b: PosePoint): PosePoint {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
    visibility: Math.min(a.visibility, b.visibility),
  };
}

function vec(a: PosePoint, b: PosePoint) {
  return { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
}

function length(v: { x: number; y: number; z: number }) {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

function angleDeg(a: PosePoint, b: PosePoint, c: PosePoint): number {
  const ba = vec(b, a);
  const bc = vec(b, c);
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const denom = Math.max(1e-6, length(ba) * length(bc));
  const cos = Math.min(1, Math.max(-1, dot / denom));
  return (Math.acos(cos) * 180) / Math.PI;
}

function chooseDominantArm(lms: PosePoint[]) {
  const lw = getPoint(lms, IDX.LEFT_WRIST);
  const rw = getPoint(lms, IDX.RIGHT_WRIST);
  const ls = getPoint(lms, IDX.LEFT_SHOULDER);
  const rs = getPoint(lms, IDX.RIGHT_SHOULDER);

  const lReach = Math.abs(lw.x - ls.x) + Math.abs(lw.y - ls.y);
  const rReach = Math.abs(rw.x - rs.x) + Math.abs(rw.y - rs.y);

  return lReach > rReach
    ? {
        side: "L" as const,
        shoulder: ls,
        elbow: getPoint(lms, IDX.LEFT_ELBOW),
        wrist: lw,
      }
    : {
        side: "R" as const,
        shoulder: rs,
        elbow: getPoint(lms, IDX.RIGHT_ELBOW),
        wrist: rw,
      };
}

export class PickleballMotionClassifier {
  private frameHistory: PoseFrame[] = [];
  private featureHistory: MotionFeatures[] = [];
  private currentLabel = "Unknown";
  private stableCount = 0;

  extractFeatures(frame: PoseFrame): MotionFeatures {
    const lms = frame.landmarks;
    const dom = chooseDominantArm(lms);

    const leftHip = getPoint(lms, IDX.LEFT_HIP);
    const rightHip = getPoint(lms, IDX.RIGHT_HIP);
    const hipMid = mid(leftHip, rightHip);

    const leftShoulder = getPoint(lms, IDX.LEFT_SHOULDER);
    const rightShoulder = getPoint(lms, IDX.RIGHT_SHOULDER);
    const shoulderMid = mid(leftShoulder, rightShoulder);

    const nose = getPoint(lms, IDX.NOSE);

    // Khuỷu tay: vai-khuỷu-cổ tay
    const elbowAngle = angleDeg(dom.shoulder, dom.elbow, dom.wrist);

    // Góc vai: hông-vai-khuỷu
    const shoulderAngle = angleDeg(hipMid, dom.shoulder, dom.elbow);

    const wristAboveShoulder = dom.wrist.y < dom.shoulder.y;
    const wristBelowHip = dom.wrist.y > hipMid.y;

    // Y nhỏ hơn nghĩa là cao hơn trong normalized image
    const contactZoneHigh = dom.wrist.y < shoulderMid.y;
    const contactZoneMid = dom.wrist.y >= shoulderMid.y && dom.wrist.y <= hipMid.y;

    // Trục cơ thể nghiêng tới trước (giả định camera ngang)
    const trunkLeaningForward = nose.y > shoulderMid.y - 0.09;

    const prev = this.frameHistory[this.frameHistory.length - 1];
    let wristSpeed = 0;
    let wristVerticalVel = 0;
    let wristHorizontalVel = 0;

    if (prev) {
      const prevDom = chooseDominantArm(prev.landmarks);
      const dt = Math.max(1, frame.t - prev.t);
      const dx = dom.wrist.x - prevDom.wrist.x;
      const dy = dom.wrist.y - prevDom.wrist.y;

      wristHorizontalVel = dx / dt;
      wristVerticalVel = dy / dt;
      wristSpeed = Math.sqrt(dx * dx + dy * dy) / dt;
    }

    const handCrossBody = dom.side === "R"
      ? dom.wrist.x < shoulderMid.x
      : dom.wrist.x > shoulderMid.x;

    const f: MotionFeatures = {
      t: frame.t,
      wristAboveShoulder,
      wristBelowHip,
      elbowAngleDeg: elbowAngle,
      shoulderAngleDeg: shoulderAngle,
      trunkLeaningForward,
      wristSpeed,
      wristVerticalVel,
      wristHorizontalVel,
      handCrossBody,
      contactZoneHigh,
      contactZoneMid,
    };

    this.frameHistory.push(frame);
    if (this.frameHistory.length > 24) this.frameHistory.shift();

    this.featureHistory.push(f);
    if (this.featureHistory.length > 24) this.featureHistory.shift();

    return f;
  }

  classify(f: MotionFeatures): MotionResult {
    // Threshold-based baseline (dễ hiểu và dễ tune)
    // Đơn vị velocity: normalized coords per ms, nên ngưỡng rất nhỏ.

    let label = "Unknown";

    const fastSwing = Math.abs(f.wristHorizontalVel) > 0.0009 || Math.abs(f.wristVerticalVel) > 0.0009;
    const mediumSwing = Math.abs(f.wristHorizontalVel) > 0.00045 || Math.abs(f.wristVerticalVel) > 0.00045;

    if (f.wristBelowHip && f.wristVerticalVel < -0.00065 && f.elbowAngleDeg > 130) {
      label = "Serve";
    } else if (f.contactZoneMid && mediumSwing && !f.wristAboveShoulder && f.trunkLeaningForward) {
      label = "Drive";
    } else if (f.contactZoneHigh && f.wristVerticalVel < -0.00035 && mediumSwing) {
      label = "Volley";
    } else if (f.contactZoneMid && !fastSwing && Math.abs(f.wristHorizontalVel) < 0.00035) {
      label = "Dink";
    } else if (f.wristAboveShoulder && f.wristVerticalVel < -0.00055 && f.elbowAngleDeg > 120) {
      label = "Lob";
    }

    // Forehand / Backhand ưu tiên khi có swing rõ
    if (mediumSwing) {
      if (f.handCrossBody) {
        label = "Backhand";
      } else {
        label = label === "Unknown" ? "Forehand" : label;
      }
    }

    // Smoothing để giảm nhấp nháy nhãn
    let isNew = false;
    if (label === this.currentLabel) {
      this.stableCount += 1;
    } else {
      this.currentLabel = label;
      this.stableCount = 1;
      isNew = true;
    }

    const stableLabel = this.stableCount >= 2 ? this.currentLabel : "Unknown";

    return {
      label: stableLabel,
      isNew,
      debug: {
        elbow: Number(f.elbowAngleDeg.toFixed(1)),
        shoulder: Number(f.shoulderAngleDeg.toFixed(1)),
        wVy: Number(f.wristVerticalVel.toFixed(6)),
        wVx: Number(f.wristHorizontalVel.toFixed(6)),
        speed: Number(f.wristSpeed.toFixed(6)),
        high: f.contactZoneHigh,
        mid: f.contactZoneMid,
        low: f.wristBelowHip,
        cross: f.handCrossBody,
      },
    };
  }
}
