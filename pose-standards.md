# Tiêu chuẩn góc độ Pickleball tham chiếu

Dùng để so sánh form người chơi, đưa ra feedback kỹ thuật.

## 1. Tư thế chuẩn bị (Ready Position)
- **Góc gập gối:** 120° - 140°
- **Góc khuỷu tay:** 90° - 120°
- **Góc mặt vợt:** ~45° hướng trước

## 2. Cú giao bóng (Serve)
- **Xoay vai & hông:** 45° - 90° (backswing) → 0° (tiếp xúc)
- **Góc khuỷu tay khi chạm bóng:** 160° - 170°
- **Góc mặt vợt:** 10° - 20° mở ngửa

## 3. Cú Dink
- **Góc gập gối:** 100° - 110°
- **Góc gập lưng:** không quá 30° (giữ lưng thẳng)
- **Góc cổ tay:** khóa 0°, không vẩy
- **Góc mặt vợt:** 45° - 60° mở ngửa

## 4. Groundstroke (Forehand/Backhand)
- **Xoay vai:** 90° (vuông góc lưới) khi chuẩn bị
- **Wrist Lag (forehand):** 30° - 45° bẻ gập ra sau
- **Góc khuỷu tay khi tiếp xúc:** 150° - 160°

## 5. Cú Volley
- **Backswing:** 0° - 15° (gần như không vung ra sau)
- **Góc khuỷu tay:** 90° - 110°, nằm trước cơ thể
- **Góc mặt vợt (block):** 0° - 5°; (slice): 10° - 20°

## Phương pháp so sánh
- Mỗi động tác: trích các góc từ MediaPipe Pose landmarks
- So với khoảng chuẩn trên → tính % khớp / sai lệch
- Output: "Gối quá cao, cần hạ thấp thêm", "Cổ tay bị vẩy", "Góc khuỷu tay tốt", v.v.
- Có thể code feedback theo 3 mức: ✅ Tốt / ⚠️ Cần điều chỉnh / ❌ Sai
