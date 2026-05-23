# Pickleball MediaPipe (Node.js + TypeScript)

Project demo nhận diện động tác Pickleball bằng **MediaPipe Pose Landmarker** với pipeline TypeScript, chạy qua webcam hoặc video file.

## Tính năng
- Trích xuất **33 pose landmarks** mỗi frame.
- Rule-based classifier cho các động tác cơ bản:
  - Serve
  - Dink
  - Drive
  - Lob
  - Volley
  - Forehand
  - Backhand
- Hiển thị động tác hiện tại trên giao diện + log console (`"Serve detected"`, ...).
- Kiến trúc rõ ràng để tái sử dụng và nâng cấp (thêm KNN, calibrate theo người chơi thật).

## Cấu trúc
- `src/main.ts`: pipeline video/webcam + MediaPipe inference + render skeleton.
- `src/motionClassifier.ts`: trích xuất feature + phân loại động tác.
- `index.html`: UI tối giản, scan nhanh.

## Cài đặt
```bash
cd pickleball-mediapipe
npm install
npm run dev
```
Mở URL Vite (thường là `http://localhost:5173`).

## Cách dùng
1. Nhấn **Dùng webcam** để chạy realtime.
2. Hoặc chọn file video Pickleball ở ô upload.
3. Theo dõi label động tác ở panel phải và console log.

## Logic phân loại (heuristic baseline)
Feature dùng để phân loại:
- Góc khuỷu tay: `shoulder-elbow-wrist`
- Góc vai: `hip-shoulder-elbow`
- Cổ tay so với vai/hông (cao - trung - thấp)
- Vận tốc cổ tay theo trục ngang/dọc giữa các frame
- Tay có cắt qua trục thân hay không (gợi ý backhand)

Rule nổi bật:
- **Serve**: tay thấp (dưới hông) rồi đi lên nhanh, tay duỗi khá lớn.
- **Dink**: tốc độ nhỏ, vùng tiếp xúc trung bình, không có swing mạnh.
- **Drive**: swing trung-bình đến nhanh tại vùng trung, thân hướng tấn công.
- **Lob**: điểm tiếp xúc cao, hướng vợt đưa bóng lên.
- **Volley**: tiếp xúc cao/trước người, nhịp đánh gọn.
- **Forehand/Backhand**: phân theo hướng tay thuận và việc cắt qua trục thân.

## Mở rộng đề xuất
- Thêm chế độ "calibration" 5-10s cho từng người để auto-tune threshold.
- Lưu feature sequence và train KNN đơn giản để tăng độ ổn định.
- Ghép thêm landmark cổ tay/bàn tay từ MediaPipe Hands để nhận diện spin chính xác hơn.

## Lưu ý
- Đây là baseline heuristic, độ chính xác phụ thuộc góc camera và người chơi.
- Nếu sai nhãn: tinh chỉnh ngưỡng trong `motionClassifier.ts` (các giá trị velocity/angle).
