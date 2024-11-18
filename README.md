# Hand Detection and Data Collection Tool

โปรเจคนี้ทำขึ้นโดยใช้ `OpenCV` และ `cvzone` เครื่องมือนี้จะทำการจับภาพมือจาก webcam ประมวลผล และบันทึกภาพที่ได้ลงใน folder ที่กำหนด สามารถนำไปใช้สำหรับการเก็บข้อมูลท่าทางมือเพื่อฝึกโมเดล Machine Learning ได้

## คุณสมบัติ
- **ตรวจจับมือแบบเรียลไทม์** โดยใช้ webcam
- **ครอบและปรับขนาด** รูปภาพที่ตรวจจับได้ให้พอดีกับขนาด 300x300 px
- **บันทึกรูปภาพมือ** ด้วยการกดปุ่ม `s`
- **สร้างโฟลเดอร์อัตโนมัติ** สำหรับเก็บรูปภาพ

## ไลบรารีที่ใช้
- **OpenCV (`cv2`)**: สำหรับการเข้าถึง webcam ประมวลผลภาพ และแสดงผล
- **cvzone (HandTrackingModule)**: สำหรับการตรวจจับมือโดยใช้ MediaPipe
- **NumPy (`numpy`)**: สำหรับการจัดการและสร้างอาเรย์ของภาพ
- **Math (`math`)**: สำหรับการคำนวณการปรับขนาดภาพ
- **OS (`os`)**: สำหรับการจัดการไฟล์และโฟลเดอร์
- **Time (`time`)**: สำหรับสร้างชื่อไฟล์ให้ไม่ซ้ำกันโดยใช้ timestamp

## วิธีการใช้งาน
- **`Data/your_sample`**: ตั้งชื่อโฟลเดอร์ที่ต้องการเก็บภาพมือที่ตรวจจับได้
- **`s`**: บันทึกรูปภาพมือที่ตรวจจับได้
- **`q`**: ปิดโปรแกรม