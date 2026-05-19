"""
Vehicle Detection and Multi-Object Tracking Script with Progress Indicators

1. 加载训练好的 YOLOv8 模型
2. 处理视频流，利用模型进行物体检测，并借助内置跟踪器分配跟踪 ID
3. 根据虚拟计数线统计通过的车辆总数
4. 保存处理后的视频与统计日志
5. 控制台输出进度标记（帧数、计数）
"""

import cv2
import numpy as np
from ultralytics import YOLO
import datetime


class VehicleTrackerAndCounter:
    def __init__(
        self,
        model_path: str,
        video_source: str,
        line_position: float = 0.5,
        line_thickness: int = 2,
        output_video: str = "output_tracked.mp4",
        log_path: str = "counting_log.txt"
    ):
        """
        初始化车辆追踪与计数系统

        :param model_path:        训练好的 YOLOv8 模型路径
        :param video_source:      输入视频文件路径
        :param line_position:     虚拟计数线的垂直位置（范围 0~1，0表示顶部，1表示底部）
        :param line_thickness:    计数线的像素厚度
        :param output_video:      输出视频保存路径
        :param log_path:          统计日志保存路径
        """
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.line_position = line_position
        self.line_thickness = line_thickness
        self.output_video = output_video
        self.log_path = log_path

        # 用于存储被统计过的目标，防止重复计数
        self.counted_ids = set()
        self.total_count = 0

        # 用于方向统计（可选）
        self.direction_counts = {"UP": 0, "DOWN": 0}

        # 日志文件初始化
        with open(self.log_path, "w") as f:
            f.write("Timestamp,ObjectID,Class,OriginalDirection,CalculatedDirection,TotalCount\n")

    def _is_line_crossed(self, current_y, previous_y, line_y):
        """
        判断物体是否跨过了计数线。
        跨线条件：初始位置与线在同一侧，而当前位置移动到了另一侧。

        :param current_y:  当前帧物体中心 y 坐标
        :param previous_y: 前一帧物体中心 y 坐标
        :param line_y:     计数线的 y 坐标
        :return:           是否触发跨线，以及跨线方向
        """
        if previous_y is None or current_y is None:
            return False, None

        before_cross = previous_y < line_y
        after_cross = current_y > line_y
        if before_cross and after_cross:
            return True, "DOWN"
        before_cross = previous_y > line_y
        after_cross = current_y < line_y
        if before_cross and after_cross:
            return True, "UP"
        return False, None

    def run(self):
        """运行跟踪与计数流程，带进度标记"""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("错误：无法打开视频文件。")
            return

        # 获取视频基本信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("=" * 50)
        print("开始处理视频")
        print(f"分辨率: {width} x {height}")
        print(f"帧率: {fps:.2f} FPS")
        print(f"总帧数: {total_frames}")
        print(f"计数线位置: Y={int(self.line_position * height)} (垂直位置 {self.line_position*100:.0f}%)")
        print("=" * 50)

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

        # 计算计数线的实际位置
        line_y = int(self.line_position * height)

        # 用于跟踪每个物体的上一帧位置（中心点 y 坐标）
        prev_centers = {}

        frame_count = 0
        # 控制台进度打印间隔（帧）
        progress_interval = 100

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 每 progress_interval 帧打印一次进度
            if frame_count % progress_interval == 0 or frame_count == total_frames:
                progress_pct = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"[进度] 已处理 {frame_count}/{total_frames} 帧 ({progress_pct:.1f}%), 当前累计计数: {self.total_count}")

            # YOLOv8 进行追踪推理
            results = self.model.track(frame, persist=True, verbose=False)

            # 提取检测结果与跟踪 ID
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # 边界框
                track_ids = results[0].boxes.id.cpu().numpy()  # 跟踪 ID
                classes = results[0].boxes.cls.cpu().numpy()  # 类别索引
                confs = results[0].boxes.conf.cpu().numpy()  # 置信度

                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    class_name = self.model.names[int(cls)]

                    # 绘制边界框与 ID
                    label = f"ID:{int(track_id)} {class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    # 绘制中心点
                    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

                    # 计数线跨线检测
                    if track_id not in self.counted_ids:
                        prev_center_y = prev_centers.get(track_id)
                        crossed, direction = self._is_line_crossed(center_y, prev_center_y, line_y)
                        if crossed:
                            self.total_count += 1
                            self.counted_ids.add(track_id)
                            self.direction_counts[direction] += 1
                            # 记录日志
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            log_entry = f"{timestamp},{int(track_id)},{class_name},{direction},{direction},{self.total_count}\n"
                            with open(self.log_path, "a") as f:
                                f.write(log_entry)
                            # 不打印跨线事件，仅记录日志

                    # 更新轨迹
                    prev_centers[track_id] = center_y

            # 绘制计数线与实时计数显示
            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), self.line_thickness)
            cv2.putText(
                frame, f"Total Count: {self.total_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
            )
            cv2.putText(
                frame, f"UP: {self.direction_counts['UP']}  DOWN: {self.direction_counts['DOWN']}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

            out.write(frame)

            # 可选：实时显示（调试用），按 'q' 可提前退出
            # cv2.imshow("Vehicle Tracker", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("用户提前终止处理")
            #     break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 最终统计
        print("\n" + "=" * 50)
        print("处理完成！")
        print(f"总处理帧数: {frame_count}")
        print(f"总计通过车辆数: {self.total_count}")
        print(f"  向上 (UP): {self.direction_counts['UP']}")
        print(f"  向下 (DOWN): {self.direction_counts['DOWN']}")
        print(f"输出视频保存至: {self.output_video}")
        print(f"详细日志保存至: {self.log_path}")
        print("=" * 50)


if __name__ == "__main__":
    tracker = VehicleTrackerAndCounter(
        model_path="/home/shichenqiao/Deep_Learning_and_Spatial_Intelligence/PJ2/task2/runs/detect/train/weights/best.pt",
        video_source="/home/shichenqiao/Deep_Learning_and_Spatial_Intelligence/PJ2/task2/time_lapse_video.mp4",
        line_position=0.7,
        output_video="/home/shichenqiao/Deep_Learning_and_Spatial_Intelligence/PJ2/task2/output_counted.mp4"
    )
    tracker.run()