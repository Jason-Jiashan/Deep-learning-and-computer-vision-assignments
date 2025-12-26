# inference.py
# ------------------------------------------------------------------------------
# 1) Video Analysis: 车辆检测+跟踪+（可选）RL 信号覆盖到视频
# 2) Simulation   : 固定/强化学习信号对比 → queue_plot.png & fuel_plot.png
# 本版在 Video 分支新增“真实流量曲线”生成，文件名 *_real.png
# ------------------------------------------------------------------------------

import cv2, os, torch, numpy as np
from traffic_models.agent import DQNAgent
from traffic_utils.env     import TrafficEnv
from traffic_utils.plotter import plot_queue_lengths, plot_fuel_consumption
from traffic_utils import plotter as _plt_mod

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    tracker_available = True
except ImportError:
    tracker_available = False

if not hasattr(_plt_mod, "_orig_plot_fuel"):          # 避免重复包裹
    _plt_mod._orig_plot_fuel = _plt_mod.plot_fuel_consumption

    def _safe_plot_fuel(time_steps, fuel_fixed, *args, **kw):
        # 若 fuel_fixed 为空，用 0-list 填充到相同长度
        if len(fuel_fixed) == 0:
            fuel_fixed = [0.0] * len(time_steps)
        return _plt_mod._orig_plot_fuel(time_steps, fuel_fixed, *args, **kw)

    _plt_mod.plot_fuel_consumption = _safe_plot_fuel
    globals()["plot_fuel_consumption"] = _plt_mod.plot_fuel_consumption
    globals()["plot_queue_lengths"] = _plt_mod.plot_queue_lengths


# --------------------------- YOLOv5 权重 -------------------------------------------------
yolo_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=r'E:/Deep learning and computer vision assignments/traffic_models/yolov5/runs/train/Finished/best_1.pt',
)
# ----------------------------------------------------------------------------------------

def run_inference(use_rl=True, source=None, sim_steps=100):
    """
    • source=None  → 纯 Simulation   (固定 vs RL)
    • source=video → 检测+跟踪 (+RL 逻辑)，并可生成真实流量曲线 *_real.png
    """
    # ============================ VIDEO ANALYSIS =========================================
    if source:
        cap = cv2.VideoCapture(source)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        out    = cv2.VideoWriter("output_annotated.mp4",
                                 cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

        tracker = DeepSort(max_age=15, n_init=1, max_iou_distance=0.7) if tracker_available else None

        phase, phase_timer, fixed_cycle = 0, 0, 50
        agent = None
        if use_rl:
            agent = DQNAgent(state_dim=3, action_dim=2)
            agent.load_state_dict(torch.load("runs/dqn_agent.pth", map_location='cpu'))
            agent.eval()

        print("Processing video for detection and tracking...")

        # === ADDED: 真实流量/能耗统计准备 =========================================
        STEP_FRAMES = int(fps)          # 约 1 秒汇总一次
        q_hist_NS, q_hist_EW   = [], []
        fuel_rl_cum            = []
        idle_fuel_rl           = 0.0
        # =========================================================================

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ---------- YOLOv5 推断 ----------

            results = yolo_model(frame)
            dets = []
            for *xyxy, conf, cls in results.xyxy[0]:
                if float(conf) < 0.10:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                dets.append((x1, y1, x2, y2, float(conf), int(cls)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色检测框
            print(f"[{frame_idx}] detections:", len(results.xyxy[0]))

            # ---------- DeepSORT ----------
            tracks = tracker.update_tracks(dets, frame=frame) if tracker else []

            # ---------- 车辆计数 ----------
            ns_count = ew_count = 0
            for t in tracks:
                if not t.is_confirmed():
                    continue
                x1, y1, x2, y2 = t.to_tlbr()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if center_x < width / 2:   # 粗略按左右半屏归 NS
                    ew_count += 1
                else:
                    ns_count += 1  # demo 简化，同样算进 NS

            # ---------- 信号决策 ----------
            if use_rl and agent:
                state = torch.FloatTensor([ns_count, ew_count, phase]).unsqueeze(0)
                action = int(torch.argmax(agent(state), 1).item())
                if action == 1:
                    phase, phase_timer = 1 - phase, 0
            else:
                if phase_timer >= fixed_cycle:
                    phase, phase_timer = 1 - phase, 0
            phase_timer += 1

            # ---------- 画框 ----------
            for t in tracks:
                if not t.is_confirmed(): continue
                x1, y1, x2, y2 = map(int, t.to_tlbr())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"ID:{t.track_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(frame,
                        "NS GREEN" if phase==0 else "EW GREEN",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0,255,0) if phase==0 else (0,0,255), 2)

            out.write(frame)

            # === ADDED: 每 STEP_FRAMES 记录一次队列长度 & 油耗 ==================
            if frame_idx % STEP_FRAMES == 0:
                q_hist_NS.append(ns_count)
                q_hist_EW.append(ew_count)
                idle_fuel_rl += ((ns_count+ew_count) * 0.2) / 60  # 粗略怠速油耗
                fuel_rl_cum.append(idle_fuel_rl)
            # ==================================================================

        cap.release(); out.release(); cv2.destroyAllWindows()
        print("Output saved to output_annotated.mp4")

        # === ADDED: 生成 *_real.png =============================================
        if q_hist_NS:
            os.makedirs("runs", exist_ok=True)
            t_axis = list(range(len(q_hist_NS)))
            plot_queue_lengths(t_axis,
                               q_hist_NS, q_hist_EW,        # 固定时序=虚线，这里 None
                               ns_rl=None, ew_rl=None,
                               save_path="runs/queue_plot_real.png")

            plot_fuel_consumption(t_axis,
                                  [],                        # fixed 留空
                                  fuel_rl=fuel_rl_cum,
                                  save_path="runs/fuel_plot_real.png")
            print("Real-flow plots saved to runs/queue_plot_real.png & runs/fuel_plot_real.png")
        # ========================================================================

    # ============================ SIMULATION ===================================
    else:
        # SIMULATION MODE (no actual video, use TrafficEnv)
        # Set up identical arrival sequence for fair comparison between RL and fixed control
        np.random.seed(0)
        sim_length = sim_steps
        # Generate traffic arrival sequences (e.g., Poisson distribution of arrivals)
        arrivals_NS = np.random.poisson(lam=2, size=sim_length)
        arrivals_EW = np.random.poisson(lam=2, size=sim_length)
        # Run simulation with RL control
        env_rl = TrafficEnv(arrivals_NS=arrivals_NS.tolist(), arrivals_EW=arrivals_EW.tolist(), max_steps=sim_length)
        # Load trained RL agent
        agent = None
        if use_rl:
            agent = DQNAgent(state_dim=3, action_dim=2)
            agent.load_state_dict(torch.load("runs/dqn_agent.pth", map_location='cpu'))
            agent.eval()
        # Data arrays for plotting
        time_steps = list(range(sim_length))
        ns_count_rl = []
        ew_count_rl = []
        fuel_cumulative_rl = []
        ns_count_fixed = []
        ew_count_fixed = []
        fuel_cumulative_fixed = []
        # Simulate RL control
        phase = 0  # NS green
        env_state = env_rl.reset()
        idle_fuel = 0.0  # cumulative fuel (in liters)
        for t in time_steps:
            # RL action decision
            if use_rl and agent is not None:
                state_tensor = torch.FloatTensor(env_state).unsqueeze(0)
                with torch.no_grad():
                    q_vals = agent(state_tensor)
                    action = int(torch.argmax(q_vals, dim=1).item())
            else:
                # If RL not used, default to not switching (we will handle fixed later separately)
                action = 0
            # Apply action
            next_state, reward, done = env_rl.step(action)
            # Gather data
            # State format: [queue_NS, queue_EW, phase]
            ns_count_rl.append(env_state[0])
            ew_count_rl.append(env_state[1])
            # Calculate fuel consumption for this step:
            idle_vehicles = -reward  # reward = - idle_count
            step_fuel = (idle_vehicles * 0.2) / 60.0  # 0.2 L/min per vehicle, step assumed 1 sec
            idle_fuel += step_fuel
            fuel_cumulative_rl.append(idle_fuel)
            env_state = next_state
            if done:
                break
        # Simulate Fixed-time control for comparison
        env_fixed = TrafficEnv(arrivals_NS=arrivals_NS.tolist(), arrivals_EW=arrivals_EW.tolist(), max_steps=sim_length)
        phase = 0
        phase_timer = 0
        fixed_cycle = 10  # e.g., switch every 10 steps (can adjust)
        env_state = env_fixed.reset()
        idle_fuel = 0.0
        for t in time_steps:
            # Determine fixed action based on cycle
            if phase_timer >= fixed_cycle:
                phase = 1 - phase
                phase_timer = 0
            phase_timer += 1
            action = 0 if phase == env_state[2] else 1  # ensure env phase matches our phase variable
            next_state, reward, done = env_fixed.step(action)
            # Record data
            ns_count_fixed.append(env_state[0])
            ew_count_fixed.append(env_state[1])
            idle_vehicles = -reward
            step_fuel = (idle_vehicles * 0.2) / 60.0
            idle_fuel += step_fuel
            fuel_cumulative_fixed.append(idle_fuel)
            env_state = next_state
            if done:
                break
        # Ensure output directory exists for plots
        os.makedirs("runs", exist_ok=True)
        # Plot and save results
        plot_queue_lengths(time_steps, ns_count_fixed, ew_count_fixed,
                           ns_rl=ns_count_rl if use_rl else None,
                           ew_rl=ew_count_rl if use_rl else None,
                           save_path="runs/queue_plot.png")
        plot_fuel_consumption(time_steps, fuel_cumulative_fixed,
                              fuel_rl=fuel_cumulative_rl if use_rl else None,
                              save_path="runs/fuel_plot.png")
        print("Simulation completed. Plots saved to runs/queue_plot.png and runs/fuel_plot.png")
