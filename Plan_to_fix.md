1. Drone spawn đúng vị trí, đúng độ cao, đúng hướng, không còn hiện tượng spawn ở chỗ lạ rồi từ từ rơi xuống.
2. Quy trình reset  takeoff  bắt đầu episode rõ ràng, ổn định cho RL.
3. Cấu hình (config) và code thống nhất với nhau và với nội dung trong report.
4. Có các test / log đơn giản để kiểm tra nhanh sau này (smoke test).

---

- Vấn đề
  - Hiện có 2 nguồn toạ độ spawn:
    - File config (YAML) dùng toạ độ rất lớn kiểu (6000, -3000, 300) (từ report cũ).
    - AirSimBridge có default spawn nhỏ kiểu (60, -30, -3) nếu không nhận config.
  - Nếu xài (6000, -3000, 300) trong hệ NED thì z = 300 là drone nằm sâu dưới mặt đất, còn nếu default (60, -30, -3) mà world/map lệch origin thì drone xuất hiện ở chỗ lạ lạ.

- File liên quan
  - config/training/environment_config.yaml (hoặc file config chính bạn đang dùng khi train).
  - src/bridges/airsim_bridge.py
  - (tuỳ repo) src/environment/world_builder.py nếu có logic gắn spawn vào cấu trúc tòa nhà.

- Hướng sửa
  1. Chọn một toạ độ spawn chuẩn mà bạn đã test tay thấy ổn (ví dụ (60.0, -30.0, -3.0) và yaw = 0).
  2. Trong file config:
     yaml
     spawn_location: [60.0, -30.0, -3.0]
     spawn_orientation: [0.0, 0.0, 0.0]

  3. Trong AirSimBridge.__init__:
     - Bỏ bớt default ảo, bắt buộc config phải truyền spawn_location:
       python
       if "spawn_location" not in config:
           raise ValueError("spawn_location must be set in config")

       self.spawn_location = tuple(config["spawn_location"])
       self.spawn_orientation = tuple(config.get("spawn_orientation", (0.0, 0.0, 0.0)))

  4. (Nếu cần) kiểm tra world_builder để chắc origin map và spawn không bị lệch quá xa.

---

- Vấn đề
  - AirSim dùng hệ NED: z âm là lên, z dương là xuống.
  - Config cũ dùng z = +300  drone bị đặt 300 m dưới origin (trong đất hoặc rất sâu), gây hiện tượng spawn kỳ lạ.

- File liên quan
  - config/training/environment_config.yaml
  - src/bridges/airsim_bridge.py (hàm reset_drone / xử lý pose).

- Hướng sửa
  - Bảo đảm spawn_location[2] là giá trị âm nhỏ (ví dụ -3.0  drone cách mặt sàn 3m).
  - Tuỳ chọn: thêm sanity check trong reset_drone:
    python
    x, y, z = self.spawn_location
    if z  0:
        self.logger.warning(f"Spawn z={z}  0 (NED). Forcing z=-3.0")
        z = -3.0
        self.spawn_location = (x, y, z)

---

- Vấn đề
  - Hiện tại reset() trong env chỉ gọi reset_drone() (đặt lại pose) mà không cất cánh.
  - Trong train: sau reset() agent chưa kịp ra lệnh, drone đã bị trọng lực kéo  spawn xong rơi từ từ.
  - Trong code tay bạn test thì thường có takeoffAsync, moveToZAsync nên không bị rơi.

- File liên quan
  - src/environment/airsim_env.py hoặc file định nghĩa env chính bạn dùng cho RL (ví dụ AirSimNavigationEnv, DroneDeliveryEnv, ).
  - scripts/training/train_full_curriculum.py hoặc scripts/training/train_phase.py (chỗ tạo env và gọi reset()).

- Hướng sửa (gợi ý)
  1. Trong class env, sau khi reset_drone(to_spawn=True) thì:
     python
     def reset(self):
         self.bridge.reset_drone(to_spawn=True)

         ok = self.bridge.takeoff(altitude=self.initial_altitude)
         if not ok:
             self.logger.warning("Takeoff failed, retrying basic reset...")
             self.bridge.reset_drone(to_spawn=True)
             self.bridge.takeoff(altitude=self.initial_altitude)

         time.sleep(0.5)

         obs = self._get_observation()
         self.current_step = 0
         return obs

  2. Nếu bạn có altitude-hold controller riêng (như mô tả trong report), bật nó ngay sau reset để PPO chỉ điều khiển ngang (vx, vy, yaw).

---

- Vấn đề
  - airsim.to_quaternion(pitch, roll, yaw) yêu cầu đúng thứ tự. Nếu config đang ghi [yaw, pitch, roll]  drone spawn ra bị nghiêng, trượt (nhìn như đang rơi lệch).

- File liên quan
  - config/training/environment_config.yaml
  - src/bridges/airsim_bridge.py (nơi tạo airsim.Pose).

- Hướng sửa
  - Tạm thời set spawn_orientation: [0.0, 0.0, 0.0] để drone đứng thẳng.
  - Nếu cần set yaw khác 0, đảm bảo truyền đúng (pitch, roll, yaw).

---

- Vấn đề
  - Nếu random spawn quá rộng (vài chục mét) rất dễ spawn drone ra ngoài khu vực tòa nhà.

- File liên quan
  - config/training/environment_config.yaml (cờ spawn_randomization).
  - Env file (nơi áp dụng random vào spawn_location).

- Hướng sửa
  1. Trong env, viết helper:
     python
     def sample_spawn(self):
         base = np.array(self.config["spawn_location"])
         noise_xy = np.random.uniform(-2.0, 2.0, size=2)
         x = base[0] + noise_xy[0]
         y = base[1] + noise_xy[1]
         z = base[2]
         return (x, y, z)

  2. Chỉ random khi flag bật, và có check giới hạn:
     python
     if self.config.get("spawn_randomization", False):
         self.bridge.spawn_location = self.sample_spawn()

  3. Có thể thêm guard: nếu x hoặc y  một ngưỡng (vd. 200m)  log warning và fallback về spawn_location gốc.

---

- Vấn đề
  - Nếu origin tòa nhà và spawn drone không cùng hệ quy chiếu, drone có thể bắt đầu ở một góc xa map (dù nhìn tạm ổn khi bay tay).

- File liên quan
  - src/environment/world_builder.py (hoặc file tương đương).
  - Config map (nếu có file YAML riêng cho layout).

- Hướng sửa
  - Kiểm tra xem origin tòa nhà (0, 0, 0) trong world builder có khớp với khu vực bạn muốn drone spawn.
  - Vẽ sơ đồ/ghi lại offset nếu có, rồi set spawn_location cho đúng.

---

- Vấn đề
  - Nếu người/obstacle spawn trong radius quá gần drone, khi reset có thể xảy ra va chạm ngay bước đầu (gây reward âm lớn, episode kết thúc sớm).

- File liên quan
  - File spawn người / dynamic obstacles (có thể trong world_builder hoặc file env riêng).

- Hướng sửa
  - Đảm bảo khoảng cách tối thiểu giữa drone spawn và obstacle spawn (vd.  5 m).
  - Khi test tay, reset nhiều lần để chắc không có case spawn dính nhau.

---

- Vấn đề
  - Nếu condition chuyển phase (success rate, số episode, v.v.) cấu hình khác report hoặc code có bug logic, training có thể dậm chân ở phase dễ hoặc nhảy phase quá sớm.

- File liên quan
  - scripts/training/train_full_curriculum.py
  - scripts/training/train_phase.py (nếu dùng).

- Hướng sửa
  - Đọc lại điều kiện chuyển phase trong report  đối chiếu với code.
  - Thêm log rõ ràng: mỗi lần đổi phase, in ra success rate, số episode, tên phase mới.
  - Tuỳ chọn: cho phép override phase bằng flag debug để test nhanh từng map.

---

- Vấn đề
  - Report mô tả observation 35 chiều, reward có nhiều term (goal, distance, time, control effort, jerk, collision). Nếu code khác quá nhiều  khó giải thích kết quả train. Ưu tiên sử dụng 40 chiều

- File liên quan
  - Env (hàm build observation, build reward).
  - Utils (nếu tách reward ra file riêng).

- Hướng sửa
  - Log lại obs.shape sau reset() và step()  confirm đúng số chiều.
  - Gom các thành phần reward vào một hàm compute_reward() với comment mapping trực tiếp sang công thức trong report.

---

- Vấn đề
  - lr, batch size, clip range, số epoch, gamma, lambda nếu khác report mà không ghi lại  khó tái lập kết quả.

- File liên quan
  - File config PPO (nếu có) hoặc trực tiếp trong train_full_curriculum.py.

- Hướng sửa
  - Đưa toàn bộ hyperparams vào 1 block config (YAML hoặc dict Python).
  - Hoặc update lại report nếu bạn đã cố tình thay đổi.

---

- Vấn đề
  - Nếu không set seed cho random, numpy, và framework RL  mỗi lần train hơi khác nhau, khó kiểm tra bug.

- File liên quan
  - Script train chính (train_full_curriculum.py).

- Hướng sửa
  - Thêm đoạn:
    python
    import random
    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

---

- File đề xuất
  - scripts/debug/test_spawn_and_takeoff.py

- Ý tưởng
  - Tạo bridge + env, gọi reset() 510 lần, chỉ lấy obs và đợi vài giây.
  - Mục tiêu: mỗi lần reset drone đều spawn đúng chỗ, lên đúng độ cao, không từ từ rơi xuống.

- File liên quan
  - Env hoặc trainer.

- Hướng sửa
  - Sau reset(), log:
    - spawn_location đang dùng,
    - orientation,
    - độ cao thực của drone đọc từ AirSim.
  - Khi có bug, chỉ cần xem log là biết ngay ổ lỗi (sai z, sai orientation, sai randomization).

---

1. (Must do) 2.1.1 + 2.1.2 + 2.1.3
    Thống nhất spawn + xử lý takeoff trong reset() để hết hiện tượng spawn lạ + rơi xuống.
2. (Should do) 2.1.4 + 2.1.5 + 2.2.x
    Dọn sạch các vấn đề orientation và random spawn, confirm spawn objects ổn.
3. (Should do) 2.3.x
    Rà lại curriculum  điều kiện chuyển phase.
4. (Nice to have nhưng nên làm) 2.4.x + 2.5.x
    Đồng bộ report, thêm seed, thêm test  log cho dễ debug về sau.

Hoàn thành checklist này là bạn sẽ có một pipeline train ổn định, spawn đúng, reset rõ ràng, dễ debug, và nhất quán với báo cáo.
