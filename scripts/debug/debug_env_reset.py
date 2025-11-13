# debug_env_reset.py
import time
import numpy as np

# TODO: chỉnh lại đường dẫn import cho phù hợp
from src.utils.config_loader import load_config
from src.environment.drone_env import DroneDeliveryEnv  # tên class env của bạn
# hoặc từ nơi bạn định nghĩa environment chính

def main():
    cfg = load_config("config/training/environment_config.yaml")
    env = DroneDeliveryEnv(cfg)

    for ep in range(3):
        print("\n==============================")
        print("Episode", ep)
        obs = env.reset()
        print("obs type:", type(obs))
        if hasattr(obs, "shape"):
            print("obs shape:", obs.shape)
        else:
            try:
                print("len(obs):", len(obs))
            except:
                pass

        # nếu env có method đọc state từ AirSim, bạn có thể log thêm ở đây
        if hasattr(env, "bridge"):
            try:
                s = env.bridge.get_drone_state()
                print("drone state right after reset:", s.position)
            except Exception as e:
                print("cannot get drone state:", e)

        # đợi 2s không làm gì để xem drone có rơi không
        for i in range(4):
            time.sleep(0.5)
            if hasattr(env, "bridge"):
                try:
                    s = env.bridge.get_drone_state()
                    print(f"  [idle t={0.5*(i+1):.1f}s] pos={s.position}")
                except Exception as e:
                    print("  error reading state:", e)

        # Thử vài step với action 0
        if hasattr(env, "action_space") and hasattr(env.action_space, "sample"):
            action_zero = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            # fallback: random
            action_zero = None

        for t in range(5):
            action = action_zero
            if action is None:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            print(f"step {t}: reward={reward}, done={done}")
            if hasattr(env, "bridge"):
                try:
                    s = env.bridge.get_drone_state()
                    print("        pos after step:", s.position)
                except:
                    pass

            if done:
                print("episode ended at step", t)
                break

if __name__ == "__main__":
    main()
