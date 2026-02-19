"""快速测试MuJoCo模型的基本功能"""
import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
data = mujoco.MjData(model)

print("✓ MuJoCo模型信息：")
print(f"  - 自由度: {model.nv}")
print(f"  - 执行器数量: {model.nu}")
print(f"  - 传感器数量: {model.nsensor}")

# 运行10步仿真
mujoco.mj_resetData(model, data)
for i in range(10):
    mujoco.mj_step(model, data)

# 获取杆头位置
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_head_site")
club_head_pos = data.site_xpos[site_id]

print(f"\n✓ 杆头位置: [{club_head_pos[0]:.3f}, {club_head_pos[1]:.3f}, {club_head_pos[2]:.3f}]")

# 计算逆向动力学
mujoco.mj_inverse(model, data)
print(f"✓ 逆向动力学计算完成")

# 检测地面接触
print(f"✓ 当前接触点数量: {data.ncon}")

print("\n🎉 MuJoCo模型工作正常！")
