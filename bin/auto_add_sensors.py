import xml.etree.ElementTree as ET
import numpy as np

INPUT_XML = "/home/zmy/MyProject/models/inspirehand/inspirehand.xml"
OUTPUT_XML = "/home/zmy/MyProject/models/inspirehand/hand_touchgrid.xml"


# =========================
# 四元数工具
# =========================
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    qv = np.array([0, *v])
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[1:]


def parse_vec(s):
    return np.array([float(x) for x in s.split()])


def vec_to_str(v):
    return " ".join(f"{x:.8g}" for x in v)


# =========================
# offset 配置（核心！！）
# =========================
OFFSET_CONFIG = {
    # 👉 默认：沿局部 Z 轴向外顶一点
    "default_pos": np.array([0.0, 0.0, -0.005]),
    "default_quat": np.array([0.0, 1.0, 0.0, 0.0]),

    # 👉 示例：如果某些面方向反了可以单独修
    # "skin_2_2_p": (
    #     np.array([0, 0, 0.002]),
    #     np.array([0, 1, 0, 0])  # 180° flip
    # )
}


# =========================
# size 规则
# =========================
def get_size(name):
    if "_0_" in name:
        return "10 7"
    elif "_1_" in name:
        return "8 5"
    else:
        return "6 5"


# =========================
# 插入 extension
# =========================
def ensure_extension(root):
    if root.find("extension") is not None:
        return

    compiler = root.find("compiler")

    ext = ET.Element("extension")
    plugin = ET.SubElement(ext, "plugin")
    plugin.set("plugin", "mujoco.sensor.touch_grid")

    root.insert(list(root).index(compiler) + 1, ext)


# =========================
# 插入 site（核心逻辑）
# =========================
def insert_sites(root):
    for body in root.iter("body"):
        children = list(body)

        for i, elem in enumerate(children):
            if elem.tag != "geom":
                continue

            name = elem.attrib.get("name", "")
            if not name.startswith("skin_"):
                continue

            # geom pose
            p_g = parse_vec(elem.attrib.get("pos", "0 0 0"))
            q_g = parse_vec(elem.attrib.get("quat", "1 0 0 0"))

            # offset
            if name in OFFSET_CONFIG:
                p_off, q_off = OFFSET_CONFIG[name]
            else:
                p_off = OFFSET_CONFIG["default_pos"]
                q_off = OFFSET_CONFIG["default_quat"]

            # 👉 合成 pose
            p_site = p_g + quat_rotate(q_g, p_off)
            q_site = quat_mul(q_g, q_off)

            # 创建 site
            site = ET.Element("site")
            site.set("name", f"touch_grid_{name}")
            site.set("pos", vec_to_str(p_site))
            site.set("quat", vec_to_str(q_site))
            site.set("size", "0.001")
            site.set("type", "sphere")
            site.set("rgba", "1 0 0 1")

            body.insert(i + 1, site)


# =========================
# 插入 sensor plugin
# =========================
def insert_sensors(root):
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")

    for elem in root.iter("geom"):
        name = elem.attrib.get("name", "")
        if not name.startswith("skin_"):
            continue

        plugin = ET.SubElement(sensor, "plugin")
        plugin.set("name", f"touch_grid_{name}")
        plugin.set("plugin", "mujoco.sensor.touch_grid")
        plugin.set("objtype", "site")
        plugin.set("objname", f"touch_grid_{name}")

        def add(k, v):
            c = ET.SubElement(plugin, "config")
            c.set("key", k)
            c.set("value", v)

        add("nchannel", "1")
        add("size", get_size(name))
        add("fov", "90 90")
        add("gamma", "0.1")


# =========================
# 主函数
# =========================
def main():
    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    ensure_extension(root)
    insert_sites(root)
    insert_sensors(root)

    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)

    print("✅ 完成！输出文件：", OUTPUT_XML)


if __name__ == "__main__":
    main()