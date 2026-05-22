import asyncio
from bleak import BleakScanner

async def scan():
    print("正在扫描蓝牙设备...")
    devices = await BleakScanner.discover(timeout=10.0)
    
    for d in devices:
        if d.address.upper() == "20:19:08:21:31:03":
            print(f"✓ 找到你的手套！")
            print(f"名称: {d.name}")
            print(f"地址: {d.address}")
            print(f"RSSI: {d.rssi} dBm")
            print(f"详情: {d.details}")
        # elif "Stretch" in (d.name or ""):  # 如果有名称也可以这样过滤

asyncio.run(scan())