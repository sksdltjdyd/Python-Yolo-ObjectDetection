import pyrealsense2 as rs

def check_firmware():
    ctx = rs.context()
    devices = ctx.query_devices()
    
    for dev in devices:
        print(f"Device: {dev.get_info(rs.camera_info.name)}")
        print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
        
        # D455 권장 펌웨어: 5.13.0.50 이상
        if "D455" in dev.get_info(rs.camera_info.name):
            fw_version = dev.get_info(rs.camera_info.firmware_version)
            if fw_version < "5.13.0.50":
                print("⚠️ 펌웨어 업데이트 필요!")
                print("rs-fw-update 도구 사용 또는 RealSense Viewer에서 업데이트")