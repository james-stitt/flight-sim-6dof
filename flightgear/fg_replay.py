import argparse
import glob
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import numpy as np
import pandas as pd

# =============================================================================
# 1. FGNetFDM PROTOCOL ENCODER (v24)
# =============================================================================

FG_NET_FDM_VERSION = 24
FG_NET_FDM_SIZE = 408

def pack_fdm_packet(lat_rad, lon_rad, alt_m, phi_rad, theta_rad, psi_rad):
    """Packs geodetic coordinates and Euler angles into FlightGear's binary struct."""
    header = struct.pack(
        ">II ddd f fff",
        FG_NET_FDM_VERSION, 0,
        lon_rad, lat_rad, alt_m,
        alt_m, phi_rad, theta_rad, psi_rad
    )
    return header.ljust(FG_NET_FDM_SIZE, b"\x00")

# =============================================================================
# 2. CSV PARSER & METADATA EXTRACTION
# =============================================================================

def load_flightgear_csv(filepath):
    """Reads trajectory kinematics and extracts model path from CSV header/columns."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FlightGear replay CSV not found: {filepath}")

    model_path = None

    with open(filepath, "r") as f:
        first_line = f.readline().strip()
        if first_line.startswith("#") and "model_path" in first_line:
            model_path = first_line.split(":", 1)[-1].strip()

    df = pd.read_csv(filepath, comment="#", header=None)

    if model_path is None and df.shape[1] >= 8:
        model_path = str(df.iloc[0, 7]).strip()

    times = df.iloc[:, 0].to_numpy(dtype=float)
    lat_deg = df.iloc[:, 1].to_numpy(dtype=float)
    lon_deg = df.iloc[:, 2].to_numpy(dtype=float)
    alt_ft = df.iloc[:, 3].to_numpy(dtype=float)
    roll_deg = df.iloc[:, 4].to_numpy(dtype=float)
    pitch_deg = df.iloc[:, 5].to_numpy(dtype=float)
    heading_deg = df.iloc[:, 6].to_numpy(dtype=float)

    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    alt_m = alt_ft * 0.3048  # Convert feet to meters
    phi_rad = np.radians(roll_deg)
    theta_rad = np.radians(pitch_deg)
    psi_rad = np.radians(heading_deg)

    return model_path, times, lat_rad, lon_rad, alt_m, phi_rad, theta_rad, psi_rad, lat_deg[0], lon_deg[0], alt_ft[0]

# =============================================================================
# 3. AUTOMATIC AIRCRAFT PACKAGE GENERATOR
# =============================================================================

def prepare_aircraft_package(model_path):
    """Creates/syncs the aircraft package without overwriting manual XML customizations."""
    if not os.path.exists(model_path):
        return None, model_path

    abs_model = os.path.abspath(model_path).replace("\\", "/")
    source_dir = os.path.dirname(abs_model)
    model_filename = os.path.basename(abs_model)
    aircraft_name = os.path.splitext(model_filename)[0]

    staging_root = os.path.abspath("./flightgear/aircraft").replace("\\", "/")
    aircraft_dir = os.path.join(staging_root, aircraft_name).replace("\\", "/")
    os.makedirs(aircraft_dir, exist_ok=True)

    # Sync 3D assets (.ac, textures)
    for item in os.listdir(source_dir):
        src_item = os.path.join(source_dir, item)
        if os.path.isfile(src_item) and item.lower().endswith((".ac", ".xml", ".png", ".rgb", ".dds")):
            dest_item = os.path.join(aircraft_dir, item)
            # Do not overwrite sphere-set.xml if you edited it manually
            if item.lower().endswith("-set.xml") and os.path.exists(dest_item):
                continue
            if not os.path.exists(dest_item) or os.path.getmtime(src_item) > os.path.getmtime(dest_item):
                shutil.copy2(src_item, dest_item)

    set_xml_path = os.path.join(aircraft_dir, f"{aircraft_name}-set.xml")

    # Only generate if the file does NOT exist, preserving manual edits
    if not os.path.exists(set_xml_path):
        cam_dist = 25.0
        cam_height = 9.0
        set_xml_content = f"""<?xml version="1.0"?>
<PropertyList>
  <sim>
    <description>{aircraft_name}</description>
    <flight-model>null</flight-model>
    <model>
      <path>{model_filename}</path>
    </model>
    <chase-distance-m>-{cam_dist}</chase-distance-m>
    <current-view>
      <view-number>1</view-number>
      <z-offset-m>-{cam_dist}</z-offset-m>
      <y-offset-m>{cam_height}</y-offset-m>
      <x-offset-m>0.0</x-offset-m>
    </current-view>
    <view n="1">
      <config>
        <default-field-of-view-deg>55</default-field-of-view-deg>
        <z-offset-m type="double">-{cam_dist}</z-offset-m>
        <y-offset-m type="double">{cam_height}</y-offset-m>
        <x-offset-m type="double">0.0</x-offset-m>
        <target-z-offset-m type="double">0.0</target-z-offset-m>
        <target-y-offset-m type="double">0.0</target-y-offset-m>
      </config>
    </view>
  </sim>
</PropertyList>
"""
        with open(set_xml_path, "w") as f:
            f.write(set_xml_content)

    return staging_root, aircraft_name

# =============================================================================
# 4. SUBPROCESS CONTROLLER
# =============================================================================

def resolve_fg_binary(binary_name):
    found_path = shutil.which(binary_name)
    if found_path:
        return found_path

    if sys.platform == "win32":
        search_patterns = [
            r"C:\Program Files\FlightGear*\bin\fgfs.exe",
            r"C:\Program Files (x86)\FlightGear*\bin\fgfs.exe",
        ]
        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[-1]

    raise FileNotFoundError(f"Could not find FlightGear executable '{binary_name}'.")

def start_flightgear(model_path, fg_binary, udp_port, init_lat, init_lon, init_alt_ft):
    """Launches FlightGear directly at the simulation origin without network hangs."""
    executable = resolve_fg_binary(fg_binary)
    staging_root, aircraft_name = prepare_aircraft_package(model_path)

# Desired exterior camera distance in meters (increase to zoom out further)
    cam_distance_m = 25.0
    cam_height_m = 4.0

    cmd = [
            executable,
            "--fdm=null",
            f"--native-fdm=socket,in,60,,{udp_port},udp",
            "--telnet=5501",                             # <-- Add this line
            "--geometry=1280x720",
            "--timeofday=noon",
            "--disable-terrasync",
            "--disable-splash-screen",
            "--disable-sound",
            "--disable-ai-traffic",
            "--disable-real-weather-fetch",
            f"--fg-aircraft={staging_root}",
            f"--aircraft={aircraft_name}",
            f"--lat={init_lat}",
            f"--lon={init_lon}",
            f"--altitude={init_alt_ft}",
        ]

    if staging_root:
        cmd.append(f"--fg-aircraft={staging_root}")

    cmd.append(f"--aircraft={aircraft_name}")
    print(f"Launching FlightGear at Lat: {init_lat:.4f}, Lon: {init_lon:.4f}, Alt: {init_alt_ft:.0f} ft")

    return subprocess.Popen(cmd)

# =============================================================================
# 5. MAIN REPLAY LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Continuous FlightGear trajectory replayer.")
    parser.add_argument("csv_path", help="Path to the FlightGear replay CSV file.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--port", type=int, default=5500, help="FlightGear UDP port (default: 5500)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="FlightGear UDP host (default: 127.0.0.1)")
    parser.add_argument("--fg-bin", type=str, default="fgfs", help="FlightGear executable (default: fgfs)")
    args = parser.parse_args()

    (model_path, times, lat_rad, lon_rad, alt_m, 
     phi_rad, theta_rad, psi_rad, 
     init_lat, init_lon, init_alt_ft) = load_flightgear_csv(args.csv_path)
    
    n_points = len(times)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    fg_proc = start_flightgear(model_path, args.fg_bin, args.port, init_lat, init_lon, init_alt_ft)

    # Broadcast initial state at 10 Hz while FlightGear opens
    print("Holding initial position while FlightGear initializes window...")
    for _ in range(80):  # 8 seconds holding initial state
        p0 = pack_fdm_packet(lat_rad[0], lon_rad[0], alt_m[0], phi_rad[0], theta_rad[0], psi_rad[0])
        sock.sendto(p0, (args.host, args.port))
        time.sleep(0.1)

    # Automatically command FlightGear to switch to External View (View 1)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as telnet_sock:
            telnet_sock.settimeout(2.0)
            telnet_sock.connect(("127.0.0.1", 5501))
            telnet_sock.sendall(b"set /sim/current-view/view-number 1\r\n")
            time.sleep(0.1)
        print("Switched FlightGear to external view.")
    except Exception as e:
        print(f"Could not switch view via telnet: {e}")

    loop_count = 0
    print(f"Streaming trajectory at {args.speed}x speed. Press Ctrl+C to terminate.")
    try:
        while True:
            loop_count += 1
            print(f"--- Iteration #{loop_count} ---")

            for i in range(n_points):
                t_start = time.perf_counter()

                packet = pack_fdm_packet(
                    lat_rad=lat_rad[i],
                    lon_rad=lon_rad[i],
                    alt_m=alt_m[i],
                    phi_rad=phi_rad[i],
                    theta_rad=theta_rad[i],
                    psi_rad=psi_rad[i]
                )
                sock.sendto(packet, (args.host, args.port))

                dt = (times[i + 1] - times[i]) if (i < n_points - 1) else 0.01
                delay = (dt / args.speed) - (time.perf_counter() - t_start)
                if delay > 0:
                    time.sleep(delay)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nReplay terminated by user.")
    finally:
        sock.close()
        fg_proc.send_signal(signal.SIGTERM)
        fg_proc.wait()
        print("FlightGear closed cleanly.")

if __name__ == "__main__":
    main()