#!/usr/bin/env python3

import json
import os
import sys
import urllib.request

# mapping from qcom-deb-images names to LAVA names
DEB_LAVA_MAP = {
    "lemans-evk": "iq-9075-evk",
    "qcs9100-ride-r3": "qcs9100-ride-sx",
}

# boards to skip (problematic)
SKIP_BOARDS = {
    # only one board in hyd lab, not passing health checks and timing out
    "qcs8300-ride",
}


def get_lava_device_types(lava_url):
    if not lava_url.endswith("/"):
        lava_url += "/"
    url = f"{lava_url}api/v0.2/devicetypes/"

    print(f"Fetching LAVA device types from {url}", file=sys.stderr)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = data.get("results", [])
            available = set()
            for device_type in results:
                available.add(device_type["name"])
                for alias in device_type["aliases"]:
                    available.add(alias)
            return available
    except Exception as e:
        print(f"Error fetching LAVA device types: {e}", file=sys.stderr)
        # fail if we can't query LAVA
        sys.exit(1)


def main():
    boards_json_str = os.environ.get("BOARDS_JSON", "[]")
    try:
        boards = json.loads(boards_json_str)
    except json.JSONDecodeError:
        print("Invalid BOARDS_JSON", file=sys.stderr)
        sys.exit(1)

    lava_url = "https://lava.infra.foundries.io"
    available_lava_device_types = get_lava_device_types(lava_url)

    print(
        f"Available LAVA types: {len(available_lava_device_types)} found",
        file=sys.stderr,
    )

    filtered_boards = []
    for board in boards:
        device_type = board.get("device_type")

        # determine LAVA name
        lava_device_type = DEB_LAVA_MAP.get(device_type, device_type)

        # check skip list
        if device_type in SKIP_BOARDS or lava_device_type in SKIP_BOARDS:
            print(f"Skipping {device_type}", file=sys.stderr)
            continue

        # check availability
        if lava_device_type in available_lava_device_types:
            print(
                f"Keeping {device_type} (mapped to {lava_device_type})",
                file=sys.stderr
            )
            # add the mapped name to the object for the job to use
            board["lava_device_type"] = lava_device_type
            filtered_boards.append(board)
        else:
            print(
                f"Skipping {device_type} (mapped to {lava_device_type}) - not found in LAVA",
                file=sys.stderr,
            )

    print(json.dumps(filtered_boards))


if __name__ == "__main__":
    main()
