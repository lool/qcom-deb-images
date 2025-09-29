#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""
Create disk images from Qualcomm flash XML files.

This script takes a bunch of XML files as input (like qdl) and creates disk-ufs.img
or disk-sdcard.img files that can be used with reassemble-disk-image.

The script validates that all XML files have the same physical_partition_number,
as it would make no sense to generate a single disk image for separate hardware partitions.

Usage:
    create-disk-image.py [options] <xml_file1> <xml_file2> ...
    create-disk-image.py [options] rawprogram0.xml patch0.xml
    create-disk-image.py [options] *.xml

Options:
    --output <filename>     Output disk image filename (required)
    --sector-size <size>    Sector size in bytes (default: from XML files)
    --image-size <size>     Image size with units (e.g., 8GiB, 1024MiB, 512MB) (default: calculated minimum size)
    --help                  Show this help message
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import struct
import re
import zlib
from typing import Union
from pathlib import Path


def parse_size_with_units(size_str):
    """Parse size string with units (e.g., '8GiB', '1024MiB', '512MB') and return bytes."""
    if isinstance(size_str, int):
        return size_str

    size_str = size_str.strip().upper()

    # Define unit multipliers (binary and decimal)
    units = {
        "B": 1,
        "KB": 1000,
        "KIB": 1024,
        "MB": 1000**2,
        "MIB": 1024**2,
        "GB": 1000**3,
        "GIB": 1024**3,
        "TB": 1000**4,
        "TIB": 1024**4,
    }

    # Extract number and unit
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([A-Z]+)?$", size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    number_str, unit = match.groups()
    number = float(number_str)

    if unit is None:
        # No unit specified, assume bytes
        return int(number)

    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")

    return int(number * units[unit])


class DiskImageCreator:
    def __init__(self, sector_size=None, output_file=None, image_size=None):
        self.sector_size = sector_size
        self.output_file = output_file
        self.image_size = image_size
        self.partitions = []
        self.patches = []
        self.xml_files = []
        self.warnings = []

    def detect_sector_size_from_xml(self):
        """Detect sector size from XML files and validate consistency."""
        detected_sector_size = None

        for xml_file in self.xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Look for sector size in XML (only check rawprogram files)
                if root.findall(".//program"):
                    for elem in root.iter():
                        sector_size_str = elem.get("SECTOR_SIZE_IN_BYTES")
                        if sector_size_str:
                            sector_size = int(sector_size_str)
                            if detected_sector_size is None:
                                detected_sector_size = sector_size
                            elif detected_sector_size != sector_size:
                                print(
                                    f"Error: Inconsistent sector sizes found in XML files: {detected_sector_size} vs {sector_size}"
                                )
                                return None
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                return None

        return detected_sector_size

    _dyn_re = re.compile(r"^NUM_DISK_SECTORS-(\d+)\.?$")

    def _eval_sector(self, val: Union[int, str], total_sectors: int) -> int:
        """Return absolute LBA from int or 'NUM_DISK_SECTORS-X'."""
        if isinstance(val, int):
            return val
        s = str(val).strip()
        m = self._dyn_re.match(s)
        if m:
            return total_sectors - int(m.group(1))
        return int(s)

    def warn(self, msg: str):
        print(f"WARNING: {msg}")
        self.warnings.append(msg)

    def _ext4_required_sectors(self, path: Path) -> int:
        """Return required sectors for an ext4 filesystem image, or 0 if not ext4.
        Reads the ext4 superblock at offset 1024, uses 32-bit block count and block size.
        """
        try:
            with open(path, "rb") as f:
                data = f.read(
                    4096 + 1024
                )  # ensure we have the superblock region
            if len(data) < 1024 + 0x3A + 2:
                return 0
            sb = memoryview(data)[1024:]  # superblock starts at offset 1024
            # Magic at offset 0x38..0x39 must be 0xEF53
            magic = int.from_bytes(sb[0x38:0x3A], "little")
            if magic != 0xEF53:
                return 0
            # Block size = 1024 << s_log_block_size (offset 0x18..0x1B)
            log_bs = int.from_bytes(sb[0x18:0x1C], "little")
            block_size = 1024 << log_bs
            # Blocks count (low 32-bit) at offset 0x04..0x07
            blocks_lo = int.from_bytes(sb[0x04:0x08], "little")
            # For most images this fits in 32-bit; ignore high 32-bit for simplicity
            fs_bytes = blocks_lo * block_size
            # Convert to sectors of current disk sector_size
            return (fs_bytes + self.sector_size - 1) // self.sector_size
        except Exception:
            return 0

    CHUNK_SIZE = 1024 * 1024

    def bytes_to_sectors(self, n: int) -> int:
        """Ceil-divide bytes to sectors using current sector_size."""
        return (n + self.sector_size - 1) // self.sector_size

    def compute_tail_reserve(self) -> int:
        """Return maximum M from any dynamic end-relative anchors NUM_DISK_SECTORS-M in patches (including CRC regions)."""
        tail = 0
        for p in self.patches:
            ps = str(p.get("start_sector", "")).strip()
            m = self._dyn_re.match(ps) if isinstance(ps, str) else None
            if m:
                try:
                    tail = max(tail, int(m.group(1)))
                except ValueError:
                    pass
            val = p.get("value", "")
            if isinstance(val, str) and val.startswith("CRC32("):
                inner = val[6:-1]
                parts = inner.split(",", 1)
                if len(parts) == 2:
                    crc_start = parts[0].strip()
                    m2 = self._dyn_re.match(crc_start)
                    if m2:
                        try:
                            tail = max(tail, int(m2.group(1)))
                        except ValueError:
                            pass
        return tail

    def sectors_needed_for_partition(self, partition) -> int:
        """Compute sectors needed by a partition from XML sectors, file size, and ext4 logical size."""
        if partition["num_partition_sectors"] > 0:
            return partition["num_partition_sectors"]
        part_path = self.resolve_filename_path(partition)
        if not part_path or not part_path.exists():
            return 0
        file_size = part_path.stat().st_size
        if partition["file_sector_offset"] > 0:
            file_size -= partition["file_sector_offset"] * self.sector_size
        file_size = max(0, file_size)
        file_sectors = self.bytes_to_sectors(file_size)
        ext4_req = self._ext4_required_sectors(part_path)
        # Warn if ext4 logical size exceeds file content size (often indicates sparse/truncated image)
        if ext4_req > file_sectors and file_sectors > 0:
            self.warn(
                f"Partition '{partition.get('label','')}' ext4 logical size ({ext4_req} sectors) "
                f"exceeds file content size ({file_sectors} sectors)."
            )
        # Warn if size_in_KB presents a different expectation than data-derived size
        size_kb = partition.get("size_in_KB", "")
        try:
            size_kb_val = float(size_kb)
            if size_kb_val > 0:
                kib_bytes = int(size_kb_val * 1024)
                kib_sectors = self.bytes_to_sectors(kib_bytes)
                est = max(file_sectors, ext4_req)
                if kib_sectors != est:
                    self.warn(
                        f"Partition '{partition.get('label','')}' size_in_KB={size_kb} (sectors={kib_sectors}) "
                        f"differs from data-derived size (sectors={est})."
                    )
        except Exception:
            pass
        return max(file_sectors, ext4_req)

    CHUNK_SIZE = 1024 * 1024

    def bytes_to_sectors(self, n: int) -> int:
        """Ceil-divide bytes to sectors using current sector_size."""
        return (n + self.sector_size - 1) // self.sector_size

    def compute_tail_reserve(self) -> int:
        """Return maximum M from any dynamic end-relative anchors NUM_DISK_SECTORS-M in patches (including CRC regions)."""
        tail = 0
        for p in self.patches:
            ps = str(p.get("start_sector", "")).strip()
            m = self._dyn_re.match(ps) if isinstance(ps, str) else None
            if m:
                try:
                    tail = max(tail, int(m.group(1)))
                except ValueError:
                    pass
            val = p.get("value", "")
            if isinstance(val, str) and val.startswith("CRC32("):
                inner = val[6:-1]
                parts = inner.split(",", 1)
                if len(parts) == 2:
                    crc_start = parts[0].strip()
                    m2 = self._dyn_re.match(crc_start)
                    if m2:
                        try:
                            tail = max(tail, int(m2.group(1)))
                        except ValueError:
                            pass
        return tail

    def sectors_needed_for_partition(self, partition) -> int:
        """Compute sectors needed by a partition from XML sectors, file size, and ext4 logical size."""
        if partition["num_partition_sectors"] > 0:
            return partition["num_partition_sectors"]
        part_path = self.resolve_filename_path(partition)
        if not part_path or not part_path.exists():
            return 0
        file_size = part_path.stat().st_size
        if partition["file_sector_offset"] > 0:
            file_size -= partition["file_sector_offset"] * self.sector_size
        file_size = max(0, file_size)
        file_sectors = self.bytes_to_sectors(file_size)
        ext4_req = self._ext4_required_sectors(part_path)
        # Warn if ext4 logical size exceeds file content size (often indicates sparse/truncated image)
        if ext4_req > file_sectors and file_sectors > 0:
            self.warn(
                f"Partition '{partition.get('label','')}' ext4 logical size ({ext4_req} sectors) "
                f"exceeds file content size ({file_sectors} sectors)."
            )
        # Warn if size_in_KB presents a different expectation than data-derived size
        size_kb = partition.get("size_in_KB", "")
        try:
            size_kb_val = float(size_kb)
            if size_kb_val > 0:
                kib_bytes = int(size_kb_val * 1024)
                kib_sectors = self.bytes_to_sectors(kib_bytes)
                est = max(file_sectors, ext4_req)
                if kib_sectors != est:
                    self.warn(
                        f"Partition '{partition.get('label','')}' size_in_KB={size_kb} (sectors={kib_sectors}) "
                        f"differs from data-derived size (sectors={est})."
                    )
        except Exception:
            pass
        return max(file_sectors, ext4_req)

    def _parse_crc32_args(self, value: str):
        """Parse 'CRC32(start,size)' and return (start_str, size_int) or None on error."""
        if not (isinstance(value, str) and value.startswith("CRC32(") and value.endswith(")")):
            return None
        inner = value[6:-1]
        parts = inner.split(",", 1)
        if len(parts) != 2:
            return None
        start_str = parts[0].strip()
        try:
            size_int = int(parts[1].strip())
        except ValueError:
            return None
        return (start_str, size_int)

    def _compute_patch_value(self, value: str, size_in_bytes: int, total_sectors: int):
        """Compute byte value to write for a patch. Returns None for CRC patches (handled in second pass)."""
        if value in ("NUM_DISK_SECTORS-6.", "NUM_DISK_SECTORS-6"):
            return (total_sectors - 6).to_bytes(8, "little")
        if value in ("NUM_DISK_SECTORS-5.", "NUM_DISK_SECTORS-5"):
            return (total_sectors - 5).to_bytes(8, "little")
        if value in ("NUM_DISK_SECTORS-1.", "NUM_DISK_SECTORS-1"):
            return (total_sectors - 1).to_bytes(8, "little")
        if isinstance(value, str) and value.startswith("CRC32("):
            return None
        if value == "0":
            return (0).to_bytes(size_in_bytes, "little")
        try:
            return int(value).to_bytes(size_in_bytes, "little")
        except Exception:
            print(f"Skipping patch with unsupported value: {value}")
            return b""

    def parse_xml_file(self, xml_file):
        """Parse XML file to extract partition and patch information."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            xml_dir = Path(xml_file).parent

            # Parse rawprogram elements
            for program in root.findall(".//program"):
                # Handle dynamic sector values
                start_sector_str = program.get("start_sector", "0")
                try:
                    start_sector = int(start_sector_str)
                except ValueError:
                    # Keep as string for dynamic values like "NUM_DISK_SECTORS-5."
                    start_sector = start_sector_str

                partition = {
                    "label": program.get("label", ""),
                    "filename": program.get("filename", ""),
                    "xml_dir": xml_dir,  # Store the directory of the XML file
                    "start_sector": start_sector,
                    "num_partition_sectors": int(
                        program.get("num_partition_sectors", "0")
                    ),
                    "physical_partition_number": int(
                        program.get("physical_partition_number", "0")
                    ),
                    "size_in_KB": program.get("size_in_KB", ""),
                    "file_sector_offset": int(
                        program.get("file_sector_offset", "0")
                    ),
                    "sparse": program.get("sparse", "false").lower() == "true",
                    "partofsingleimage": program.get(
                        "partofsingleimage", "false"
                    ).lower()
                    == "true",
                }
                self.partitions.append(partition)

            # Parse patch elements
            for patch in root.findall(".//patch"):
                patch_info = {
                    "filename": patch.get("filename", ""),
                    "start_sector": patch.get("start_sector", "0"),
                    "size_in_bytes": int(patch.get("size_in_bytes", "0")),
                    "physical_partition_number": int(
                        patch.get("physical_partition_number", "0")
                    ),
                    "byte_offset": int(patch.get("byte_offset", "0")),
                    "value": patch.get("value", ""),
                    "what": patch.get("what", ""),
                }
                self.patches.append(patch_info)

        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
            return False
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return False

        return True

    def calculate_disk_size(self):
        """Calculate the total disk size needed, accounting for dynamic end-relative LBAs in partitions and patches."""
        static_end = (
            0  # minimum sectors required by statically addressed content
        )
        dynamic_min = (
            0  # minimum sectors required to satisfy NUM_DISK_SECTORS-N anchors
        )

        # Helper to ceil-divide bytes to sectors
        def bytes_to_sectors(n: int) -> int:
            return (n + self.sector_size - 1) // self.sector_size

        # Compute tail reserve implied by dynamic end-relative anchors in patches (e.g., backup GPT/header)
        tail_reserve = 0
        for p in self.patches:
            # dynamic write anchor
            ps = str(p.get("start_sector", "")).strip()
            m = self._dyn_re.match(ps) if isinstance(ps, str) else None
            if m:
                try:
                    tail_reserve = max(tail_reserve, int(m.group(1)))
                except ValueError:
                    pass
            # dynamic CRC region anchor
            val = p.get("value", "")
            if isinstance(val, str) and val.startswith("CRC32("):
                inner = val[6:-1]
                parts = inner.split(",", 1)
                if len(parts) == 2:
                    crc_start = parts[0].strip()
                    m2 = self._dyn_re.match(crc_start)
                    if m2:
                        try:
                            tail_reserve = max(tail_reserve, int(m2.group(1)))
                        except ValueError:
                            pass

        # Partitions: compute sectors needed and constraints from start_sector
        dynamic_sum_min = 0  # minimum sectors to satisfy static-start partitions that extend to end
        for partition in self.partitions:
            # Determine sectors_needed via helper
            sectors_needed = self.sectors_needed_for_partition(partition)

            start = partition["start_sector"]
            if isinstance(start, int):
                static_end = max(static_end, start + sectors_needed)
                # If this partition grows to the end (no explicit sector count), account for tail reservation
                if (
                    partition["num_partition_sectors"] == 0
                    and sectors_needed > 0
                ):
                    dynamic_sum_min = max(
                        dynamic_sum_min, start + sectors_needed + tail_reserve
                    )
            else:
                s = str(start).strip()
                m = self._dyn_re.match(s)
                if m:
                    # NUM_DISK_SECTORS-M anchors require at least M + sectors_needed total sectors
                    M = int(m.group(1))
                    dynamic_min = max(dynamic_min, M + sectors_needed)
                else:
                    # Fallback if an arbitrary numeric string
                    try:
                        static_end = max(static_end, int(s) + sectors_needed)
                    except ValueError:
                        # Unknown format; ignore for sizing
                        pass

        # Patches: account for both the write location span and CRC regions
        for patch in self.patches:
            # Span of the write starting at start_sector + byte_offset, of size size_in_bytes
            write_span_sectors = bytes_to_sectors(
                patch["byte_offset"] + patch["size_in_bytes"]
            )
            pstart = patch["start_sector"]
            if isinstance(pstart, int):
                static_end = max(static_end, pstart + write_span_sectors)
            else:
                ps = str(pstart).strip()
                pm = self._dyn_re.match(ps)
                if pm:
                    dynamic_min = max(
                        dynamic_min, int(pm.group(1)) + write_span_sectors
                    )
                else:
                    try:
                        static_end = max(
                            static_end, int(ps) + write_span_sectors
                        )
                    except ValueError:
                        pass

            # CRC region constraints (reading region may be dynamic relative to end)
            val = patch["value"]
            if isinstance(val, str) and val.startswith("CRC32("):
                inner = val[6:-1]
                parts = inner.split(",")
                if len(parts) == 2:
                    crc_start_str = parts[0].strip()
                    crc_size_str = parts[1].strip()
                    try:
                        crc_size = int(crc_size_str)
                        crc_span_sectors = bytes_to_sectors(crc_size)
                        mm = self._dyn_re.match(crc_start_str)
                        if mm:
                            dynamic_min = max(
                                dynamic_min,
                                int(mm.group(1)) + crc_span_sectors,
                            )
                        else:
                            try:
                                crc_start = int(crc_start_str)
                                static_end = max(
                                    static_end, crc_start + crc_span_sectors
                                )
                            except ValueError:
                                pass
                    except ValueError:
                        # Ignore invalid CRC size for sizing
                        pass

        total_sectors = max(static_end, dynamic_min, dynamic_sum_min)
        return total_sectors * self.sector_size

    def resolve_filename_path(self, partition):
        """Resolve the actual path for a partition filename relative to its XML file."""
        filename = partition["filename"]
        xml_dir = partition["xml_dir"]

        if not filename:
            return None

        # All filenames are relative to the XML file directory
        file_path = xml_dir / filename
        if file_path.exists():
            return file_path

        self.warn(f"Partition file {filename} not found relative to {xml_dir}")
        return None

    def calculate_crc32(self, disk_file, start_sector, size_bytes):
        """Calculate CRC32 for a region of the disk.
        If the region looks like a GPT header (starts with 'EFI PART'), we:
        - use HeaderSize (at offset 12) instead of the full sector, and
        - zero the HeaderCRC32 field (offset 16..19) before computing.
        """
        current_pos = disk_file.tell()
        try:
            seek_pos = start_sector * self.sector_size
            disk_file.seek(seek_pos)
            data = disk_file.read(size_bytes)
            if len(data) != size_bytes:
                self.warn(f"CRC calculation read {len(data)} bytes instead of {size_bytes} at sector {start_sector}")
            disk_file.seek(current_pos)
            # Detect GPT header by signature 'EFI PART' at offset 0
            if len(data) >= 24 and data[0:8] == b"EFI PART":
                header_size = struct.unpack_from("<I", data, 12)[0]
                header_size = max(
                    92, min(header_size, len(data))
                )  # guard; GPT header is at least 92 bytes
                hdr = bytearray(data[:header_size])
                # Zero header CRC field (offset 16..19) per spec
                hdr[16:20] = b"\x00\x00\x00\x00"
                return zlib.crc32(hdr) & 0xFFFFFFFF
            else:
                return zlib.crc32(data) & 0xFFFFFFFF
        except Exception as e:
            print(
                f"Error calculating CRC32 at sector {start_sector}, size {size_bytes}: {e}"
            )
            disk_file.seek(current_pos)
            raise

    def apply_patches(self, disk_file, disk_size):
        total_sectors = disk_size // self.sector_size
        disk_patches = 0
        crc_patches = []

        for patch in self.patches:
            try:
                start_lba = self._eval_sector(
                    patch["start_sector"], total_sectors
                )
            except Exception:
                print(
                    f"Skipping patch with invalid start sector: {patch['start_sector']}"
                )
                continue
            file_pos = start_lba * self.sector_size + patch["byte_offset"]
            value = patch["value"]
            patch_value = self._compute_patch_value(value, patch["size_in_bytes"], total_sectors)
            if patch_value is None:
                crc_patches.append((patch, start_lba, file_pos))
                continue
            if patch_value == b"":
                continue
            print(f"Applying patch at LBA {start_lba}, offset {patch['byte_offset']}: {patch['what']}")
            disk_file.seek(file_pos)
            disk_file.write(patch_value)
            disk_patches += 1

        # CRC pass
        for patch, start_lba, file_pos in crc_patches:
            args = self._parse_crc32_args(patch["value"])
            if not args:
                print(f"Skipping CRC patch with invalid params: {patch['value']}")
                continue
            crc_start_str, crc_size = args
            try:
                crc_start_lba = self._eval_sector(crc_start_str, total_sectors)
            except Exception:
                print(f"Skipping CRC patch with invalid start sector: {crc_start_str}")
                continue
            crc_val = self.calculate_crc32(disk_file, crc_start_lba, crc_size)
            disk_file.seek(file_pos)
            disk_file.write(crc_val.to_bytes(4, "little"))
            print(
                f"Applying CRC32 patch at LBA {start_lba}, offset {patch['byte_offset']} "
                f"(CRC=0x{crc_val:08x}): {patch['what']}"
            )
            disk_patches += 1

        print(f"Applied {disk_patches} patches total")

    def create_disk_image(self, xml_files):
        """Create the disk image file from XML files."""
        self.xml_files = xml_files

        # Detect sector size from XML files
        xml_sector_size = self.detect_sector_size_from_xml()

        # Validate sector size
        if self.sector_size is None:
            if xml_sector_size is None:
                print(
                    "Error: No sector size found in XML files and none provided via --sector-size"
                )
                return False
            self.sector_size = xml_sector_size
        else:
            # User provided sector size - validate against XML
            if (
                xml_sector_size is not None
                and xml_sector_size != self.sector_size
            ):
                print(
                    f"Warning: Provided sector size ({self.sector_size}) differs from XML files ({xml_sector_size})"
                )

        # Require output filename
        if self.output_file is None:
            print("Error: Output filename is required (use --output)")
            return False

        print(f"Creating disk image: {self.output_file}")
        print(f"Sector size: {self.sector_size} bytes")

        # Parse XML files
        rawprogram_count = 0
        patch_count = 0

        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                has_programs = bool(root.findall(".//program"))
                has_patches = bool(root.findall(".//patch"))

                if has_programs or has_patches:
                    print(f"Parsing XML file: {xml_file}")
                    if not self.parse_xml_file(xml_file):
                        return False

                    if has_programs:
                        rawprogram_count += 1
                    if has_patches:
                        patch_count += 1
                else:
                    print(
                        f"Warning: {xml_file} doesn't appear to be a rawprogram or patch file, skipping"
                    )
                    continue

            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                return False

        if rawprogram_count == 0:
            print("Error: No rawprogram XML files found")
            return False

        # Calculate minimum disk size needed
        min_disk_size = self.calculate_disk_size()
        print(
            f"Minimum disk size needed: {min_disk_size} bytes ({min_disk_size // (1024*1024)} MB)"
        )

        # Validate and set final disk size
        if self.image_size is None:
            disk_size = min_disk_size
        else:
            if self.image_size < min_disk_size:
                print(
                    f"Error: Specified image size ({self.image_size} bytes, {self.image_size // (1024*1024)} MB) is smaller than minimum required size ({min_disk_size} bytes, {min_disk_size // (1024*1024)} MB)"
                )
                return False
            disk_size = self.image_size

        print(
            f"Creating disk image with size: {disk_size} bytes ({disk_size // (1024*1024)} MB)"
        )

        # Create the disk image
        try:
            with open(self.output_file, "w+b") as disk_file:
                # Initialize with zeros
                print("Initializing disk image...")
                disk_file.seek(disk_size - 1)
                disk_file.write(b"\0")
                disk_file.seek(0)

                total_sectors = disk_size // self.sector_size

                # Compute tail reserve implied by dynamic end-relative anchors in patches
                tail_reserve = self.compute_tail_reserve()

                # Write partitions (static and dynamic start sectors)
                for partition in self.partitions:
                    if not partition["filename"]:
                        continue
                    part_file = self.resolve_filename_path(partition)
                    if not part_file:
                        continue

                    start_lba = self._eval_sector(
                        partition["start_sector"], total_sectors
                    )
                    file_pos = start_lba * self.sector_size
                    print(
                        f"Writing partition {partition['label']} from {partition['filename']} at LBA {start_lba}"
                    )
                    with open(part_file, "rb") as pfile:
                        # Compute source size after offset
                        file_size = part_file.stat().st_size
                        if partition["file_sector_offset"] > 0:
                            pfile.seek(
                                partition["file_sector_offset"]
                                * self.sector_size
                            )
                            file_size -= (
                                partition["file_sector_offset"]
                                * self.sector_size
                            )

                        # Validate filesystem fits declared or dynamic capacity
                        ext4_req_sectors = self._ext4_required_sectors(
                            part_file
                        )

                        # If partition length is declared, don't truncate silently and ensure ext4 fits
                        if partition["num_partition_sectors"] > 0:
                            capacity = (
                                partition["num_partition_sectors"]
                                * self.sector_size
                            )
                            # ext4 logical size must fit in capacity
                            if ext4_req_sectors * self.sector_size > capacity:
                                raise RuntimeError(
                                    f"Partition '{partition['label']}' ext4 filesystem size "
                                    f"({ext4_req_sectors * self.sector_size} B) exceeds capacity from XML "
                                    f"({capacity} B). Increase num_partition_sectors or enlarge image."
                                )
                            # file content must not exceed capacity either
                            if file_size > capacity:
                                raise RuntimeError(
                                    f"Partition '{partition['label']}' content ({file_size} B) exceeds "
                                    f"capacity from XML ({capacity} B). Shrink FS or enlarge image."
                                )
                        else:
                            # Dynamic-length: ensure ext4 logical size fits into available range to end minus tail reserve
                            available_sectors = (
                                total_sectors - tail_reserve - start_lba
                            )
                            if ext4_req_sectors > available_sectors:
                                raise RuntimeError(
                                    f"Partition '{partition['label']}' ext4 filesystem requires "
                                    f"{ext4_req_sectors} sectors, but only {available_sectors} sectors are "
                                    f"available before tail-reserved regions. Enlarge image size."
                                )

                        # Copy (full file by default; optional padding can be added if desired)
                        disk_file.seek(file_pos)
                        remaining = file_size
                        chunk = self.CHUNK_SIZE
                        while remaining:
                            buf = pfile.read(min(chunk, remaining))
                            if not buf:
                                break
                            disk_file.write(buf)
                            remaining -= len(buf)

                # Apply patches to update GPT and other dynamic data
                print("Applying patches...")
                self.apply_patches(disk_file, disk_size)

                print(f"Successfully created {self.output_file}")
                if self.warnings:
                    print(f"Completed with {len(self.warnings)} warning(s):")
                    for w in self.warnings:
                        print(f"WARNING: {w}")
                return True

        except Exception as e:
            print(f"Error creating disk image: {e}")
            return False


def validate_physical_partition_numbers(xml_files):
    """Validate that all XML files have the same physical_partition_number."""
    physical_partition_numbers = set()

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Check for physical_partition_number in program elements (rawprogram files)
            for program in root.findall(".//program"):
                ppn = program.get("physical_partition_number")
                if ppn is not None:
                    physical_partition_numbers.add(int(ppn))

            # Check for physical_partition_number in patch elements (patch files)
            for patch in root.findall(".//patch"):
                ppn = patch.get("physical_partition_number")
                if ppn is not None:
                    physical_partition_numbers.add(int(ppn))

        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
            return False
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return False

    if len(physical_partition_numbers) == 0:
        print("Warning: No physical_partition_number found in any XML files")
        return True
    elif len(physical_partition_numbers) == 1:
        ppn = list(physical_partition_numbers)[0]
        print(f"All XML files use physical_partition_number: {ppn}")
        return True
    else:
        print(
            f"Error: XML files contain different physical_partition_numbers: {sorted(physical_partition_numbers)}"
        )
        print(
            "It makes no sense to generate a single disk image for separate hardware partitions"
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create disk images from Qualcomm flash XML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output disk.img rawprogram0.xml patch0.xml
  %(prog)s --output my-disk.img rawprogram0.xml patch0.xml
  %(prog)s --output disk.img --image-size 8GiB rawprogram0.xml patch0.xml
  %(prog)s --output disk.img --image-size 1024MiB rawprogram0.xml patch0.xml
        """,
    )

    parser.add_argument(
        "xml_files",
        nargs="+",
        help="XML files (rawprogram*.xml and patch*.xml files)",
    )
    parser.add_argument(
        "--output", required=True, help="Output disk image filename (required)"
    )
    parser.add_argument(
        "--sector-size",
        type=int,
        help="Sector size in bytes (default: from XML files)",
    )
    parser.add_argument(
        "--image-size",
        type=parse_size_with_units,
        help="Image size with optional units (e.g., 8GiB, 1024MiB, 512MB) (default: calculated minimum size)",
    )

    args = parser.parse_args()

    # Validate that all XML files exist
    xml_files = []
    for xml_file in args.xml_files:
        if not os.path.isfile(xml_file):
            print(f"Error: {xml_file} not found")
            return 1
        xml_files.append(Path(xml_file))

    # Validate that all XML files have the same physical_partition_number
    if not validate_physical_partition_numbers(xml_files):
        return 1

    # Create disk image creator and process files directly
    creator = DiskImageCreator(
        sector_size=args.sector_size,
        output_file=args.output,
        image_size=args.image_size,
    )

    success = creator.create_disk_image(xml_files)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
