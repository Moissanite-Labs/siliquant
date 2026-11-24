"""Read weights out of a GGUF model file and write a changed copy.

Uses the gguf package that ships with llama.cpp (pass its folder as
GGUF_PY, default ~/src/llama.cpp/gguf-py). Nothing is downloaded.

A changed tensor is stored as 16-bit floats, whatever it was rounded to, so
that llama.cpp reads exactly the values the rounding produced (to about four
digits). The file therefore does not shrink: the bit rate of an arm is
counted from its codebook, not from the file.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.modules.setdefault("yaml", types.ModuleType("yaml"))  # gguf imports it for model cards only
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/src/llama.cpp/gguf-py")))
import gguf  # noqa: E402


def open_model(path):
    return gguf.GGUFReader(path)


def tensor_float(t):
    """A tensor as float32 in NumPy order (the reverse of the order GGUF lists)."""
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
        return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32)
    return gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)


def write_copy(reader, out_path, replace):
    """Write a copy of the model with the tensors named in `replace` swapped for the given float arrays."""
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(out_path, arch=arch, endianess=reader.endianess)
    align = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    data = []
    for t in reader.tensors:
        if t.name in replace:
            arr = np.ascontiguousarray(replace[t.name])
            if arr.dtype == t.data.dtype and arr.shape == t.data.shape and arr.dtype == np.uint8:
                # Raw bytes in the tensor's own storage format, for edits that keep the format.
                writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, t.tensor_type)
                data.append(arr)
                continue
            if arr.dtype == np.float16:
                kind = gguf.GGMLQuantizationType.F16
            elif arr.dtype == np.float32:
                kind = gguf.GGMLQuantizationType.F32
            else:
                raise ValueError(f"{t.name}: replacement must be float16 or float32")
            if tuple(arr.shape) != tuple(int(d) for d in reversed(t.shape)):
                raise ValueError(f"{t.name}: shape {arr.shape} does not match {tuple(reversed(t.shape))}")
            writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, kind)
            data.append(arr)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            data.append(t.data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for arr in data:
        writer.write_tensor_data(arr, tensor_endianess=reader.endianess)
    writer.close()


def write_copy_streaming(reader, out_path, replace_files):
    """Like write_copy, but each replacement is a path to a .npy file loaded one tensor at
    a time. Byte-identical output to write_copy for the same arrays, while keeping only a
    single replacement matrix in memory, so 27B-class arm files can be written on a 64 GB host.
    """
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(out_path, arch=arch, endianess=reader.endianess)
    align = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    for t in reader.tensors:
        if t.name in replace_files:
            arr = np.load(replace_files[t.name], mmap_mode="r")
            if arr.dtype == np.float16:
                kind = gguf.GGMLQuantizationType.F16
            elif arr.dtype == np.float32:
                kind = gguf.GGMLQuantizationType.F32
            else:
                raise ValueError(f"{t.name}: replacement must be float16 or float32")
            if tuple(arr.shape) != tuple(int(d) for d in reversed(t.shape)):
                raise ValueError(f"{t.name}: shape {arr.shape} does not match {tuple(reversed(t.shape))}")
            writer.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, kind)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        if t.name in replace_files:
            arr = np.ascontiguousarray(np.load(replace_files[t.name]))
            writer.write_tensor_data(arr, tensor_endianess=reader.endianess)
            del arr
        else:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
