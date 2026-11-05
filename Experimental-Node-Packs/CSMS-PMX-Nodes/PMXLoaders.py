# CyberSleuth-MS/Spare-ComfyUI-Nodes/Experimental-Node-Packs/CSMS-PMX-Nodes/PMXLoaders.py
# Status: Completely Operational
# TODO: 1. Add a Bool to handle the verbose debugging. 2. Fix annoying word wrapping formatting.
import struct
import os
import uuid
import shutil
from aiohttp import web
from server import PromptServer
import math
import numpy as np
import torch
import folder_paths
import hashlib
from comfy_api.latest import io
try:
    import pypmxvmd
except ImportError:
    pypmxvmd = None


PMX_MODEL = io.Custom("PMX_MODEL")


@PromptServer.instance.routes.post("/pmx/upload")
async def upload_pmx_asset(request):
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pmx_assets")
    os.makedirs(asset_root, exist_ok=True)
    temporary_directory = os.path.join(asset_root, f"_upload_{uuid.uuid4().hex}")
    os.makedirs(temporary_directory, exist_ok=True)
    try:
        reader = await request.multipart()
        pmx_path = None
        
        while True:
            part = await reader.next()
            
            if part is None:
                break
            if part.name == "pmx_path":
                pmx_path = await part.text()
                continue
            if part.name != "files":
                continue
            relative_path = part.filename
            if not relative_path:
                continue
                
            relative_path = relative_path.replace("\\", "/")
            relative_path = os.path.normpath(relative_path)

            if (
                relative_path.startswith("..")
                or os.path.isabs(relative_path)
                or os.path.basename(relative_path) == ".."
            ):
                raise ValueError("Invalid PMX asset path.")

            destination = os.path.abspath(os.path.join(temporary_directory, relative_path))
            
            if os.path.commonpath((temporary_directory, destination)) != os.path.abspath(temporary_directory):
                raise ValueError("PMX asset path escapes asset directory.")
                
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            with open(destination, "wb") as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            
        if not pmx_path:
            raise ValueError("No PMX path was supplied.")
        pmx_path = pmx_path.replace("\\", "/")
        pmx_path = os.path.normpath(pmx_path)

        if (pmx_path.startswith("..") or os.path.isabs(pmx_path)):
            raise ValueError("Invalid PMX path.")

        temporary_pmx_path = os.path.abspath(os.path.join(temporary_directory,pmx_path))

        if os.path.commonpath((temporary_directory, temporary_pmx_path)) != os.path.abspath(temporary_directory):
            raise ValueError("PMX path escapes asset directory.")

        if not os.path.isfile(temporary_pmx_path):
            raise FileNotFoundError("Uploaded PMX file was not found.")

        md5 = hashlib.md5()

        with open(temporary_pmx_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                md5.update(chunk)
        asset_hash = md5.hexdigest()
        asset_directory = os.path.join(asset_root, asset_hash)

        if os.path.exists(asset_directory):
            shutil.rmtree(temporary_directory, ignore_errors=True)

            final_pmx_path = os.path.abspath(
                os.path.join(
                    asset_directory,
                    pmx_path,
                )
            )

            if not os.path.isfile(final_pmx_path):
                raise FileNotFoundError("Existing PMX asset does not contain the PMX file.")

            return web.json_response(
                {
                    "pmx_path": final_pmx_path,
                    "asset_hash": asset_hash,
                }
            )
            print(
                "[PMX UPLOAD] FINAL PMX PATH:",
                repr(final_pmx_path),
                flush=True,
            )

            print(
                "[PMX UPLOAD] ASSET DIRECTORY:",
                repr(asset_directory),
                flush=True,
            )

            print(
                "[PMX UPLOAD] PMX EXISTS:",
                os.path.isfile(final_pmx_path),
                flush=True,
            )

        os.replace(
            temporary_directory,
            asset_directory,
        )

        final_pmx_path = os.path.abspath(
            os.path.join(
                asset_directory,
                pmx_path,
            )
        )

        return web.json_response(
            {
                "pmx_path": final_pmx_path,
                "asset_hash": asset_hash,
            }
        )
        print(
            "[PMX UPLOAD] FINAL PMX PATH:",
            repr(final_pmx_path),
            flush=True,
        )
        print(
            "[PMX UPLOAD] ASSET DIRECTORY:",
            repr(asset_directory),
            flush=True,
        )
        print(
            "[PMX UPLOAD] PMX EXISTS:",
            os.path.isfile(final_pmx_path),
            flush=True,
        )
    except Exception:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )
        raise


@PromptServer.instance.routes.post("/pmx/debug")
async def pmx_debug(request):
    data = await request.json()
    print(
        "\n[PMX JS DEBUG]",
        flush=True,
    )
    print(
        repr(data),
        flush=True,
    )
    return web.json_response(
        {"ok": True}
    )


class PMXLoader(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PMXLoader",
            display_name="PMX File",
            category="Utility",
            inputs=[
                io.String.Input(
                    "pmx_model",
                    display_name="pmx_model",
                ),
            ],
            outputs=[
                PMX_MODEL.Output(
                    display_name="pmx_model_bones",
                ),
                PMX_MODEL.Output(
                    display_name="pmx_model_full",
                ),
            ],
        )

    @classmethod
    def execute(cls, pmx_model):
        if not pmx_model:
            raise ValueError("No file selected.")
        #path = folder_paths.get_annotated_filepath(pmx_model)
        if os.path.isabs(pmx_model):
            path = os.path.abspath(pmx_model)
        else:
            path = folder_paths.get_annotated_filepath(pmx_model)
        print(f"[PMXLoader] PMX path: '{path}'", flush=True)
        print(f"[PMXLoader] texture-aka-main_folder: '{os.path.dirname(path)}'", flush=True)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"File not found: {pmx_model}"
            )
        #bone_data = parse_pmx_bones(path)
        pmx_data = parse_pmx_bones(path)
        # PASS main folder/textures base input folder for _render_mesh in PMXRendererNodes.py, and so on
        # DO NOT REMOVE
        # NEVER REMOVE
        pmx_data["texture-aka-main_folder"] = os.path.dirname(path)
        # DO NOT REMOVE WHAT IS ABOVE THIS LINE. EVER.
        print("\n" + "=" * 80)
        print("DIRECT PMX BONE PARSER")
        print("=" * 80)
        print(
            f"Found {len(pmx_data['bones'])} bones"
        )
        for bone in pmx_data["bones"][:20]:
            print(
                bone["index"],
                bone["name_jp"],
                "parent:",
                bone["parent_index"],
            )
        print("=" * 80 + "\n")
        #
        #return io.NodeOutput(pmx_data)
        pmx_model_bones = {
            "version": pmx_data["version"],
            "encoding": pmx_data["encoding"],
            "bone_index_size": pmx_data["bone_index_size"],
            "bones": pmx_data["bones"],
            "bone_section_offset": pmx_data["bone_section_offset"],
        }
        pmx_model_full = pmx_data

        return io.NodeOutput(pmx_model_bones, pmx_model_full)


class PMXBoneHierarchy(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PMXBoneHierarchy",
            display_name="PMX Bone Hierarchy",
            category="Utility",
            inputs=[
                PMX_MODEL.Input(
                    "pmx_model",
                    display_name="pmx_model",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="bone_hierarchy",
                ),
            ],
        )

    @classmethod
    def execute(cls, pmx_model):
        bones = pmx_model["bones"]

        if not bones:
            return io.NodeOutput(
                "No bones found in PMX model."
            )

        children = [[] for _ in bones]
        roots = []

        for bone in bones:
            index = bone["index"]
            parent_index = bone["parent_index"]

            if 0 <= parent_index < len(bones):
                children[parent_index].append(index)
            else:
                roots.append(index)

        lines = []

        def get_name(index):
            bone = bones[index]

            if bone["name_jp"]:
                return bone["name_jp"]

            if bone["name_en"]:
                return bone["name_en"]

            return f"Bone {index}"

        def append_bone(index, prefix, last):
            connector = "└── " if last else "├── "

            lines.append(
                prefix
                + connector
                + get_name(index)
            )

            child_list = children[index]

            for child_position, child_index in enumerate(
                child_list
            ):
                child_last = (
                    child_position == len(child_list) - 1
                )

                if last:
                    child_prefix = prefix + "    "
                else:
                    child_prefix = prefix + "│   "

                append_bone(
                    child_index,
                    child_prefix,
                    child_last,
                )

        for root_position, root_index in enumerate(roots):
            root_last = (
                root_position == len(roots) - 1
            )

            lines.append(get_name(root_index))

            for child_position, child_index in enumerate(
                children[root_index]
            ):
                child_last = (
                    child_position
                    == len(children[root_index]) - 1
                )

                append_bone(
                    child_index,
                    "",
                    child_last,
                )

            if not root_last:
                lines.append("")

        return io.NodeOutput(
            "\n".join(lines)
        )


def _pmx_read_text(data, offset, encoding):
    length = struct.unpack_from("<i", data, offset)[0]
    offset += 4

    raw = data[offset:offset + length]
    offset += length

    if encoding == 0:
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return text, offset


def _pmx_read_index(data, offset, size, signed=True):
    if size == 1:
        fmt = "<b" if signed else "<B"
    elif size == 2:
        fmt = "<h" if signed else "<H"
    elif size == 4:
        fmt = "<i" if signed else "<I"
    else:
        raise ValueError(
            f"Invalid PMX index size: {size}"
        )
    return struct.unpack_from(fmt, data, offset)[0], offset + size


def _pmx_read_vec3(data, offset):
    return (
        struct.unpack_from("<fff", data, offset),
        offset + 12,
    )


def parse_pmx_bones(path):
    with open(path, "rb") as f:
        data = f.read()

    offset = 0

    signature = data[offset:offset + 4]
    offset += 4

    if signature != b"PMX ":
        raise ValueError(f"Not a PMX file. Signature: {signature!r}")

    version = struct.unpack_from(
        "<f",
        data,
        offset,
    )[0]
    offset += 4

    globals_count = data[offset]
    offset += 1

    globals_data = data[
        offset:offset + globals_count
    ]
    offset += globals_count

    if globals_count < 8:
        raise ValueError(f"Invalid PMX globals size: {globals_count}")

    encoding = globals_data[0]
    additional_uv_count = globals_data[1]
    vertex_index_size = globals_data[2]
    texture_index_size = globals_data[3]
    material_index_size = globals_data[4]
    bone_index_size = globals_data[5]
    morph_index_size = globals_data[6]
    rigidbody_index_size = globals_data[7]

    # Model names/comments
    _, offset = _pmx_read_text(
        data,
        offset,
        encoding,
    )
    _, offset = _pmx_read_text(
        data,
        offset,
        encoding,
    )
    _, offset = _pmx_read_text(
        data,
        offset,
        encoding,
    )
    _, offset = _pmx_read_text(
        data,
        offset,
        encoding,
    )

    # Vertices
    vertex_count = struct.unpack_from(
        "<i",
        data,
        offset,
    )[0]
    offset += 4

    vertices = []
    vertex_weights = []

    for vertex_index in range(vertex_count):
        # Position
        position, offset = _pmx_read_vec3(
            data,
            offset,
        )

        # Normal
        normal, offset = _pmx_read_vec3(
            data,
            offset,
        )

        # UV
        uv = struct.unpack_from(
            "<ff",
            data,
            offset,
        )
        offset += 8

        # Additional UV
        additional_uv = []

        for _ in range(additional_uv_count):
            value = struct.unpack_from(
                "<ffff",
                data,
                offset,
            )
            offset += 16
            additional_uv.append(value)

        deform_type = data[offset]
        offset += 1

        weights = []

        if deform_type == 0:
            # BDEF1
            bone_index, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )

            weights.append({
                "bone": bone_index,
                "weight": 1.0,
            })

        elif deform_type == 1:
            # BDEF2
            bone_a, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )

            bone_b, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )

            weight = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]
            offset += 4

            weights.append({
                "bone": bone_a,
                "weight": weight,
            })

            weights.append({
                "bone": bone_b,
                "weight": 1.0 - weight,
            })

        elif deform_type == 2:
            # BDEF4
            bone_indices = []

            for _ in range(4):
                bone_index, offset = _pmx_read_index(
                    data,
                    offset,
                    bone_index_size,
                    signed=True,
                )
                bone_indices.append(bone_index)

            bone_weights = struct.unpack_from(
                "<ffff",
                data,
                offset,
            )
            offset += 16

            for bone_index, weight in zip(
                bone_indices,
                bone_weights,
            ):
                weights.append({
                    "bone": bone_index,
                    "weight": weight,
                })

        elif deform_type == 3:
            # SDEF
            bone_a, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )

            bone_b, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )

            weight = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]
            offset += 4

            # SDEF parameters
            offset += 36

            weights.append({
                "bone": bone_a,
                "weight": weight,
            })

            weights.append({
                "bone": bone_b,
                "weight": 1.0 - weight,
            })

        elif deform_type == 4:
            # QDEF
            bone_indices = []

            for _ in range(4):
                bone_index, offset = _pmx_read_index(
                    data,
                    offset,
                    bone_index_size,
                    signed=True,
                )
                bone_indices.append(bone_index)

            bone_weights = struct.unpack_from(
                "<ffff",
                data,
                offset,
            )
            offset += 16

            for bone_index, weight in zip(
                bone_indices,
                bone_weights,
            ):
                weights.append({
                    "bone": bone_index,
                    "weight": weight,
                })

        else:
            raise ValueError(
                f"Unknown PMX vertex deform type: "
                f"{deform_type}"
            )

        # Edge scale
        edge_scale = struct.unpack_from(
            "<f",
            data,
            offset,
        )[0]
        offset += 4

        vertices.append({
            "index": vertex_index,
            "position": position,
            "normal": normal,
            "uv": uv,
            "additional_uv": additional_uv,
            "deform_type": deform_type,
            "edge_scale": edge_scale,
        })

        vertex_weights.append(weights)

    # Faces
    face_index_count = struct.unpack_from(
        "<i",
        data,
        offset,
    )[0]
    offset += 4

    face_indices = []

    for _ in range(face_index_count):
        vertex_index, offset = _pmx_read_index(
            data,
            offset,
            vertex_index_size,
            signed=False,
        )

        face_indices.append(vertex_index)

    faces = []

    for index in range(
        0,
        len(face_indices),
        3,
    ):
        faces.append([
            face_indices[index],
            face_indices[index + 1],
            face_indices[index + 2],
        ])

    # Textures
    texture_count = struct.unpack_from(
        "<i",
        data,
        offset,
    )[0]
    offset += 4

    textures = []

    pmx_directory = os.path.dirname(
        os.path.abspath(path)
    )

    print(
        f"[PMX] Texture count: {texture_count}, "
        f"offset={offset}"
    )

    for texture_index in range(texture_count):
        texture_path, offset = _pmx_read_text(
            data,
            offset,
            encoding,
        )

        texture_path = texture_path.replace(
            "\\",
            os.sep,
        )

        textures.append({
            "index": texture_index,
            "path": texture_path,
            "full_path": os.path.normpath(
                os.path.join(
                    pmx_directory,
                    texture_path,
                )
            ),
        })

    # Materials
    material_count = struct.unpack_from(
        "<i",
        data,
        offset,
    )[0]
    offset += 4

    materials = []

    face_start = 0

    print(
        f"[PMX] Material count: {material_count}, "
        f"offset={offset}"
    )

    for material_index in range(material_count):
        material_offset = offset

        # Names
        name_jp, offset = _pmx_read_text(
            data,
            offset,
            encoding,
        )

        name_en, offset = _pmx_read_text(
            data,
            offset,
            encoding,
        )

        # Diffuse
        diffuse = struct.unpack_from(
            "<ffff",
            data,
            offset,
        )
        offset += 16

        # Specular
        specular = struct.unpack_from(
            "<fff",
            data,
            offset,
        )
        offset += 12

        # Specular power
        specular_power = struct.unpack_from(
            "<f",
            data,
            offset,
        )[0]
        offset += 4

        # Ambient
        ambient = struct.unpack_from(
            "<fff",
            data,
            offset,
        )
        offset += 12

        # Drawing flags
        drawing_flags = data[offset]
        offset += 1

        # Edge color
        edge_color = struct.unpack_from(
            "<ffff",
            data,
            offset,
        )
        offset += 16

        # Edge size
        edge_size = struct.unpack_from(
            "<f",
            data,
            offset,
        )[0]
        offset += 4

        # Texture index
        texture_index, offset = _pmx_read_index(
            data,
            offset,
            texture_index_size,
            signed=True,
        )

        # Sphere texture index
        sphere_texture_index, offset = _pmx_read_index(
            data,
            offset,
            texture_index_size,
            signed=True,
        )

        # Sphere mode
        sphere_mode = data[offset]
        offset += 1
        # Toon flag
        toon_flag = data[offset]
        offset += 1

        if toon_flag == 0:
            toon_texture_index, offset = _pmx_read_index(
                data,
                offset,
                texture_index_size,
                signed=True,
            )
        elif toon_flag == 1:
            toon_texture_index = data[offset]
            offset += 1
        else:
            raise ValueError(
                f"Invalid PMX toon flag "
                f"{toon_flag} at material "
                f"{material_index}, offset={offset - 1}"
            )

        # Memo
        memo, offset = _pmx_read_text(
            data,
            offset,
            encoding,
        )

        # Face vertex count
        face_vertex_count = struct.unpack_from(
            "<i",
            data,
            offset,
        )[0]
        offset += 4

        if face_vertex_count < 0:
            raise ValueError(
                f"Invalid PMX face vertex count "
                f"{face_vertex_count} in material "
                f"{material_index}, "
                f"offset={offset - 4}"
            )

        if face_vertex_count % 3 != 0:
            raise ValueError(
                f"PMX material {material_index} has "
                f"face vertex count {face_vertex_count}, "
                f"which is not divisible by 3."
            )

        face_count = face_vertex_count // 3

        materials.append({
            "index": material_index,
            "name_jp": name_jp,
            "name_en": name_en,
            "diffuse": diffuse,
            "specular": specular,
            "specular_power": specular_power,
            "ambient": ambient,
            "drawing_flags": drawing_flags,
            "edge_color": edge_color,
            "edge_size": edge_size,
            "texture_index": texture_index,
            "sphere_texture_index": sphere_texture_index,
            "sphere_mode": sphere_mode,
            "toon_flag": toon_flag,
            "toon_texture_index": toon_texture_index,
            "memo": memo,
            "face_start": face_start,
            "face_count": face_count,
        })

        face_start += face_count

        print(
            f"[PMX] Material {material_index}: "
            f"{name_jp!r}, "
            f"texture={texture_index}, "
            f"faces={face_count}, "
            f"offset={material_offset}->{offset}"
        )
    # =========================================================
    # BONE SECTION
    # =========================================================
    print("[PMX DEBUG] bytes before bone count:", data[offset:offset + 16].hex(" "))
    
    bone_count = struct.unpack_from(
        "<i",
        data,
        offset,
    )[0]
    offset += 4

    print(f"[PMX] Bone section found: {bone_count} bones")

    bones = []

    for bone_index in range(bone_count):
        name_jp, offset = _pmx_read_text(
            data,
            offset,
            encoding,
        )

        name_en, offset = _pmx_read_text(
            data,
            offset,
            encoding,
        )

        position, offset = _pmx_read_vec3(
            data,
            offset,
        )

        parent_index, offset = _pmx_read_index(
            data,
            offset,
            bone_index_size,
            signed=True,
        )

        deform_layer = struct.unpack_from(
            "<i",
            data,
            offset,
        )[0]
        offset += 4

        flags = struct.unpack_from(
            "<H",
            data,
            offset,
        )[0]
        offset += 2

        bone = {
            "index": bone_index,
            "name_jp": name_jp,
            "name_en": name_en,
            "position": position,
            "parent_index": parent_index,
            "deform_layer": deform_layer,
            "flags": flags,
        }

        # Bit 0: tail is a bone index
        if flags & 0x0001:
            tail_index, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )
            bone["tail_index"] = tail_index
            bone["tail_position"] = None
        else:
            tail_position, offset = _pmx_read_vec3(
                data,
                offset,
            )
            bone["tail_index"] = None
            bone["tail_position"] = tail_position

        # Bits 8/9: inherit rotation/translation
        if flags & 0x0300:
            inherit_parent, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )
            inherit_weight = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]
            offset += 4
            bone["inherit_parent"] = inherit_parent
            bone["inherit_weight"] = inherit_weight
        else:
            bone["inherit_parent"] = -1
            bone["inherit_weight"] = 0.0

        # Bit 10: fixed axis
        if flags & 0x0400:
            fixed_axis, offset = _pmx_read_vec3(
                data,
                offset,
            )
            bone["fixed_axis"] = fixed_axis
        else:
            bone["fixed_axis"] = None

        # Bit 11: local coordinate
        if flags & 0x0800:
            local_x, offset = _pmx_read_vec3(
                data,
                offset,
            )
            local_z, offset = _pmx_read_vec3(
                data,
                offset,
            )
            bone["local_x"] = local_x
            bone["local_z"] = local_z
        else:
            bone["local_x"] = None
            bone["local_z"] = None

        # Bit 13: external parent
        if flags & 0x2000:
            external_parent = struct.unpack_from(
                "<i",
                data,
                offset,
            )[0]
            offset += 4
            bone["external_parent"] = external_parent
        else:
            bone["external_parent"] = None

        # Bit 5: IK
        if flags & 0x0020:
            ik_target, offset = _pmx_read_index(
                data,
                offset,
                bone_index_size,
                signed=True,
            )
            ik_iterations = struct.unpack_from(
                "<i",
                data,
                offset,
            )[0]
            offset += 4
          
            ik_limit = struct.unpack_from(
                "<f",
                data,
                offset,
            )[0]
            offset += 4
          
            ik_link_count = struct.unpack_from(
                "<i",
                data,
                offset,
            )[0]
            offset += 4

            ik_links = []

            for _ in range(ik_link_count):
                link_bone, offset = _pmx_read_index(
                    data,
                    offset,
                    bone_index_size,
                    signed=True,
                )

                has_limit = data[offset]
                offset += 1

                link = {
                    "bone_index": link_bone,
                    "has_limit": bool(has_limit),
                }

                if has_limit:
                    lower, offset = _pmx_read_vec3(
                        data,
                        offset,
                    )
                    upper, offset = _pmx_read_vec3(
                        data,
                        offset,
                    )
                    link["lower_limit"] = lower
                    link["upper_limit"] = upper
                ik_links.append(link)
            bone["ik"] = {
                "target": ik_target,
                "iterations": ik_iterations,
                "limit_radian": ik_limit,
                "links": ik_links,
            }
        else:
            bone["ik"] = None

        bones.append(bone)
    print(f"[PMX] Successfully parsed {len(bones)} bones")
    # for bone in bones[:10]:
        # print(
            # f"[PMX] "
            # f"{bone['index']:4d} "
            # f"{bone['name_jp']!r} "
            # f"parent={bone['parent_index']}"
        # )

    return {
        "version": version,
        "encoding": encoding,
        "bone_index_size": bone_index_size,
        "pmx_path": os.path.abspath(path),
        "pmx_directory": pmx_directory,
        "vertices": vertices,
        "faces": faces,
        "vertex_weights": vertex_weights,
        "textures": textures,
        "materials": materials,
        "bones": bones,
        "bone_section_offset": offset,
    }
