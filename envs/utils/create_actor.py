import sapien.core as sapien
import numpy as np
from pathlib import Path
from copy import deepcopy
import transforms3d as t3d
import sapien.physx as sapienp
import json
import os, re

from .actor_utils import Actor, ArticulationActor


class UnStableError(Exception):

    def __init__(self, msg):
        super().__init__(msg)


def preprocess(scene, pose: sapien.Pose) -> tuple[sapien.Scene, sapien.Pose]:
    """Add entity to scene. Add bias to z axis if scene is not sapien.Scene."""
    if isinstance(scene, sapien.Scene):
        return scene, pose
    else:
        return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)


# create box
def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    boxtype="default",
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
    )
    if boxtype == "default":
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0, 0.0],
                    [0.0, 0, 1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data)


# create spere
def create_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeSphere(radius=radius, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/textures/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeSphere(radius=radius, material=material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create cylinder
def create_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=scene.default_physical_material,
        ))

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=sapien.render.RenderMaterial(base_color=[*color[:3], 1]),
        ))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# AA-style battery as a single rigid body. Visual stays a cylinder; collision is
# the bounding box so a side-lying cell cannot roll forever on the table.
BATTERY_RADIUS = 0.02
BATTERY_HALF_LENGTH = 0.0275
BATTERY_MASS = 0.025
BATTERY_BODY_COLOR = [0.20, 0.55, 0.95, 1.0]
BATTERY_CAP_COLOR = [0.75, 0.75, 0.78, 1.0]
# SAPIEN cylinders extend along +X; rotate onto +Z to match the original URDF frame.
_BATTERY_CYLINDER_TO_Z = t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0)
BATTERY_MODEL_DATA = {
    "scale": 1.0,
    "contact_points_pose": [
        [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    ],
    "functional_matrix": [
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    ],
}


def create_battery(scene, pose: sapien.Pose, name="018_battery") -> Actor:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    friction = scene.create_physical_material(1.2, 1.0, 0.0)
    half_size = [BATTERY_RADIUS, BATTERY_RADIUS, BATTERY_HALF_LENGTH]
    rigid = sapien.physx.PhysxRigidDynamicComponent()
    rigid.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=friction))
    rigid.linear_damping = 0.05
    rigid.angular_damping = 0.4

    body_mat = sapien.render.RenderMaterial(base_color=BATTERY_BODY_COLOR)
    cap_mat = sapien.render.RenderMaterial(base_color=BATTERY_CAP_COLOR)
    render = sapien.render.RenderBodyComponent()
    body = sapien.render.RenderShapeCylinder(radius=BATTERY_RADIUS, half_length=0.025, material=body_mat)
    body.local_pose = sapien.Pose(q=_BATTERY_CYLINDER_TO_Z)
    pos_cap = sapien.render.RenderShapeCylinder(radius=0.01, half_length=0.002, material=cap_mat)
    pos_cap.local_pose = sapien.Pose(p=[0.0, 0.0, 0.026], q=_BATTERY_CYLINDER_TO_Z)
    neg_cap = sapien.render.RenderShapeCylinder(radius=0.01, half_length=0.001, material=cap_mat)
    neg_cap.local_pose = sapien.Pose(p=[0.0, 0.0, -0.026], q=_BATTERY_CYLINDER_TO_Z)
    render.attach(body)
    render.attach(pos_cap)
    render.attach(neg_cap)

    entity.add_component(rigid)
    entity.add_component(render)
    entity.set_pose(pose)
    scene.add_entity(entity)
    actor = Actor(entity, deepcopy(BATTERY_MODEL_DATA), mass=BATTERY_MASS)
    hx, hy, hz = (2.0 * BATTERY_RADIUS, 2.0 * BATTERY_RADIUS, 2.0 * BATTERY_HALF_LENGTH)
    for component in entity.get_components():
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            component.inertia = np.array(
                [
                    BATTERY_MASS / 12.0 * (hy * hy + hz * hz),
                    BATTERY_MASS / 12.0 * (hx * hx + hz * hz),
                    BATTERY_MASS / 12.0 * (hx * hx + hy * hy),
                ],
                dtype=np.float32,
            )
    return actor


# create box
def create_visual_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])))

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_table(
        scene,
        pose: sapien.Pose,
        length: float,
        width: float,
        height: float,
        thickness=0.1,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    # Tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])  # Center the tabletop at z=0
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(
        pose=tabletop_pose,
        half_size=tabletop_half_size,
        material=scene.default_physical_material,
    )

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
    else:
        builder.add_box_visual(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=color,
        )

    # Table legs (x4)
    leg_spacing = 0.1
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_spacing / 2)
            y = j * (width / 2 - leg_spacing / 2)
            table_leg_pose = sapien.Pose([x, y, -height / 2 - 0.002])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2 - 0.002]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    return table


# create obj model
def create_obj(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
        no_collision=False,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "textured.obj"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"textured{model_id}.obj"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if not no_collision:
        if convex == True:
            builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
        else:
            builder.add_nonconvex_collision_from_file(filename=str(file_name), scale=scale)

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


# create glb model
def create_glb(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "base.glb"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"base{model_id}.glb"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(file_name),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


def get_glb_or_obj_file(modeldir, model_id):
    modeldir = Path(modeldir)
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        if model_id is None:
            file = modeldir / "textured.obj"
        else:
            file = modeldir / f"textured{model_id}.obj"
    return file


def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=0,
) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets/objects") / modelname

    if model_id is None:
        json_file_path = modeldir / "model_data.json"
    else:
        json_file_path = modeldir / f"model_data{model_id}.json"

    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)

    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)

    if not collision_file.exists() or not visual_file.exists():
        print(modelname, "is not exist model file!")
        return None

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_name(modelname)
    mesh.set_pose(pose)
    return Actor(mesh, model_data)


# create urdf model
def create_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    json_file_path = modeldir / "model_data.json"
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        loader.scale = model_data["scale"][0]
    except:
        model_data = None

    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object: sapien.Articulation = loader.load(str(modeldir / "mobility.urdf"))

    object.set_root_pose(pose)
    object.set_name(modelname)
    return ArticulationActor(object, model_data)


def create_sapien_urdf_obj(
    scene,
    pose: sapien.Pose,
    modelname: str,
    scale=1.0,
    modelid: int = None,
    fix_root_link=False,
) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets") / "objects" / modelname
    if modelid is not None:
        model_list = [model for model in modeldir.iterdir() if model.is_dir() and model.name != "visual"]

        def extract_number(filename):
            match = re.search(r"\d+", filename.name)
            return int(match.group()) if match else 0

        model_list = sorted(model_list, key=extract_number)

        if modelid >= len(model_list):
            is_find = False
            for model in model_list:
                if modelid == int(model.name):
                    modeldir = model
                    is_find = True
                    break
            if not is_find:
                raise ValueError(f"modelid {modelid} is out of range for {modelname}.")
        else:
            modeldir = model_list[modelid]
    json_file = modeldir / "model_data.json"

    if json_file.exists():
        with open(json_file, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
        trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
    else:
        model_data = None
        trans_mat = np.eye(4)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object = loader.load_multiple(str(modeldir / "mobility.urdf"))[0][0]

    pose_mat = pose.to_transformation_matrix()
    pose = sapien.Pose(
        p=pose_mat[:3, 3] + trans_mat[:3, 3],
        q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
    )
    object.set_pose(pose)

    if model_data is not None:
        if "init_qpos" in model_data and len(model_data["init_qpos"]) > 0:
            object.set_qpos(np.array(model_data["init_qpos"]))
        if "mass" in model_data and len(model_data["mass"]) > 0:
            for link in object.get_links():
                link.set_mass(model_data["mass"].get(link.get_name(), 0.1))

        bounding_box_file = modeldir / "bounding_box.json"
        if bounding_box_file.exists():
            bounding_box = json.load(open(bounding_box_file, "r", encoding="utf-8"))
            model_data["extents"] = (np.array(bounding_box["max"]) - np.array(bounding_box["min"])).tolist()
    object.set_name(modelname)
    return ArticulationActor(object, model_data)
