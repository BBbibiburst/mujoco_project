"""
场景构建器.

将基础场景构建逻辑（地板、墙壁、灯光、桌子）从基类中抽离，
使基类保持精简，同时便于子类替换或扩展场景。
"""

from pathlib import Path
from typing import Optional, Tuple
import mujoco
import numpy as np

from .config import DefaultTextures
from .env_config import SceneConfig


class BaseSceneBuilder:
    """
    构建 empty_arena 风格的基础场景.

    包含：天空盒、地板、墙壁、灯光、可选桌子。
    可被子类继承以替换或扩展。
    """

    def build(self, spec: mujoco.MjSpec, cfg: SceneConfig) -> float:
        """
        向 spec 添加基础场景元素.

        Args:
            spec: 未编译的 MjSpec。
            cfg:  场景外观配置。

        Returns:
            table_height: 桌面 Z 高度（无桌子时返回 0.0）。
        """
        self._add_skybox(spec)
        self._add_floor(spec)
        self._add_walls(spec)
        self._add_lighting(spec)

        table_height = 0.0
        if cfg.has_table:
            self._add_table(spec, cfg)
            table_height = cfg.table_pos[2]

        return table_height

    # ====================== 内部方法 ======================

    def _add_skybox(self, spec: mujoco.MjSpec) -> None:
        sky_tex = spec.add_texture()
        sky_tex.name = "skybox"
        sky_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky_tex.width = 256
        sky_tex.height = 256
        sky_tex.rgb1 = [0.9, 0.9, 1.0]
        sky_tex.rgb2 = [0.2, 0.3, 0.4]

    def _add_floor(self, spec: mujoco.MjSpec) -> None:
        floor_tex_path = Path(DefaultTextures.FLOOR)
        floor_material = None

        if floor_tex_path.exists():
            tex = spec.add_texture()
            tex.name = "texplane"
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.file = str(floor_tex_path)

            mat = spec.add_material()
            mat.name = "floorplane"
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            mat.texrepeat = [2, 2]
            mat.texuniform = True
            mat.reflectance = 0.01
            mat.shininess = 0.0
            mat.specular = 0.0
            floor_material = mat.name

        geom = spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3, 3, 0.125],
            pos=[0, 0, 0],
            condim=3,
            group=1,
        )
        if floor_material:
            geom.material = floor_material

    def _add_walls(self, spec: mujoco.MjSpec) -> None:
        wall_tex_path = Path(DefaultTextures.WALL)
        wall_material = None

        if wall_tex_path.exists():
            tex = spec.add_texture()
            tex.name = "tex-light-gray-plaster"
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.file = str(wall_tex_path)

            mat = spec.add_material()
            mat.name = "walls_mat"
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            mat.texrepeat = [3, 3]
            mat.texuniform = True
            mat.reflectance = 0.0
            mat.shininess = 0.1
            mat.specular = 0.1
            wall_material = mat.name

        walls = [
            ("wall_leftcorner_visual",  [-1.25,  2.25, 1.5], [0.6532815,  0.6532815,  0.2705981,  0.2705981],  [1.06, 1.5, 0.01]),
            ("wall_rightcorner_visual", [-1.25, -2.25, 1.5], [0.6532815,  0.6532815, -0.2705981, -0.2705981], [1.06, 1.5, 0.01]),
            ("wall_left_visual",        [ 1.25,  3.0,  1.5], [0.7071,     0.7071,     0,          0],         [1.75, 1.5, 0.01]),
            ("wall_right_visual",       [ 1.25, -3.0,  1.5], [0.7071,    -0.7071,     0,          0],         [1.75, 1.5, 0.01]),
            ("wall_rear_visual",        [-2.0,   0.0,  1.5], [0.5,        0.5,        0.5,        0.5],       [1.5,  1.5, 0.01]),
            ("wall_front_visual",       [ 3.0,   0.0,  1.5], [0.5,        0.5,       -0.5,       -0.5],       [3.0,  1.5, 0.01]),
        ]
        for name, pos, quat, size in walls:
            geom = spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos, quat=quat, size=size,
                contype=0, conaffinity=0, group=1,
            )
            if wall_material:
                geom.material = wall_material

    def _add_lighting(self, spec: mujoco.MjSpec) -> None:
        spec.worldbody.add_light(
            name="main_light",
            pos=[1.0, 1.0, 1.5],
            dir=[-0.2, -0.2, -1.0],
            specular=[0.3, 0.3, 0.3],
            type=1,
            castshadow=False,
        )

    def _add_table(self, spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
        table_size = np.array(cfg.table_size)
        half = table_size / 2.0
        table_center = np.array([
            cfg.table_pos[0],
            cfg.table_pos[1],
            cfg.table_pos[2] - half[2],
        ])

        table_body = spec.worldbody.add_body(name="table", pos=table_center.tolist())

        # 桌面材质
        surface_mat = self._make_texture_material(
            spec,
            tex_name="table_surface_tex",
            mat_name="table_surface_mat",
            tex_path=cfg.table_surface_texture,
            reflectance=0.0, shininess=0.1, specular=0.1,
        )

        # 碰撞几何体（透明）
        table_body.add_geom(
            name="table_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half.tolist(),
            friction=[1.0, 0.005, 0.0001],
            rgba=[0, 0, 0, 0],
        )

        # 视觉几何体
        visual = table_body.add_geom(
            name="table_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half.tolist(),
            contype=0, conaffinity=0,
        )
        if surface_mat:
            visual.material = surface_mat
        else:
            visual.rgba = list(cfg.table_surface_rgba)

        # 桌面 site
        table_body.add_site(name="table_top", pos=[0, 0, half[2]], size=[0.001], rgba=[0, 0, 0, 0])

        # 桌腿
        leg_mat = self._make_texture_material(
            spec,
            tex_name="table_leg_tex",
            mat_name="table_leg_mat",
            tex_path=cfg.table_leg_texture,
            reflectance=0.3, shininess=0.5, specular=0.5,
            texrepeat=[1, 1],
        )

        leg_radius = 0.025
        leg_height = (cfg.table_pos[2] - table_size[2]) / 2.0
        for i, (sx, sy) in enumerate([(1,1),(-1,1),(-1,-1),(1,-1)], start=1):
            leg = table_body.add_geom(
                name=f"table_leg{i}_visual",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[sx * (half[0] - 0.1), sy * (half[1] - 0.1), -leg_height],
                size=[leg_radius, leg_height],
                contype=0, conaffinity=0,
            )
            if leg_mat:
                leg.material = leg_mat
            else:
                leg.rgba = list(cfg.table_leg_rgba)

    @staticmethod
    def _make_texture_material(
        spec: mujoco.MjSpec,
        tex_name: str,
        mat_name: str,
        tex_path: Optional[str],
        reflectance: float = 0.0,
        shininess: float = 0.1,
        specular: float = 0.1,
        texrepeat: Optional[list] = None,
    ) -> Optional[str]:
        """添加纹理+材质，返回材质名称；tex_path 为 None 或文件不存在时返回 None."""
        if tex_path is None or not Path(tex_path).exists():
            return None

        tex = spec.add_texture()
        tex.name = tex_name
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = tex_path

        mat = spec.add_material()
        mat.name = mat_name
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        mat.reflectance = reflectance
        mat.shininess = shininess
        mat.specular = specular
        if texrepeat:
            mat.texrepeat = texrepeat

        return mat.name